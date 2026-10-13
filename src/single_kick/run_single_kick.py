import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents if (p / "config.json").is_file())
_archive_sys.path.insert(0, str(_archive_root / "scripts"))
from archive_paths import ROOT, DATA, MCMC, EVIDENCE, run_dir

import matplotlib
matplotlib.use("Agg")
import numpy as np
import dynesty
from dynesty import plotting as dyplot
from multiprocessing import Pool
import contextlib
from dynesty.utils import resample_equal
# --- 1. 数据加载与筛选 ---
base_path = str(DATA) + "/"
v_obs_raw = np.loadtxt(base_path + 'vpec_lsr_median_47.txt')
p_obs_raw = np.loadtxt(base_path + 'period_2_new_47.txt')
# 载入速度误差上限和下限
v_err_low_raw = np.loadtxt(base_path + 'vpec_lsr_err_low_47.txt')
v_err_up_raw = np.loadtxt(base_path + 'vpec_lsr_err_up_47.txt') 

# 筛选 P > 0.1s 的数据
mask = p_obs_raw > 0.1
v_obs = v_obs_raw[mask]
p_obs = p_obs_raw[mask]
v_err_low = v_err_low_raw[mask]
v_err_up = v_err_up_raw[mask]

# 物理常量 
M_NS = 1.4  # M_sun
I_NS = 50.0 # M_sun * km^2

# --- 2. 处理非对称误差的采样 (用于似然计算中的边缘化) ---
# 为每个观测点预生成速度样本云，以考虑观测不确定性
N_SAMPLES = 100 
np.random.seed(42)
v_samples_cloud = np.zeros((len(v_obs), N_SAMPLES))

for i in range(len(v_obs)):
    # 使用分段正态分布模拟非对称误差
    # 产生标准正态分布随机数，然后根据正负号缩放误差
    z = np.random.normal(0, 1, N_SAMPLES)
    v_samples_cloud[i, :] = np.where(z < 0, 
                                     v_obs[i] + z * v_err_low[i], 
                                     v_obs[i] + z * v_err_up[i])
# 确保速度为正
v_samples_cloud = np.maximum(v_samples_cloud, 1.0)

# 将数据放入全局变量，方便多核调用
GLOBAL_V_CLOUD = v_samples_cloud
GLOBAL_P_OBS = p_obs

# --- 3. 定义先验和似然 ---
def prior_transform(u):
    v = np.array(u)
    v[0] = u[0] * 10.0                 # R_kick: [0, 10] km 
    v[1] = np.arccos(1.0 - 2.0 * u[1])  # alpha (各向同性) 
    v[2] = np.arccos(1.0 - 2.0 * u[2])  # beta (各向同性) 
    v[3] = np.arccos(1.0 - 2.0 * u[3])  # gamma (各向同性) 
    v[4] = u[4] * 1.0                  # P_birth: [0, 1] s 
    v[5] = u[5] * 1.0                  # tau_tilde: [0, 1] 
    return v

def log_likelihood(theta):
    R_kick, alpha, beta, gamma, P_birth, tau_tilde = theta
    
    sb = np.sin(beta)
    if sb <= 0: return -1e30
    
    # 核心物理模型计算
    # 使用预生成的包含误差的速度云进行计算
    # delta_S 维度: (n_pulsars, N_SAMPLES)
    delta_S = (M_NS * R_kick * GLOBAL_V_CLOUD * np.sin(alpha)) / (I_NS * sb)
    S_birth_star = (2.0 * np.pi) / P_birth
    
    # 合成出生角速度 S0
    S0 = np.sqrt((S_birth_star + delta_S * np.cos(gamma))**2 + (delta_S * np.sin(gamma))**2)
    
    # 预测当前周期 P_pred
    P_pred = (2.0 * np.pi) / (S0 * np.sqrt(1.0 - tau_tilde))
    # Period likelihood
    sigma =  GLOBAL_P_OBS
    # 对每个观测点的采样点计算对数似然
    # 先在采样维度(axis=1)求平均，然后再对所有脉冲星求和
    logL_elements = -0.5 * ((GLOBAL_P_OBS[:, None] - P_pred) / sigma[:, None])**2 - np.log(np.sqrt(2 * np.pi) * sigma[:, None])
    
    # 通过对数空间下的平均(logsumexp)来实现边缘化速度误差
    from scipy.special import logsumexp
    logL_per_star = logsumexp(logL_elements, axis=1) - np.log(N_SAMPLES)
    
    return np.sum(logL_per_star)

# --- 4. 执行并行采样 ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Archived single-kick comparison producer")
    parser.add_argument("--execute-fit", action="store_true")
    parser.add_argument("--ncores", type=int, default=8)
    args = parser.parse_args()
    if not args.execute_fit:
        parser.error("Use --execute-fit to run nested sampling")
    ndim = 6
    print(f"开始采样，包含速度误差处理，使用 10 核心...")
    with Pool(processes=args.ncores) as pool:
        sampler = dynesty.NestedSampler(
            log_likelihood, 
            prior_transform, 
            ndim, 
            pool=pool, 
            queue_size=args.ncores,
            rstate=np.random.default_rng(42), 
            nlive=1000
        )
        sampler.run_nested()
        results = sampler.results

    # --- 5. 绘制 Corner 图 ---
    labels = [r'$R_{kick}$ [km]', r'$\alpha$', r'$\beta$', r'$\gamma$', r'$P_{birth}$ [s]', r'$\tilde{\tau}$']
    fig, axes = dyplot.cornerplot(results, labels=labels, color='red', show_titles=True)
    fig.savefig(run_dir('single_kick') / 'single_kick_corner.png')
    
    # 输出最终 LogZ
    print(f"最终对数证据度 LogZ: {results.logz[-1]:.2f}")
    
    import pickle
    posterior_samples = resample_equal(results.samples, np.exp(results.logwt - results.logz[-1]), rstate=np.random.default_rng(43))
    with (run_dir("single_kick") / "posterior_LSR_47_samples.pkl").open("wb") as f:
        pickle.dump(posterior_samples, f)
