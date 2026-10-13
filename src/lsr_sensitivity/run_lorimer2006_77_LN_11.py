import sys as _package_sys
from pathlib import Path as _PackagePath
_package_root=next(p for p in _PackagePath(__file__).resolve().parents if (p/'config.json').is_file())
_package_sys.path.insert(0,str(_package_root/'scripts'))
from archive_paths import DATA,run_dir
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, lognorm, weibull_min, norm
from scipy.special import logsumexp, gamma
import time
import multiprocessing as mp
import sys

# 尝试导入 corner
try:
    import corner
except ImportError:
    corner = None

# ==========================================
# 1. 物理模型：11参数 (Pb_* 版本 + zeta 相干因子 + Sigmoid 平滑)
# ==========================================
def simulate_full_population(N_sim, theta):
    """
    theta 参数列表 (11个):
    0. mu_log:      Kick速度 Log-Normal 的 mu
    1. sigma_log:   Kick速度 Log-Normal 的 sigma
    2. R_eff_km:    有效半径 (km)
    3. zeta:        [新增] 相干因子 zeta in [0,1]
    4. sin_alpha:   几何因子 alpha 的正弦值
    5. sin_beta:    几何因子 beta 的正弦值
    6. P_b_star_ms: 基础自旋周期 P_b,* (单位 ms)
    7. k_weibull:   Weibull 分布形状参数 k
    8. f_evo:       周期演化因子 <1/(1-tau)>
    9. M_sol:       中子星质量 (太阳质量)
    10. I_45:       转动惯量 (10^45 g cm^2)
    """
    try:
        # [修改] 解包 11 个参数，zeta 插在 R_eff_km 之后
        mu_log, sigma_log, R_eff_km, zeta, sin_alpha, sin_beta, \
            P_b_star_ms, k_w, f_evo, M_sol, I_45 = theta
    except ValueError:
        return np.array([]), np.array([])

    # 物理常数计算
    M_ns = M_sol * 1.989e33   # g
    I_ns = I_45 * 1e45        # g cm^2

    # 将 P_b (ms) 转换为 S_b (cgs)
    P_b_s = P_b_star_ms * 1e-3
    S_b_star = (2 * np.pi * I_ns) / P_b_s

    # 1. Kick (km/s -> cm/s)
    v_3d_kms = lognorm.rvs(s=sigma_log, scale=np.exp(mu_log), size=N_sim)
    v_3d_cgs = v_3d_kms * 1e5

    u = np.random.uniform(-1, 1, N_sim)
    sin_proj = np.sqrt(1 - u**2)
    v_trans = v_3d_cgs * sin_proj

    # 2. RMS Coupling (引入 zeta: delta_Omega_rms 正比于 zeta)
    R_cgs = R_eff_km * 1e5
    delta_S_dist = (M_ns * R_cgs * sin_alpha * v_trans) / (sin_beta + 1e-10)
    # [关键修改] spin-up 方差项乘以 zeta^2 (即 delta_S_rms 正比 zeta)
    delta_S_sq_mean = (zeta**2) * np.mean(delta_S_dist**2)

    S0_rms_val = np.sqrt(S_b_star**2 + delta_S_sq_mean)
    P0_rms = (2 * np.pi * I_ns) / S0_rms_val

    # 3. Weibull
    gamma_factor = np.sqrt(gamma(1 + 2.0 / k_w))
    lambda_w = P0_rms / gamma_factor
    P_birth = weibull_min.rvs(c=k_w, scale=lambda_w, size=N_sim)

    # 4. Evolution & Sigmoid Selection
    P_obs = P_birth * f_evo

    p_detect = 1.0 / (1.0 + np.exp(-(P_obs - 0.1) / 0.01))
    random_roll = np.random.rand(N_sim)

    mask = (random_roll < p_detect) & (P_obs < 100) & (v_trans > 0)

    return np.log10(P_obs[mask]), np.log10(v_trans[mask])

# ==========================================
# 2. 似然函数
# ==========================================
def log_prior(theta):
    if len(theta) != 11:
        return -np.inf
    # [修改] 解包 11 参数
    mu, sig, R, zeta, sa, sb, Pb_ms, k, f, M_sol, I_45 = theta

    if not (4.0 < mu < 7.5):        return -np.inf
    if not (0.1 < sig < 2.0):       return -np.inf
    # [修改] R 先验上界放宽到 50
    if not (0.1 < R < 50.0):        return -np.inf
    # [新增] zeta 先验 U[0,1]
    if not (0.0 <= zeta <= 1.0):    return -np.inf
    if not (0.01 < sa < 1.0):       return -np.inf
    if not (0.01 < sb < 1.0):       return -np.inf
    if not (10.0 < Pb_ms < 5000.0): return -np.inf
    if not (0.1 < k < 5.0):         return -np.inf
    if not (1.0 < f < 50.0):        return -np.inf
    if not (0.8 < M_sol < 2.5):     return -np.inf
    if not (0.5 < I_45 < 3.0):      return -np.inf

    return 0.0

def log_likelihood(theta, obs_data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        sim_logP, sim_logV = simulate_full_population(5000, theta)
    except Exception:
        return -np.inf
    if len(sim_logP) < 50:
        return -np.inf
    try:
        values = np.vstack([sim_logP, sim_logV])
        kernel = gaussian_kde(values, bw_method='scott')
        obs_P_mat, obs_V_mat = obs_data
        n_obs, n_samples = obs_P_mat.shape
        check_points = np.vstack([obs_P_mat.ravel(), obs_V_mat.ravel()])
        log_pdf_all = kernel.logpdf(check_points)
        log_pdf_all = np.maximum(log_pdf_all, -100.0)
        log_pdf_mat = log_pdf_all.reshape(n_obs, n_samples)
        log_likelihood_per_star = logsumexp(log_pdf_mat, axis=1) - np.log(n_samples)
        return np.sum(log_likelihood_per_star) + lp
    except Exception:
        return -np.inf

# ==========================================
# 3. 单链 MCMC
# ==========================================
def run_single_chain(args):
    seed, start_theta, n_steps, obs_data, is_representative = args
    np.random.seed(seed)
    ndim = len(start_theta)
    current_theta = np.array(start_theta) * (1 + np.random.normal(0, 0.01, ndim))
    current_lp = log_likelihood(current_theta, obs_data)
    if current_lp == -np.inf:
        current_theta = np.array(start_theta)
        current_lp = log_likelihood(current_theta, obs_data)

    chain = []
    accepted = 0
    # [修改] 步长设置 (11维):
    # 索引顺序: mu, sig, R, zeta, sa, sb, Pb_ms, k, f, M, I
    # zeta 步长设 0.05 (与 sin_alpha/sin_beta 同量级)
    step_scale = np.array([0.05, 0.03, 0.2, 0.05, 0.02, 0.02,
                           20.0, 0.05, 0.2, 0.05, 0.05])

    start_time = time.time()
    for i in range(n_steps):
        proposal = current_theta + np.random.normal(0, 1, ndim) * step_scale
        proposal_lp = log_likelihood(proposal, obs_data)
        if proposal_lp > -np.inf:
            if current_lp == -np.inf:
                ratio = 1.0
            else:
                ratio = np.exp(proposal_lp - current_lp)
            if np.random.rand() < ratio:
                current_theta = proposal
                current_lp = proposal_lp
                accepted += 1
        chain.append(current_theta)
        if is_representative and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            steps_done = i + 1
            avg_time = elapsed / steps_done
            remain = (n_steps - steps_done) * avg_time
            print(f"[进度] {steps_done/n_steps*100:.1f}% ({steps_done}/{n_steps}) "
                  f"| Acc: {accepted/steps_done:.2f} | ETA: {remain/60:.1f} min")
            sys.stdout.flush()
    return np.array(chain), accepted / n_steps

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    import argparse,pickle
    parser=argparse.ArgumentParser(description="Lorimer Model C: 77-source Lognormal population fit")
    parser.add_argument('--execute-fit',action='store_true')
    parser.add_argument('--ncores',type=int,default=300)
    parser.add_argument('--steps',type=int,default=2000)
    args=parser.parse_args()
    if not args.execute_fit:parser.error('Use --execute-fit to run MCMC. Recorded results are in results/lsr_sensitivity/.')
    if args.ncores<1 or args.steps<1:parser.error('ncores and steps must be positive')
    fit_output=run_dir('lsr_sensitivity/lorimer2006')
    # --- 数据导入 (请确保路径正确) ---
    base_path = str(DATA) + "/"
    try:
        raw_velocities = np.loadtxt(base_path + 'vtrans_87_data.txt')
        raw_periods = np.loadtxt(base_path + 'period_87_data.txt')
        raw_v_err_low = np.loadtxt(base_path + 'v_err_low_87_data.txt')
        raw_v_err_up = np.loadtxt(base_path + 'v_err_up_87_data.txt')
    except Exception as e:
        print(f"数据加载失败: {e}")
        sys.exit(1)

    mask_period = raw_periods > 0.1
    mask_velocity = (raw_velocities <= 1000.0) & (raw_velocities > 10.0)
    final_mask = mask_period & mask_velocity
    # ============================================================
# Load Lorimer Model C LSR-corrected transverse peculiar velocity cloud
# ============================================================

# period_1_new 必须是与这 77 个源顺序一致的周期数组
    clean_periods = np.loadtxt(base_path+'period_1_new_77.txt')

    n_obs = len(clean_periods)
    print(f"载入 {n_obs} 个有效观测源。")

    # 每个源的速度云样本数
    N_samples_per_obs = 100

    # 读取 Lorimer Model C 的 LSR-corrected Vpec 速度云
    obs_velocity_samples = np.loadtxt(
        base_path + "lorimer2006/vpec_lsr_lorimerC_quantile_cloud_77x100.txt"
    )

    print("Lorimer C velocity cloud shape:", obs_velocity_samples.shape)

    # 防止转置问题
    if obs_velocity_samples.shape == (N_samples_per_obs, n_obs):
        print("Detected transposed Lorimer C cloud. Transposing to n_obs x N_samples.")
        obs_velocity_samples = obs_velocity_samples.T

    if obs_velocity_samples.shape != (n_obs, N_samples_per_obs):
        raise ValueError(
            f"Expected obs_velocity_samples shape "
            f"{(n_obs, N_samples_per_obs)}, "
            f"but got {obs_velocity_samples.shape}."
    )

    if not np.all(np.isfinite(obs_velocity_samples)):
        raise ValueError("obs_velocity_samples contains NaN or inf.")

    if np.any(obs_velocity_samples <= 0):
        raise ValueError("obs_velocity_samples contains non-positive velocities.")

# 从 Lorimer C 速度云中计算 median 和 asymmetric error
# 这些主要用于画图和检查，不是似然函数必须的
    v_lorimer_16 = np.percentile(obs_velocity_samples, 16, axis=1)
    v_lorimer_50 = np.percentile(obs_velocity_samples, 50, axis=1)
    v_lorimer_84 = np.percentile(obs_velocity_samples, 84, axis=1)

    clean_velocities = v_lorimer_50
    clean_err_low = v_lorimer_50 - v_lorimer_16
    clean_err_up = v_lorimer_84 - v_lorimer_50

    print("Lorimer C velocity summary check:")
    print(
        "J1818-1607, index 47: "
        f"{clean_velocities[47]:.3f} "
        f"+{clean_err_up[47]:.3f} "
        f"-{clean_err_low[47]:.3f} km/s"
    )

# The model returns log10(V_trans / cm s^-1),
# so observed km/s velocities must be converted to cm/s.
    obs_logV_cloud = np.log10(obs_velocity_samples * 1e5)

    obs_logP_cloud = np.tile(
        np.log10(clean_periods)[:, np.newaxis],
        (1, N_samples_per_obs),
    )

    obs_data = (obs_logP_cloud, obs_logV_cloud)


    if obs_velocity_samples.shape != (n_obs, N_samples_per_obs):
        raise ValueError(
            f"Expected obs_velocity_samples shape "
            f"{(n_obs, N_samples_per_obs)}, "
            f"but got {obs_velocity_samples.shape}."
        )

    if not np.all(np.isfinite(obs_velocity_samples)):
        raise ValueError("obs_velocity_samples contains NaN or inf.")

    if np.any(obs_velocity_samples <= 0):
        raise ValueError("obs_velocity_samples contains non-positive velocities.")

    # The model returns log10(V_trans / cm s^-1),
    # so the observed km/s velocities must be converted to cm/s.
    obs_logV_cloud = np.log10(obs_velocity_samples * 1e5)

    obs_logP_cloud = np.tile(
    np.log10(clean_periods)[:, np.newaxis],
    (1, N_samples_per_obs),
    )

    obs_data = (obs_logP_cloud, obs_logV_cloud)


    # --- 并行 MCMC ---
    n_cores = args.ncores
    try:
        max_cpu = mp.cpu_count()
        if n_cores > max_cpu:
            n_cores = max_cpu
    except Exception:
        pass

    print(f"\n>>> 启动并行 MCMC ({n_cores} 核心) - 11参数(Pb_ms + zeta)版本...")

    steps_per_chain = args.steps
    # [修改] 初始猜测 (11参数), zeta 初值 0.5
    # 顺序: mu, sig, R, zeta, sa, sb, Pb_ms, k, f, M, I
    init_guess = [5.5, 0.5, 10.0, 0.5, 0.5, 0.5, 400.0, 1.0, 5.0, 1.4, 1.0]

    seeds = [1000 + i for i in range(n_cores)]
    tasks = []
    for i, seed in enumerate(seeds):
        is_representative = (i == 0)
        tasks.append((seed, init_guess, steps_per_chain, obs_data, is_representative))

    start_time = time.time()
    results = []
    with mp.Pool(processes=n_cores) as pool:
        iterator = pool.imap_unordered(run_single_chain, tasks)
        for result in iterator:
            results.append(result)
    print(f"\n>>> 计算完成! 耗时: {time.time() - start_time:.1f} s")

    # --- 绘图 ---
    all_chains = []
    acc_rates = []
    for chain, acc in results:
        burn_in = int(steps_per_chain * 0.3)
        if len(chain) > burn_in:
            all_chains.append(chain[burn_in:])
        acc_rates.append(acc)

    print(f"平均接受率: {np.mean(acc_rates):.2f}")
    combined_samples = np.vstack(all_chains)
    columns=['mu_v','sigma_v','R_eff','zeta','sin_alpha','sin_beta','Pb_ms','k_w','f_evo','M_sol','I_45']
    payload={'columns':columns,'samples':combined_samples,'model_name':'Lognormal_77_Lorimer2006_11param','steps':args.steps,'n_chains':n_cores,'burn_fraction':0.3,'seeds':seeds}
    with (fit_output/'posterior_lorimer2006_77_LN_11.pkl').open('wb') as file:pickle.dump(payload,file)
    np.savez_compressed(fit_output/'posterior_lorimer2006_77_LN_11.npz',samples=combined_samples,columns=np.array(columns))
    best_theta = np.median(combined_samples, axis=0)

    # [修改] 标签更新 (11个), 插入 zeta
    labels = [
        r"$\mu_{kick}$", r"$\sigma_{kick}$", r"$R_{eff}$",
        r"$\zeta$",
        r"$\sin\alpha$", r"$\sin\beta$", r"$P_{b,*}$ (ms)",
        r"$k_{w}$", r"$f_{evo}$",
        r"$M_{NS} (M_\odot)$", r"$I_{45}$"
    ]

    print("\n>>> 最佳拟合参数 (Median):")
    for l, v in zip(labels, best_theta):
        print(f"{l}: {v:.4f}")

    if corner:
        fig = corner.corner(combined_samples, labels=labels,
                            quantiles=[0.16, 0.5, 0.84],
                            color='#2C7BB6',
                            show_titles=True, title_fmt=".2f")
        fig.savefig(fit_output/'lorimer2006_corner.pdf',bbox_inches='tight')
        plt.close(fig)

    # 散点图
    sim_logP, sim_logV_cgs = simulate_full_population(20000, best_theta)
    sim_logV_kms = sim_logV_cgs - 5.0

    plt.figure(figsize=(9, 7))
    plt.scatter(sim_logP, sim_logV_kms, s=10, color='red', alpha=0.1, label='Model Posterior')

    v_lsr_16 = np.percentile(obs_velocity_samples, 16, axis=1)
    v_lsr_50 = np.percentile(obs_velocity_samples, 50, axis=1)
    v_lsr_84 = np.percentile(obs_velocity_samples, 84, axis=1)

    obs_logP_plot = np.log10(clean_periods)
    obs_logV_plot = np.log10(v_lsr_50)

    yerr_low = obs_logV_plot - np.log10(np.maximum(v_lsr_16, 1e-10))
    yerr_up = np.log10(v_lsr_84) - obs_logV_plot


    plt.errorbar(
    obs_logP_plot,
    obs_logV_plot,
    yerr=[yerr_low, yerr_up],
    fmt='o',
    color='blue',
    ecolor='#2C7BB6',
    elinewidth=1,
    capsize=2,
    label='Observed LSR-corrected',
    zorder=10,
    )


    plt.xlabel(r'$\log_{10}(P / s)$')
    plt.ylabel(r'$\log_{10}(V_{trans} / km \cdot s^{-1})$')
    plt.title("Weibull-RMS Coupling Fit (11 Parameters, with zeta)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(fit_output/'lorimer2006_predictive.pdf',bbox_inches='tight')
    plt.close('all')
