import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_root = next(p for p in _ArchivePath(__file__).resolve().parents if (p / "config.json").is_file())
_archive_sys.path.insert(0, str(_archive_root / "scripts"))
from archive_paths import ROOT, DATA, MCMC, EVIDENCE, run_dir

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, maxwell, weibull_min, norm
from scipy.special import logsumexp, gamma
import time
import multiprocessing as mp
import sys
# 2) 先建一个输出目录
from pathlib import Path
outdir = run_dir("mcmc")
outdir.mkdir(parents=True, exist_ok=True)
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
    theta 参数列表 (10个):
    1. sigma_v
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
        sigma_v, R_eff_km, zeta, sin_alpha, sin_beta, \
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
    v_3d_kms = maxwell.rvs(scale=sigma_v, size=N_sim)
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
    if len(theta) != 10:
        return -np.inf
    # [修改] 解包 10 参数
    sigma_v, R, zeta, sa, sb, Pb_ms, k, f, M_sol, I_45 = theta

    if not (10.0 < sigma_v < 1500.0): return -np.inf
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
    # 索引顺序: sigma_v, R, zeta, sa, sb, Pb_ms, k, f, M, I
    # zeta 步长设 0.05 (与 sin_alpha/sin_beta 同量级)
    step_scale = np.array([10, 0.2, 0.05, 0.02, 0.02,
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
    import argparse
    parser = argparse.ArgumentParser(description="Archived MCMC producer; original scientific settings are the defaults.")
    parser.add_argument("--execute-fit", action="store_true", help="Explicitly start the expensive fit")
    parser.add_argument("--ncores", type=int, default=300)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    if not args.execute_fit:
        parser.error("Use --execute-fit to run a fit. Existing results are in results/mcmc.")
    if args.ncores < 1 or args.steps < 1:
        parser.error("ncores and steps must be positive")
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
    clean_velocities = np.loadtxt(base_path + "vpec_lsr_median_47.txt")
    clean_err_low = np.loadtxt(base_path + "vpec_lsr_err_low_47.txt")
    clean_err_up = np.loadtxt(base_path + "vpec_lsr_err_up_47.txt")
    clean_periods=np.loadtxt(base_path+'period_2_new_47.txt')
    n_obs = len(clean_periods)
    print(f"载入 {n_obs} 个有效观测源。")

    # --- 生成误差云 ---
    # --- Read LSR-corrected transverse peculiar velocity samples ---
    N_samples_per_obs = 100

    obs_velocity_samples = np.loadtxt(
    base_path + "vpec_lsr_samples_matrix_47x100.txt"
    )

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
    # [修改] 初始猜测 (10参数), zeta 初值 0.5
    # 顺序:sigma_v, R, zeta, sa, sb, Pb_ms, k, f, M, I
    init_guess = [265, 10.0, 0.5, 0.5, 0.5, 400.0, 1.0, 5.0, 1.4, 1.0]

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
    best_theta = np.median(combined_samples, axis=0)

    # [修改] 标签更新 (11个), 插入 zeta
    labels = [
    r"$\sigma_{v}$ (km/s)", r"$R_{eff}$",
    r"$\zeta$",
    r"$\sin\alpha$", r"$\sin\beta$", r"$P_{b,*}$ (ms)",
    r"$k_{w}$", r"$f_{evo}$",
    r"$M_{NS} (M_\odot)$", r"$I_{45}$"]

    print("\n>>> 最佳拟合参数 (Median):")
    for l, v in zip(labels, best_theta):
        print(f"{l}: {v:.4f}")
    # ===== 保存 posterior 样本 =====
    import pickle

    save_dict = {
        "model_name": "M_47_10_Param",
        "columns": ["sigma_v", "R_eff", "zeta", "sin_alpha", "sin_beta",
        "Pb_ms", "k_w", "f_evo", "M_sol", "I_45"
        ],
        "samples": combined_samples
    }

    with open(outdir / "posterior_young_47_10param.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print(">>> 已保存为 pickle 格式！")

    if corner:
        fig = corner.corner(combined_samples, labels=labels,
                            quantiles=[0.16, 0.5, 0.84],
                            color='#2C7BB6',
                            show_titles=True, title_fmt=".2f")
        fig.savefig(outdir / "corner_47_M_10.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    # 散点图
    sim_logP, sim_logV_cgs = simulate_full_population(20000, best_theta)
    sim_logV_kms = sim_logV_cgs - 5.0
    data_out = np.column_stack([sim_logP, sim_logV_kms])
    np.savetxt(outdir / "p_v_scatter_47_M_10.txt", data_out,
           header="log10P  log10V_kms")

    print(">>> 已保存散点数据为 txt!")
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
    plt.title("M_Coupling Fit (10 Parameters, with zeta)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outdir /"pv_scatter_47_M_10.png", dpi=300, bbox_inches="tight")
    plt.close()
