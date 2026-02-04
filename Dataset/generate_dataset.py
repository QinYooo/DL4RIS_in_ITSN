import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np
import torch
import sys
import argparse
import multiprocessing
import concurrent.futures
from tqdm import tqdm

# Windows multiprocessing fix: 必须在导入其他模块前设置
if os.name == 'nt':
    multiprocessing.set_start_method('spawn', force=True)

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Scenario.scenario import ITSNScenario
from Scenario.baseline_optimizer import BaselineZFSDROptimizer

# === 默认配置 ===
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_BATCH_SIZE = 2        # 每个子进程一次处理多少样本
SAVE_DIR = "dataset_ris_itsn_raw"
NOISE_POWER_DBM = -174 + 10*np.log10(10e6)
NOISE_POWER = 10**(NOISE_POWER_DBM/10) / 1000

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_batch(args):
    """
    子进程执行函数：生成一批数据
    """
    batch_idx, batch_size, base_seed, pos = args
    # 可选：显示进程信息
    pid = os.getpid()
    
    # 为当前 Batch 设置唯一的随机种子
    seed = base_seed + batch_idx + np.random.randint(0, 10000)
    
    # 1. 初始化场景 (每个进程独立)
    env = ITSNScenario()
    
    # 2. 初始化优化器 (Teacher)
    optimizer = BaselineZFSDROptimizer(
        K=env.K,
        J=env.SK,
        N_t=env.N_t,
        N_s=1,  # 卫星等效为单天线
        N=env.N_ris,
        P_max=env.P_bs_max,
        sigma2=env.P_noise,
        gamma_k=np.ones((env.K, 1)) * env.db2pow(13), # UE SINR 阈值 13dB
        gamma_j=env.db2pow(13),                       # SU SINR 阈值 13dB
        ris_amplitude_gain=env.ris_amplitude_gain,
        N_iter=5,     # BCD 迭代次数
        verbose=False # 关闭详细日志
    )
    
    local_data = []

    # position=pos 确保每个进程占一行
    # leave=False 避免结束后残留
    pbar = tqdm(
        total=batch_size,
        desc=f"PID {pid} | Batch {batch_idx}",
        position=pos,
        leave=False,
        dynamic_ncols=True
    )
    
    # 在 Batch 内循环生成
    for _ in range(batch_size):
        try:
            # --- A. 随机化环境 ---
            env.reset_user_positions() 
            
            # 卫星角度限制: 仰角 40~80, 方位角 -180~45
            ele = np.random.uniform(40, 80)
            azi = np.random.uniform(-180, 0) 
            env.update_satellite_position(ele, azi)
            
            # --- B. 生成原始信道 ---
            raw_channels = env.generate_channels()
            
            # --- C. 计算卫星等效单天线信道 ---
            W_sat_beam = raw_channels['W_sat'] 
            
            h_s_k_eff = raw_channels['H_SAT2UE'] @ W_sat_beam 
            h_s_j_eff = raw_channels['H_SAT2SUE'] @ W_sat_beam 
            G_S_eff = raw_channels['G_SAT'] @ W_sat_beam 
            
            h_k = raw_channels['H_BS2UE']
            h_j = raw_channels['H_BS2SUE']
            h_k_r = raw_channels['H_RIS2UE']
            h_j_r = raw_channels['H_RIS2SUE']
            G_BS = raw_channels['G_BS']

            # --- D. 运行凸优化 (获取 Label) ---
            # Get current channels
            channels = raw_channels

            # Extract channels (using scenario's naming convention) with conjugate
            h_k_opt = channels['H_BS2UE'].conj()      # BS -> BS users (K, N_t)
            h_j_opt = channels['H_BS2SUE'].conj()     # BS -> SAT users (J, N_t)
            h_s_k_opt = channels['H_SAT2UE'].conj()   # SAT -> BS users (K, N_s)
            h_s_j_opt = channels['H_SAT2SUE'].conj()  # SAT -> SAT users (J, N_s)
            h_k_r_opt = channels['H_RIS2UE'].conj()   # BS users -> RIS (K, N)
            h_j_r_opt = channels['H_RIS2SUE'].conj()  # SAT users -> RIS (J, N)
            G_BS_opt = channels['G_BS'].conj()        # RIS -> BS (N, N_t)
            G_S_opt = channels['G_SAT'].conj()        # RIS -> SAT (N, N_s)
            W_sat = channels['W_sat']                # Satellite beamforming

            # Run baseline optimization
            w_opt_norm, Phi_opt, info = optimizer.optimize(
                h_k_opt,
                h_j_opt,
                h_s_k_opt,
                h_s_j_opt,
                h_k_r_opt,
                h_j_r_opt,
                G_BS_opt,
                G_S_opt,  # Use true G_SAT (no ephemeris uncertainty)
                W_sat
            )
            
            # 提取结果
            theta_opt = np.angle(np.diag(Phi_opt))
            W_opt = optimizer.w
            P_sat_opt = info['final_P_sat']

            # --- E. 数据封装 (无缩放) ---
            hk_stack = np.vstack([h_k, h_j]) 
            hrk_stack = np.vstack([h_k_r, h_j_r])
            hs_stack = np.vstack([h_s_k_eff, h_s_j_eff])
            
            sample_dict = {
                'theta_opt': theta_opt.astype(np.float32),      # (N_ris,)
                'W_opt': W_opt.astype(np.complex64),            # (Nt, K)
                'P_sat': np.float32(P_sat_opt),                 # Scalar
                'hk': hk_stack.astype(np.complex64),            # (K+SK, Nt)
                'hrk': hrk_stack.astype(np.complex64),          # (K+SK, N_ris)
                'GB': G_BS.astype(np.complex64),                # (N_ris, Nt)
                'hs': hs_stack.astype(np.complex64),            # (K+SK, 1)
                'GSAT': G_S_eff.astype(np.complex64),           # (N_ris, 1)
            }
            
            local_data.append(sample_dict)

        except Exception:
            # 忽略优化失败的样本
            continue
        finally:
            pbar.update(1)
    pbar.close()        
    return local_data

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Multiprocess ITSN Dataset Generation")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Total samples to generate")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of processes (default: CPU cores)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_name", type=str, default="train_dataset_raw.pt", help="Output filename")
    args = parser.parse_args()

    ensure_dir(SAVE_DIR)

    # 确定实际进程数
    if args.num_workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = args.num_workers

    # 任务切分：根据 num_samples 和 num_workers 自动计算每个进程的采样数量
    batch_size = max(1, args.num_samples // num_workers)
    num_batches = args.num_samples // batch_size
    if args.num_samples % batch_size != 0:
        num_batches += 1
        
    tasks = []
    for i in range(num_batches):
        current_size = min(batch_size, args.num_samples - i * batch_size)
        pos = 1 + (i % num_workers)  # 每个worker复用固定行
        # 每个 Batch 分配一个基于全局 Seed 的偏移 Seed
        tasks.append((i, current_size, args.seed, pos))

    print(f"=== 多进程数据集生成 ===")
    print(f"目标样本: {args.num_samples}")
    print(f"进程数量: {num_workers}")
    print(f"每进程样本: {batch_size}")
    print(f"输出路径: {os.path.join(SAVE_DIR, args.output_name)}")

    all_data = []
    
    # 启动多进程池
    # Windows 下必须在 if __name__ == '__main__': 保护块内执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_batch, t) for t in tasks]
        
        # 使用 tqdm 显示总体进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Batches"):
            try:
                batch_data = future.result()
                all_data.extend(batch_data)
            except Exception as e:
                print(f"Batch Error: {e}")

    # 保存结果
    save_path = os.path.join(SAVE_DIR, args.output_name)
    torch.save(all_data, save_path)
    print(f"\n完成！成功生成样本数: {len(all_data)} / {args.num_samples}")
    print(f"保存至: {save_path}")

if __name__ == "__main__":
    # Windows 必须调用 freeze_support
    multiprocessing.freeze_support()
    main()