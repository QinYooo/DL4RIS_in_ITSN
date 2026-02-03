import numpy as np
import torch
import os
import sys
import argparse
import multiprocessing
import concurrent.futures
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Scenario.scenario import ITSNScenario
from Scenario.baseline_optimizer import BaselineZFSDROptimizer

# === 默认配置 ===
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_BATCH_SIZE = 50       # 每个子进程一次处理多少样本
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
    batch_idx, batch_size, base_seed = args
    
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
    
    # 在 Batch 内循环生成
    for _ in range(batch_size):
        try:
            # --- A. 随机化环境 ---
            env.reset_user_positions() 
            
            # 卫星角度限制: 仰角 40~80, 方位角 -180~45
            ele = np.random.uniform(40, 80)
            azi = np.random.uniform(-180, 45) 
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
            
            W_sat_dummy = np.ones((1, env.SK), dtype=complex)

            # --- D. 运行凸优化 (获取 Label) ---
            w_opt_norm, Phi_opt, info = optimizer.optimize(
                h_k, h_j, 
                h_s_k_eff, h_s_j_eff, 
                h_k_r, h_j_r, 
                G_BS, G_S_eff, 
                W_sat_dummy
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
    
    # 任务切分
    batch_size = DEFAULT_BATCH_SIZE
    num_batches = args.num_samples // batch_size
    if args.num_samples % batch_size != 0:
        num_batches += 1
        
    tasks = []
    for i in range(num_batches):
        current_size = min(batch_size, args.num_samples - i * batch_size)
        # 每个 Batch 分配一个基于全局 Seed 的偏移 Seed
        tasks.append((i, current_size, args.seed))

    print(f"=== 多进程数据集生成 ===")
    print(f"目标样本: {args.num_samples}")
    print(f"进程数量: {args.num_workers if args.num_workers else 'Auto'}")
    print(f"输出路径: {os.path.join(SAVE_DIR, args.output_name)}")

    all_data = []
    
    # 启动多进程池
    # Windows 下必须在 if __name__ == '__main__': 保护块内执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
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