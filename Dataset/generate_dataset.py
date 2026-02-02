import numpy as np
import torch
import os
import sys
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Scenario.scenario import ITSNScenario
from Scenario.baseline_optimizer import BaselineZFSDROptimizer

# === 参数配置 ===
NUM_SAMPLES = 5000       # 生成样本总数
# 注意：这里不再设置 SCALE_FACTOR，直接保存原始物理值
SAVE_DIR = "dataset_ris_itsn_raw" # 修改目录名以示区别
NOISE_POWER_DBM = -174 + 10*np.log10(10e6) # -104 dBm
NOISE_POWER = 10**(NOISE_POWER_DBM/10) / 1000 # Watts

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_dataset():
    ensure_dir(SAVE_DIR)
    
    # 1. 初始化场景
    env = ITSNScenario()
    
    # 2. 初始化优化器 (Teacher)
    # 这里的参数保持不变，优化器内部会处理数值稳定性（如果求解器够强）
    # 或者优化器内部也是基于物理值计算的
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
        N_iter=5,    # BCD 迭代次数
        verbose=False # 关闭详细日志
    )

    data_list = []
    
    print(f"开始生成 {NUM_SAMPLES} 条数据...")
    print(f"注意：所有数据均为原始物理数值 (无缩放)")

    success_count = 0
    
    for i in tqdm(range(NUM_SAMPLES)):
        try:
            # --- A. 随机化环境 ---
            env.reset_user_positions() 
            
            # [修改点 1] 卫星方位角限制在 -180 ~ 45 度
            ele = np.random.uniform(40, 80) # 仰角保持不变
            azi = np.random.uniform(-180, 45) 
            env.update_satellite_position(ele, azi)
            
            # --- B. 生成原始信道 ---
            # 这里的信道是物理值 (极小)
            raw_channels = env.generate_channels()
            
            # --- C. 计算卫星等效单天线信道 ---
            # W_sat 是卫星对准 SU 的波束 (N_sat, 1)
            W_sat_beam = raw_channels['W_sat'] 
            
            # 将 (K, N_sat) 压缩为 (K, 1)
            h_s_k_eff = raw_channels['H_SAT2UE'] @ W_sat_beam 
            h_s_j_eff = raw_channels['H_SAT2SUE'] @ W_sat_beam 
            # 将 (N_ris, N_sat) 压缩为 (N_ris, 1)
            G_S_eff = raw_channels['G_SAT'] @ W_sat_beam 
            
            # 获取其他信道
            h_k = raw_channels['H_BS2UE']    # (K, Nt)
            h_j = raw_channels['H_BS2SUE']   # (SK, Nt)
            h_k_r = raw_channels['H_RIS2UE'] # (K, N_ris)
            h_j_r = raw_channels['H_RIS2SUE']# (SK, N_ris)
            G_BS = raw_channels['G_BS']      # (N_ris, Nt)
            
            # 构造一个虚假的 W_sat (1x1) 给优化器
            W_sat_dummy = np.ones((1, env.SK), dtype=complex)

            # --- D. 运行凸优化 (获取 Label) ---
            w_opt_norm, Phi_opt, info = optimizer.optimize(
                h_k, h_j, 
                h_s_k_eff, h_s_j_eff, 
                h_k_r, h_j_r, 
                G_BS, G_S_eff, 
                W_sat_dummy
            )
            
            # 提取优化结果
            # 1. RIS 相位 (Teacher Theta)
            theta_opt = np.angle(np.diag(Phi_opt)) # (N_ris,)
            
            # 2. 基站波束赋形 W (融合了功率 P_bs)
            W_opt = optimizer.w # (Nt, K) - 原始物理值
            
            # 3. 卫星功率
            P_sat_opt = info['final_P_sat'] # Watts - 原始物理值

            # --- E. 数据封装 (无缩放) ---
            
            # 堆叠 UE 和 SU 的信道 (适配 AQEnetwork 输入格式)
            # hk (K+SK, Nt)
            hk_stack = np.vstack([h_k, h_j]) 
            # hrk (K+SK, N_ris)
            hrk_stack = np.vstack([h_k_r, h_j_r])
            # hs (K+SK, 1)
            hs_stack = np.vstack([h_s_k_eff, h_s_j_eff])
            
            # [修改点 2] 移除所有 scale 操作，保存原始数据
            sample_dict = {
                # --- Labels (Teacher) ---
                'theta_opt': theta_opt.astype(np.float32),      # (N_ris,)
                'W_opt': W_opt.astype(np.complex64),            # (Nt, K)
                'P_sat': np.float32(P_sat_opt),                 # Scalar
                
                # --- Inputs (Features) ---
                'hk': hk_stack.astype(np.complex64),            # (K+SK, Nt)
                'hrk': hrk_stack.astype(np.complex64),          # (K+SK, N_ris)
                'GB': G_BS.astype(np.complex64),                # (N_ris, Nt)
                'hs': hs_stack.astype(np.complex64),            # (K+SK, 1)
                'GSAT': G_S_eff.astype(np.complex64),           # (N_ris, 1)
            }
            
            data_list.append(sample_dict)
            success_count += 1

        except Exception as e:
            # 忽略优化失败的样本
            continue

    # 4. 保存为 PyTorch 格式
    save_path = os.path.join(SAVE_DIR, "train_dataset_raw.pt")
    torch.save(data_list, save_path)
    print(f"数据集生成完毕！成功生成样本数: {success_count}/{NUM_SAMPLES}")
    print(f"保存路径: {save_path}")

if __name__ == "__main__":
    generate_dataset()