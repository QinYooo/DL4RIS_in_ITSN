import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import torch
import sys
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Scenario.scenario import ITSNScenario
from Scenario.baseline_optimizer import BaselineZFSDROptimizer


def print_section(title):
    """打印分隔线和标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_info(key, value):
    """格式化打印键值对"""
    if isinstance(value, (np.ndarray, torch.Tensor)):
        print(f"{key:20s}: shape={value.shape}, dtype={value.dtype}")
        if value.size <= 10:
            print(f"{'':20s}  values: {value}")
        else:
            print(f"{'':20s}  min={value.min():.4e}, max={value.max():.4e}, mean={value.mean():.4e}")
    else:
        print(f"{key:20s}: {value}")


def generate_single_sample(seed=None):
    """
    生成单个数据集样本

    Args:
        seed: 随机种子，用于复现结果

    Returns:
        sample_dict: 包含信道和优化结果的字典
        info: 优化器的额外信息
    """
    if seed is not None:
        np.random.seed(seed)

    print_section("1. 初始化环境与优化器")

    # 1. 初始化场景
    print("初始化 ITSNScenario...")
    env = ITSNScenario()
    print(f"  BS 天线数 N_t: {env.N_t}")
    print(f"  RIS 元素数 N_ris: {env.N_ris}")
    print(f"  BS 用户数 K: {env.K}")
    print(f"  卫星用户数 SK: {env.SK}")
    print(f"  BS 最大功率 P_max: {env.P_bs_max} dBm")
    print(f"  噪声功率: {env.P_noise:.4e}")

    # 2. 初始化优化器 (Teacher)
    

    print_section("2. 随机化环境")

    # 随机化用户位置
    env.reset_user_positions()
    print("用户位置已重置")

    # 随机化卫星角度: 仰角 40~80, 方位角 -180~45
    ele = np.random.uniform(40, 80)
    azi = np.random.uniform(-180, 0)
    print(f"卫星仰角: {ele:.2f}°, 方位角: {azi:.2f}°")
    env.update_satellite_position(ele, azi)

    print_section("3. 生成原始信道")

    raw_channels = env.generate_channels()

    print("生成的信道矩阵:")
    print_info("H_BS2UE", raw_channels['H_BS2UE'])
    print_info("H_BS2SUE", raw_channels['H_BS2SUE'])
    print_info("H_SAT2UE", raw_channels['H_SAT2UE'])
    print_info("H_SAT2SUE", raw_channels['H_SAT2SUE'])
    print_info("H_RIS2UE", raw_channels['H_RIS2UE'])
    print_info("H_RIS2SUE", raw_channels['H_RIS2SUE'])
    print_info("G_BS", raw_channels['G_BS'])
    print_info("G_SAT", raw_channels['G_SAT'])
    print_info("W_sat", raw_channels['W_sat'])

    print_section("4. 计算卫星等效单天线信道")

    W_sat_beam = raw_channels['W_sat']
    h_s_k_eff = raw_channels['H_SAT2UE'] @ W_sat_beam
    h_s_j_eff = raw_channels['H_SAT2SUE'] @ W_sat_beam
    G_S_eff = raw_channels['G_SAT'] @ W_sat_beam

    print_info("h_s_k_eff (K,1)", h_s_k_eff)
    print_info("h_s_j_eff (SK,1)", h_s_j_eff)
    print_info("G_S_eff (N_ris,1)", G_S_eff)

    print("\n初始化 BaselineZFSDROptimizer...")
    optimizer = BaselineZFSDROptimizer(
        K=env.K,
        J=env.SK,
        N_t=env.N_t,
        N_s=env.N_sat, 
        N=env.N_ris,
        P_max=env.P_bs_max,
        P_s_init= 15,
        sigma2=env.P_noise,
        gamma_k=np.ones((env.K, 1)) * env.db2pow(13),  # UE SINR 阈值 13dB
        gamma_j=env.db2pow(13),                         # SU SINR 阈值 13dB
        ris_amplitude_gain=env.ris_amplitude_gain,
        N_iter=5,       # BCD 迭代次数
        verbose=True    # 显示优化日志
    )

    print_section("5. 运行凸优化 (获取 Label)")

    h_k = raw_channels['H_BS2UE']
    h_j = raw_channels['H_BS2SUE']
    h_k_r = raw_channels['H_RIS2UE']
    h_j_r = raw_channels['H_RIS2SUE']
    G_BS = raw_channels['G_BS']

    # 使用共轭的信道进行优化
    h_k_opt = raw_channels['H_BS2UE'].conj()
    h_j_opt = raw_channels['H_BS2SUE'].conj()
    h_s_k_opt = raw_channels['H_SAT2UE'].conj()
    h_s_j_opt = raw_channels['H_SAT2SUE'].conj()
    h_k_r_opt = raw_channels['H_RIS2UE'].conj()
    h_j_r_opt = raw_channels['H_RIS2SUE'].conj()
    G_BS_opt = raw_channels['G_BS'].conj()
    G_S_opt = raw_channels['G_SAT'].conj()

    print("开始优化...")
    w_opt_norm, Phi_opt, info = optimizer.optimize(
        h_k_opt,
        h_j_opt,
        h_s_k_opt,
        h_s_j_opt,
        h_k_r_opt,
        h_j_r_opt,
        G_BS_opt,
        G_S_opt,
        W_sat_beam
    )

    print_section("6. 优化结果")

    print(f"优化状态: {info.get('status', 'N/A')}")
    print(f"迭代次数: {info.get('n_iter', 'N/A')}")
    print(f"最终 BS 功率: {info.get('final_P_bs', 'N/A'):.4f} dBm")
    print(f"最终卫星功率: {info.get('final_P_sat', 'N/A'):.4f} dBm")

    print("\nBS 用户 SINR (dB):")
    for i, sinr in enumerate(info.get('sinr_k', [])):
        print(f"  用户 {i}: {10*np.log10(sinr):.2f} dB")

    print("\n卫星用户 SINR (dB):")
    for i, sinr in enumerate(info.get('sinr_j', [])):
        print(f"  用户 {i}: {10*np.log10(sinr):.2f} dB")

    theta_opt = np.angle(np.diag(Phi_opt))
    W_opt = optimizer.w
    P_sat_opt = info['final_P_sat']

    print("\n优化变量:")
    print_info("theta_opt", theta_opt)
    print_info("W_opt", W_opt)
    print_info("P_sat_opt", P_sat_opt)

    print_section("7. 数据封装")

    hk_stack = np.vstack([h_k, h_j])
    hrk_stack = np.vstack([h_k_r, h_j_r])
    hs_stack = np.vstack([h_s_k_eff, h_s_j_eff])

    sample_dict = {
        'theta_opt': theta_opt.astype(np.float32),      # (N_ris,)
        'W_opt': W_opt.astype(np.complex64),           # (Nt, K)
        'P_sat': np.float32(P_sat_opt),                # Scalar
        'hk': hk_stack.astype(np.complex64),           # (K+SK, Nt)
        'hrk': hrk_stack.astype(np.complex64),         # (K+SK, N_ris)
        'GB': G_BS.astype(np.complex64),               # (N_ris, Nt)
        'hs': hs_stack.astype(np.complex64),           # (K+SK, 1)
        'GSAT': G_S_eff.astype(np.complex64),          # (N_ris, 1)
    }

    print("数据封装完成，各字段信息:")
    for key, value in sample_dict.items():
        print_info(key, value)

    return sample_dict, info


def save_sample(sample_dict, filename="test_sample.pt"):
    """保存样本到文件"""
    save_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, filename)
    torch.save(sample_dict, filepath)
    print(f"\n样本已保存至: {filepath}")
    return filepath


def load_sample(filepath):
    """从文件加载样本"""
    sample = torch.load(filepath,weights_only=False, map_location="cpu")
    print(f"\n样本已从 {filepath} 加载")
    for key, value in sample.items():
        print_info(key, value)
    return sample


def main():
    """主测试函数"""
    print(f"\n{'#'*60}")
    print(f"#  单样本生成测试脚本")
    print(f"#  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    # 设置随机种子以便复现
    seed = 42
    print(f"使用随机种子: {seed}")

    try:
        # 生成单个样本
        sample_dict, info = generate_single_sample(seed=seed)

        # 保存样本
        print_section("8. 保存样本")
        filepath = save_sample(sample_dict, filename="test_sample.pt")
        # filepath = os.path.join(os.path.dirname(__file__), "test_output", "test_sample.pt")

        # 验证加载
        print_section("9. 验证加载")
        loaded_sample = load_sample(filepath)

        # 验证数据一致性
        print_section("10. 数据一致性验证")
        all_match = True
        for key in loaded_sample.keys():
            if isinstance(loaded_sample[key], np.ndarray):
                match = np.allclose(loaded_sample[key], loaded_sample[key])
            else:
                match = loaded_sample[key] == loaded_sample[key]
            status = "✓" if match else "✗"
            print(f"{status} {key}: {'匹配' if match else '不匹配'}")
            if not match:
                all_match = False

        print(f"\n{'='*60}")
        print(f"{'测试成功!' if all_match else '测试失败: 数据不一致'}")
        print(f"{'='*60}\n")

        return all_match

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"测试失败，发生错误:")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
