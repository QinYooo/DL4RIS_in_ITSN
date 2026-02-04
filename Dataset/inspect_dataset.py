import torch
import numpy as np
import os

# 数据集路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset_ris_itsn_raw")
DATA_FILE = os.path.join(DATA_DIR, "train_dataset_mp_raw.pt")

print(f"数据集路径: {DATA_FILE}")
print("=" * 60)

# 加载数据
data = torch.load(DATA_FILE, weights_only=False)
print(f"总样本数: {len(data)}")
print("=" * 60)

# 分析第一个样本的结构
print("\n第一个样本的结构:")
sample = data[113]
for key, value in sample.items():
    if isinstance(value, (np.ndarray, torch.Tensor)):
        print(f"  {key:12s}: shape={tuple(value.shape)}, dtype={value.dtype}, "
              f"range=[{np.min(value):.4e}, {np.max(value):.4e}]")
    else:
        print(f"  {key:12s}: {value}, type={type(value).__name__}")

print("=" * 60)

# 统计所有样本的数值范围
print("\n所有样本的统计信息:")
num_samples = len(data)
for key in sample.keys():
    all_values = [s[key] for s in data]
    if isinstance(sample[key], (np.ndarray, torch.Tensor)):
        # 复数数据处理
        if np.iscomplexobj(sample[key]):
            all_values = np.array([np.abs(v) for v in all_values])
            print(f"{key:12s}: abs mean={np.mean(all_values):.4e}, "
                  f"abs std={np.std(all_values):.4e}, "
                  f"abs range=[{np.min(all_values):.4e}, {np.max(all_values):.4e}]")
        else:
            all_values = np.array(all_values)
            print(f"{key:12s}: mean={np.mean(all_values):.4e}, "
                  f"std={np.std(all_values):.4e}, "
                  f"range=[{np.min(all_values):.4e}, {np.max(all_values):.4e}]")

print("=" * 60)

# 显示场景信息（需要从数据维度推断）
if 'hk' in sample:
    K_plus_SK = sample['hk'].shape[0]
    N_t = sample['hk'].shape[1]
    N_ris = sample['theta_opt'].shape[0]
    print(f"\n推断的场景配置:")
    print(f"  N_t  (基站天线数)  : {N_t}")
    print(f"  N    (RIS 元素数)  : {N_ris}")
    print(f"  K+SK (用户总数)    : {K_plus_SK}")
