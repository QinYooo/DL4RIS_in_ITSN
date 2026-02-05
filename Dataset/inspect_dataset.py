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

# 分析第2个样本的结构
print("\n第2个样本的结构:")
sample = data[114]
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

print("=" * 60)

# 计算10个样本的相似度
print("\n10个样本的相似度分析:")
num_samples_to_check = min(10, len(data))
sample_indices = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122][:num_samples_to_check]
samples = [data[i] for i in sample_indices]

def estimate_angle_from_channel(hk, H_r):
    """
    通过信道矩阵估计角度
    hk: (K+SK, N_t) 用户到基站信道
    H_r: (N_ris, N_t) RIS到基站信道
    返回: 角度特征向量 (用于相似度比较)
    """
    # 计算信道协方差矩阵
    R_hk = np.array([np.outer(h, h.conj()) for h in hk])  # (K+SK, N_t, N_t)
    R_Hr = np.outer(H_r, H_r.conj())  # (N_ris, N_ris)

    # 提取特征值作为角度相关特征
    eigvals_hk = np.real(np.linalg.eigvals(R_hk))  # (K+SK, N_t)
    eigvals_Hr = np.real(np.linalg.eigvals(R_Hr))   # (N_ris,)

    # 归一化后拼接作为特征
    features = np.concatenate([
        eigvals_hk.flatten(),
        eigvals_Hr.flatten()
    ])
    return features

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def angular_distance(features_list):
    """计算特征之间的角度距离"""
    n = len(features_list)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                dist = np.arccos(np.clip(cosine_similarity(features_list[i], features_list[j]), -1, 1))
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    return distance_matrix

# 提取特征
print(f"\n选择的样本索引: {sample_indices}")
features_list = []
for idx, sample in zip(sample_indices, samples):
    if 'hk' in sample and 'hrk' in sample:
        feat = estimate_angle_from_channel(sample['hk'], sample['hrk'])
        features_list.append(feat)

# 计算相似度矩阵
if features_list:
    similarity_matrix = np.zeros((len(features_list), len(features_list)))
    distance_matrix = angular_distance(features_list)

    for i in range(len(features_list)):
        for j in range(len(features_list)):
            similarity_matrix[i, j] = cosine_similarity(features_list[i], features_list[j])

    print("\n余弦相似度矩阵:")
    print("样本编号  " + "  ".join([f"{i:5d}" for i in sample_indices]))
    for i, row in enumerate(similarity_matrix):
        print(f"{sample_indices[i]:6d}  " + "  ".join([f"{v:5.3f}" for v in row]))

    print("\n角度距离矩阵 (弧度):")
    print("样本编号  " + "  ".join([f"{i:5d}" for i in sample_indices]))
    for i, row in enumerate(distance_matrix):
        print(f"{sample_indices[i]:6d}  " + "  ".join([f"{v:5.3f}" for v in row]))

    # 统计
    n = len(distance_matrix)
    upper_tri = distance_matrix[np.triu_indices(n, k=1)]
    print(f"\n角度距离统计:")
    print(f"  最小距离: {np.min(upper_tri):.4f} 弧度 ({np.degrees(np.min(upper_tri)):.2f}°)")
    print(f"  最大距离: {np.max(upper_tri):.4f} 弧度 ({np.degrees(np.max(upper_tri)):.2f}°)")
    print(f"  平均距离: {np.mean(upper_tri):.4f} 弧度 ({np.degrees(np.mean(upper_tri)):.2f}°)")
    print(f"  标准差  : {np.std(upper_tri):.4f} 弧度 ({np.degrees(np.std(upper_tri)):.2f}°)")
else:
    print("未找到用于计算相似度的字段 'hk' 和 'H_r'")

print("=" * 60)
