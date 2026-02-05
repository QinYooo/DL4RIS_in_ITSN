import torch
import os
from Network.RISCNN import RisCnnNet
from Network.ZF_Power_Allocater import ZF_Power_Allocator
from Trainer.train import SemiSupervisedTrainer, IntegratedRISModel

# ==========================================
# 配置参数
# ==========================================
CONFIG = {
    'K_UE': 4, 'K_SUE': 1, 'Nt': 36,
    'Rows': 10, 'Cols': 10,
    'Batch_Size': 64,
    'Stage1_Epochs': 100,
    'Stage2_Epochs': 400,
    'LR': 1e-3,
    'Log_Dir': './runs/semi_supervised'  # TensorBoard 日志目录
}

N_ris = CONFIG['Rows'] * CONFIG['Cols']
K_total = CONFIG['K_UE'] + CONFIG['K_SUE']  # 4 + 1 = 5

# 数据集划分参数
NUM_UNLABELED_TRAIN = 6000   # 无标签训练样本数
NUM_UNLABELED_EVAL = 2000    # 无标签评估样本数

# ==========================================
# 1. 数据集类 (支持半监督三划分)
# ==========================================
class SemiSupervisedRisDataset(torch.utils.data.Dataset):
    """
    支持半监督训练的数据集
    - labeled: 有标签样本 (含 theta_opt)，来自独立数据文件
    - unlabeled_train: 无标签样本 (仅信道数据)，来自独立数据文件
    - unlabeled_eval: 无标签样本 (仅用于评估)，来自独立数据文件
    """
    def __init__(self, data_path, split='labeled',
                 num_unlabeled_train=NUM_UNLABELED_TRAIN,
                 num_unlabeled_eval=NUM_UNLABELED_EVAL):
        """
        Args:
            data_path: .pt 数据文件路径
            split: 'labeled', 'unlabeled_train', 'unlabeled_eval'
            num_unlabeled_train: 无标签训练样本数
            num_unlabeled_eval: 无标签评估样本数

        数据划分：
        - labeled: 加载全部有标签数据
        - unlabeled_train: [0:num_unlabeled_train] (无标签训练)
        - unlabeled_eval: [num_unlabeled_train:] (无标签评估)
        """
        self.data = torch.load(data_path, weights_only=False)
        print(f"Loaded {len(self.data)} samples from {data_path}")

        # 划分数据集
        if split == 'labeled':
            # labeled 数据使用全部样本（因为来自独立文件）
            self.samples = self.data
            self.use_label = True
            print(f"  Split: labeled ({len(self.samples)} samples)")
        elif split == 'unlabeled_train':
            self.samples = self.data[:num_unlabeled_train]
            self.use_label = False
            print(f"  Split: unlabeled_train ({len(self.samples)} samples)")
        elif split == 'unlabeled_eval':
            self.samples = self.data[num_unlabeled_train:]
            self.use_label = False
            print(f"  Split: unlabeled_eval ({len(self.samples)} samples)")
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回格式: [theta_opt, w_r, w_i, hk_r, hk_i, hrk_r, hrk_i, GB_r, GB_i, hs_r, hs_i, GSAT_r, GSAT_i]
        所有 tensor 类型为 float32

        注意：无标签数据中 theta_opt 会被填充为占位符，因为训练时 p_theta=1.0 不会使用它
        """
        sample = self.samples[idx]

        # 1. RIS 相位 (label) - 已经是 float32
        if self.use_label:
            theta = torch.from_numpy(sample['theta_opt']).float()  # (N_ris,)
        else:
            # 无标签数据：填充零占位符 (不会被使用，因为 p_theta=1.0)
            theta = torch.zeros(N_ris).float()

        # 2. W_opt 拆分实虚部 (占位用, 网络不使用)
        W = sample['W_opt']
        w_r = 1e6 * torch.from_numpy(W.real).float()  # (Nt, K_total)
        w_i = 1e6 * torch.from_numpy(W.imag).float()  # (Nt, K_total)

        # 3. hk (基站到用户信道) - shape: (K_total, Nt)
        hk = sample['hk']
        hk_r = 1e6 * torch.from_numpy(hk.real).float()   # (K_total, Nt)
        hk_i = 1e6 * torch.from_numpy(hk.imag).float()   # (K_total, Nt)

        # 4. hrk (RIS到用户信道) - shape: (K_total, N_ris)
        hrk = sample['hrk']
        hrk_r = 1e6 * torch.from_numpy(hrk.real).float() # (K_total, N_ris)
        hrk_i = 1e6 * torch.from_numpy(hrk.imag).float() # (K_total, N_ris)

        # 5. GB (RIS到基站信道) - shape: (N_ris, Nt)
        GB = sample['GB']
        GB_r = torch.from_numpy(GB.real).float()   # (N_ris, Nt)
        GB_i = torch.from_numpy(GB.imag).float()   # (N_ris, Nt)

        # 6. hs (卫星到用户信道) - shape: (K_total, 1)
        hs = sample['hs']
        hs_r = 1e6 * torch.from_numpy(hs.real).float()   # (K_total, 1)
        hs_i = 1e6 * torch.from_numpy(hs.imag).float()   # (K_total, 1)

        # 7. GSAT (RIS到卫星信道) - shape: (N_ris, 1)
        GSAT = sample['GSAT']
        GSAT_r = torch.from_numpy(GSAT.real).float() # (N_ris, 1)
        GSAT_i = torch.from_numpy(GSAT.imag).float() # (N_ris, 1)

        # 返回 list，顺序必须与网络 forward 一致
        return [theta, w_r, w_i, hk_r, hk_i, hrk_r, hrk_i, GB_r, GB_i, hs_r, hs_i, GSAT_r, GSAT_i]

# 数据集路径
LABELED_DATA_PATH = os.path.join(os.path.dirname(__file__), "Dataset", "dataset_ris_itsn_raw", "train_dataset_mp_raw.pt")
UNLABELED_DATA_PATH = os.path.join(os.path.dirname(__file__), "Dataset", "dataset_ris_itsn_raw", "train_dataset_mp_raw_nonlabel.pt")

# ==========================================
# 2. 创建三个数据集和 DataLoader
# ==========================================
print("\n=== Dataset Partition ===")
labeled_ds = SemiSupervisedRisDataset(LABELED_DATA_PATH, split='labeled')
unlabeled_train_ds = SemiSupervisedRisDataset(UNLABELED_DATA_PATH, split='unlabeled_train')
unlabeled_eval_ds = SemiSupervisedRisDataset(UNLABELED_DATA_PATH, split='unlabeled_eval')

# 创建 DataLoader
labeled_loader = torch.utils.data.DataLoader(
    labeled_ds, batch_size=CONFIG['Batch_Size'], shuffle=True, num_workers=0
)
unlabeled_train_loader = torch.utils.data.DataLoader(
    unlabeled_train_ds, batch_size=CONFIG['Batch_Size'], shuffle=True, num_workers=0
)
unlabeled_eval_loader = torch.utils.data.DataLoader(
    unlabeled_eval_ds, batch_size=CONFIG['Batch_Size'], num_workers=0
)

# ==========================================
# 3. 初始化模型
# ==========================================
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"\nDevice: {device}")

# 实例化 CNN
cnn_net = RisCnnNet(K_UE=CONFIG['K_UE'], K_SUE=CONFIG['K_SUE'], Nt=CONFIG['Nt'],
                    N_ris_rows=CONFIG['Rows'], N_ris_cols=CONFIG['Cols'])

# 实例化 ZF Solver (物理层)
Bw = 10e6
k_boltz = 1.38064852e-23
T0 = 290
F_dB = 3
P_noise = k_boltz * T0 * Bw * (10**(F_dB/10))
from numpy import log10
sigma2_dbm = 10 * log10(torch.tensor(P_noise)) + 30  # 转为 dBm
zf_solver = ZF_Power_Allocator(K_UE=CONFIG['K_UE'], K_SUE=CONFIG['K_SUE'],
                               Nt=CONFIG['Nt'], N_ris=N_ris, sigma2_dbm= sigma2_dbm.item())

# 组合
model = IntegratedRISModel(cnn_net, zf_solver).to(device)

# ==========================================
# 4. 初始化优化器和训练器
# ==========================================
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])

# 创建半监督训练器
trainer = SemiSupervisedTrainer(
    model=model,
    labeled_loader=labeled_loader,
    unlabeled_train_loader=unlabeled_train_loader,
    unlabeled_eval_loader=unlabeled_eval_loader,
    optimizer=optimizer,
    device=device,
    stage1_epochs=CONFIG['Stage1_Epochs'],
    stage2_epochs=CONFIG['Stage2_Epochs'],
    log_dir=CONFIG['Log_Dir']
)

# ==========================================
# 5. 执行两阶段训练
# ==========================================
history = trainer.run_training()

# ==========================================
# 6. (可选) 保存模型
# ==========================================
# torch.save(model.state_dict(), 'semi_supervised_ris_model.pt')
print("\nTraining completed!")
