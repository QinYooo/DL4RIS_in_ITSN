import torch
import os
from Network.RISCNN import RisCnnNet
from Network.ZF_Power_Allocater import ZF_Power_Allocator
from Trainer.train import RisTrainer, IntegratedRISModel

# 假设参数
CONFIG = {
    'K_UE': 4, 'K_SUE': 1, 'Nt': 36,
    'Rows': 10, 'Cols': 10,
    'Batch_Size': 64, 'Epochs': 50, 'LR': 1e-3
}
N_ris = CONFIG['Rows'] * CONFIG['Cols']
K_total = CONFIG['K_UE'] + CONFIG['K_SUE']  # 4 + 1 = 5

# ==========================================
# 1. 数据集类 (从 .pt 文件加载真实数据)
# ==========================================
class RisDataset(torch.utils.data.Dataset):
    """
    从 Dataset/dataset_ris_itsn_raw/train_dataset_mp_raw.pt 加载预生成的数据集

    每个样本包含:
        - theta_opt: (N_ris,) RIS 相位配置 (label)
        - W_opt: (Nt, K) 基站波束成形 (label, 占位)
        - P_sat: scalar 卫星发射功率 (label, 占位)
        - hk: (K_total, Nt) 基站到用户信道
        - hrk: (K_total, N_ris) RIS 到用户信道
        - GB: (N_ris, Nt) RIS 到基站信道
        - hs: (K_total, 1) 卫星到用户信道
        - GSAT: (N_ris, 1) RIS 到卫星信道
    """
    def __init__(self, data_path, split='train', train_ratio=0.8):
        """
        Args:
            data_path: .pt 数据文件路径
            split: 'train' 或 'val'
            train_ratio: 训练集比例
        """
        self.data = torch.load(data_path, weights_only=False)
        print(f"Loaded {len(self.data)} samples from {data_path}")

        # 划分训练/验证集
        split_idx = int(len(self.data) * train_ratio)
        if split == 'train':
            self.samples = self.data[:split_idx]
            print(f"  Split: train ({len(self.samples)} samples)")
        else:
            self.samples = self.data[split_idx:]
            print(f"  Split: val ({len(self.samples)} samples)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回格式: [theta_opt, w_r, w_i, hk_r, hk_i, hrk_r, hrk_i, GB_r, GB_i, hs_r, hs_i, GSAT_r, GSAT_i]
        所有 tensor 类型为 float32
        """
        sample = self.samples[idx]

        # 1. RIS 相位 (label) - 已经是 float32
        theta = torch.from_numpy(sample['theta_opt']).float()  # (N_ris,)

        # 2. W_opt 拆分实虚部 (占位用, 网络不使用)
        W = sample['W_opt']
        w_r = torch.from_numpy(W.real).float()  # (Nt, K_total)
        w_i = torch.from_numpy(W.imag).float()  # (Nt, K_total)

        # 3. hk (基站到用户信道) - shape: (K_total, Nt)
        hk = sample['hk']
        hk_r = torch.from_numpy(hk.real).float()   # (K_total, Nt)
        hk_i = torch.from_numpy(hk.imag).float()   # (K_total, Nt)

        # 4. hrk (RIS到用户信道) - shape: (K_total, N_ris)
        hrk = sample['hrk']
        hrk_r = torch.from_numpy(hrk.real).float() # (K_total, N_ris)
        hrk_i = torch.from_numpy(hrk.imag).float() # (K_total, N_ris)

        # 5. GB (RIS到基站信道) - shape: (N_ris, Nt)
        GB = sample['GB']
        GB_r = torch.from_numpy(GB.real).float()   # (N_ris, Nt)
        GB_i = torch.from_numpy(GB.imag).float()   # (N_ris, Nt)

        # 6. hs (卫星到用户信道) - shape: (K_total, 1)
        hs = sample['hs']
        hs_r = torch.from_numpy(hs.real).float()   # (K_total, 1)
        hs_i = torch.from_numpy(hs.imag).float()   # (K_total, 1)

        # 7. GSAT (RIS到卫星信道) - shape: (N_ris, 1)
        GSAT = sample['GSAT']
        GSAT_r = torch.from_numpy(GSAT.real).float() # (N_ris, 1)
        GSAT_i = torch.from_numpy(GSAT.imag).float() # (N_ris, 1)

        # 返回 list，顺序必须与网络 forward 一致
        return [theta, w_r, w_i, hk_r, hk_i, hrk_r, hrk_i, GB_r, GB_i, hs_r, hs_i, GSAT_r, GSAT_i]

# 数据集路径
DATA_PATH = os.path.join(os.path.dirname(__file__), "Dataset", "dataset_ris_itsn_raw", "train_dataset_mp_raw.pt")

# 创建训练集和验证集 (按 8:2 划分)
train_ds = RisDataset(DATA_PATH, split='train', train_ratio=0.8)
val_ds = RisDataset(DATA_PATH, split='val', train_ratio=0.8)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG['Batch_Size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=CONFIG['Batch_Size'])

# ==========================================
# 2. 初始化模型
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 实例化 CNN
cnn_net = RisCnnNet(K_UE=CONFIG['K_UE'], K_SUE=CONFIG['K_SUE'], Nt=CONFIG['Nt'], 
                    N_ris_rows=CONFIG['Rows'], N_ris_cols=CONFIG['Cols'])

# 实例化 ZF Solver (物理层)
zf_solver = ZF_Power_Allocator(K_UE=CONFIG['K_UE'], K_SUE=CONFIG['K_SUE'], 
                               Nt=CONFIG['Nt'], N_ris=N_ris)

# 组合
model = IntegratedRISModel(cnn_net, zf_solver).to(device)

# ==========================================
# 3. 开始训练
# ==========================================
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
trainer = RisTrainer(model, train_loader, val_loader, optimizer, device)

print("Starting Training...")
for epoch in range(CONFIG['Epochs']):
    train_loss, train_power = trainer.train_epoch(epoch, CONFIG['Epochs'])
    val_phase_err, val_power = trainer.validate()
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train P={train_power:.2f} | "
          f"Val PhaseErr={val_phase_err:.4f}, Val P={val_power:.2f}")

print("Training Finished.")