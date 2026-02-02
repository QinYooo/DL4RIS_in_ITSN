import torch
from Network.RISCNN import RisCnnNet
from Network.ZF_Power_Allocater import ZF_Power_Allocator
from Trainer.train import RisTrainer, IntegratedRISModel

# 假设参数
CONFIG = {
    'K_UE': 4, 'K_SUE': 1, 'Nt': 32, 
    'Rows': 10, 'Cols': 10,
    'Batch_Size': 64, 'Epochs': 50, 'LR': 1e-3
}
N_ris = CONFIG['Rows'] * CONFIG['Cols']

# ==========================================
# 1. 准备数据 (Mock Dataset)
# ==========================================
# 在实际使用中，你需要写一个 torch.utils.data.Dataset 来加载你的 .csv 数据
class RisDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # 这里返回一个包含所有 tensor 的 list，顺序必须和网络 forward 一致
        # [theta_opt, w_r, w_i, hk_r, hk_i, hrk_r, hrk_i, GB_r, GB_i, hs_r, hs_i, GSAT_r, GSAT_i]
        # 这里生成随机数据演示
        return [torch.randn(N_ris), torch.randn(1), torch.randn(1), # w_r, w_i 占位
                torch.randn(CONFIG['K_UE']+CONFIG['K_SUE'], CONFIG['Nt']), # hk_r
                torch.randn(CONFIG['K_UE']+CONFIG['K_SUE'], CONFIG['Nt']), # hk_i
                torch.randn(CONFIG['K_UE']+CONFIG['K_SUE'], N_ris),       # hrk_r
                torch.randn(CONFIG['K_UE']+CONFIG['K_SUE'], N_ris),       # hrk_i
                torch.randn(N_ris, CONFIG['Nt']),                         # GB_r
                torch.randn(N_ris, CONFIG['Nt']),                         # GB_i
                torch.randn(CONFIG['K_UE']+CONFIG['K_SUE'], 1),           # hs_r
                torch.randn(CONFIG['K_UE']+CONFIG['K_SUE'], 1),           # hs_i
                torch.randn(N_ris, 1),                                    # GSAT_r
                torch.randn(N_ris, 1)]                                    # GSAT_i

train_ds = RisDataset(num_samples=2000)
val_ds = RisDataset(num_samples=200)
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