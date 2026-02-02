import torch
import torch.nn as nn
import math

class RisCnnNet(nn.Module):
    def __init__(self, K_UE=4, K_SUE=1, Nt=32, N_ris_rows=10, N_ris_cols=10):
        super(RisCnnNet, self).__init__()
        
        self.K = K_UE + K_SUE  # 总用户数 = 5
        self.Nt = Nt           # 基站天线数 = 32
        self.rows = N_ris_rows # 10
        self.cols = N_ris_cols # 10
        self.N_ris = N_ris_rows * N_ris_cols # 100
        
        # ==========================================
        # 1. CNN 分支：处理 RIS 相关信道 + 输入的 Theta 提示
        # ==========================================
        # 输入通道数计算:
        # Theta提示: (1) * 2(cos/sin) = 2
        # GB (BS->RIS): (Nt) * 2(实虚) = 32 * 2 = 64
        # hrk (RIS->User): (K) * 2(实虚) = 5 * 2 = 10
        # GSAT (Sat->RIS): (1) * 2(实虚) = 2
        # ------------------------------------------
        # 总通道数 = 2 + 64 + 10 + 2 = 78
        cnn_in_channels = 2 + 2*self.Nt + 2*self.K + 2
        
        self.cnn_encoder = nn.Sequential(
            # Layer 1: 保持空间尺寸 10x10，通道数 78 -> 64
            nn.Conv2d(cnn_in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 特征提取，保持 10x10
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 瓶颈层，保持 10x10
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # CNN 输出扁平化维度: 64通道 * 10 * 10 = 6400
        self.cnn_out_dim = 64 * self.rows * self.cols

        # ==========================================
        # 2. MLP 分支：处理直连信道 (hk, hs)
        # ==========================================
        # 输入维度:
        # hk (BS->User): (K, Nt) * 2 = 5 * 32 * 2 = 320
        # hs (Sat->User): (K, 1) * 2 = 5 * 1 * 2 = 10
        # Total = 330
        mlp_in_dim = 2 * self.K * Nt + 2 * self.K
        
        self.direct_link_encoder = nn.Sequential(
            nn.Linear(mlp_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ==========================================
        # 3. 融合层 (Fusion & Output)
        # ==========================================
        # 融合维度: 6400 + 256 = 6656
        fusion_dim = self.cnn_out_dim + 256
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.N_ris) # 输出 100 个相位值
        )

    # --- 辅助方法: 生成随机相位 ---
    @staticmethod
    def _random_theta_like(theta, mode="uniform"):
        if mode == "uniform":
            return (2 * math.pi) * torch.rand_like(theta) - math.pi
        elif mode == "normal":
            return torch.randn_like(theta)
        else:
            raise ValueError(f"Unknown theta random mode: {mode}")

    # --- 辅助方法: Input Dropout (模拟退火核心) ---
    @staticmethod
    def _apply_input_dropout(x_oracle, x_fallback, p_drop: float, training: bool):
        if (not training) or (p_drop <= 0.0):
            return x_oracle
        if p_drop >= 1.0:
            return x_fallback
        
        B = x_oracle.shape[0]
        # 生成 Mask: 1=Keep Oracle, 0=Use Fallback
        keep = (torch.rand(B, device=x_oracle.device) > p_drop).float().view(B, 1)
        return keep * x_oracle + (1.0 - keep) * x_fallback

    def forward(self, x, p_theta: float = 0.0, theta_random_mode: str = "uniform"):
        # x 是 list，包含 [theta_opt, w_r, w_i, hk_r, hk_i, hrk_r, hrk_i, GB_r, GB_i, hs_r, hs_i, GSAT_r, GSAT_i]
        bs = x[3].shape[0] 

        # ==========================================
        # A. 处理 Theta 输入 (Simulated Annealing)
        # ==========================================
        theta_oracle = x[0].float() # (Batch, 100)
        
        # 1. 生成随机噪声作为 Fallback
        theta_rand = self._random_theta_like(theta_oracle, mode=theta_random_mode)
        
        # 2. 应用 Input Dropout
        # 训练初期 p_theta=0 (使用 Teacher), 后期 p_theta=1 (使用 Random)
        theta_in = self._apply_input_dropout(theta_oracle, theta_rand, p_theta, self.training)
        
        # 3. 将 Theta 转换为空间特征图 (Batch, 1, 10, 10)
        theta_cos = torch.cos(theta_in).view(bs, 1, self.rows, self.cols)
        theta_sin = torch.sin(theta_in).view(bs, 1, self.rows, self.cols)
        
        # ==========================================
        # B. 准备 CNN 输入 (RIS信道 + Theta)
        # ==========================================
        
        # 1. hrk (Batch, 5, 100) -> (Batch, 5, 10, 10)
        hrk_r = x[5].view(bs, self.K, self.rows, self.cols)
        hrk_i = x[6].view(bs, self.K, self.rows, self.cols)
        
        # 2. GB (Batch, 100, 32) -> transpose -> (Batch, 32, 10, 10)
        GB_r = x[7].transpose(1, 2).view(bs, self.Nt, self.rows, self.cols)
        GB_i = x[8].transpose(1, 2).view(bs, self.Nt, self.rows, self.cols)

        # 3. GSAT (Batch, 100, 1) -> transpose -> (Batch, 1, 10, 10)
        GSAT_r = x[11].transpose(1, 2).view(bs, 1, self.rows, self.cols)
        GSAT_i = x[12].transpose(1, 2).view(bs, 1, self.rows, self.cols)

        # 4. 拼接所有特征 (Batch, 78, 10, 10)
        ris_features = torch.cat([
            theta_cos, theta_sin, # 2
            GB_r, GB_i,           # 64
            hrk_r, hrk_i,         # 10
            GSAT_r, GSAT_i        # 2
        ], dim=1) 
        
        # ==========================================
        # C. 准备 MLP 输入 (直连信道)
        # ==========================================
        hk_r = torch.flatten(x[3], 1)
        hk_i = torch.flatten(x[4], 1)
        hs_r = torch.flatten(x[9], 1)
        hs_i = torch.flatten(x[10], 1)
        
        direct_features = torch.cat([hk_r, hk_i, hs_r, hs_i], dim=1)

        # ==========================================
        # D. 前向计算
        # ==========================================
        feat_map = self.cnn_encoder(ris_features) # (Batch, 64, 10, 10)
        feat_cnn = torch.flatten(feat_map, 1)     # (Batch, 6400)
        
        feat_direct = self.direct_link_encoder(direct_features) # (Batch, 256)
        
        combined = torch.cat([feat_cnn, feat_direct], dim=1) # (Batch, 6656)
        theta_out = self.regressor(combined) # (Batch, 100)
        
        return theta_out

# ==========================================
# 简单的测试代码 (验证维度是否匹配)
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化网络
    net = RisCnnNet(K_UE=4, K_SUE=1, Nt=32, N_ris_rows=10, N_ris_cols=10).to(device)
    print(f"网络已初始化: {device}")
    
    # 构造虚假数据 (Batch Size = 16)
    B = 16
    K, Nt, Nris = 5, 32, 100
    
    # 模拟 x 列表中的各个张量
    theta_opt = torch.randn(B, Nris).to(device)
    w_r = torch.randn(B, K, Nt).to(device) # 不会被使用，但占位
    w_i = torch.randn(B, K, Nt).to(device)
    
    hk_r  = torch.randn(B, K, Nt).to(device)
    hk_i  = torch.randn(B, K, Nt).to(device)
    
    hrk_r = torch.randn(B, K, Nris).to(device)
    hrk_i = torch.randn(B, K, Nris).to(device)
    
    GB_r  = torch.randn(B, Nris, Nt).to(device)
    GB_i  = torch.randn(B, Nris, Nt).to(device)
    
    hs_r  = torch.randn(B, K, 1).to(device)
    hs_i  = torch.randn(B, K, 1).to(device)
    
    GSAT_r = torch.randn(B, Nris, 1).to(device)
    GSAT_i = torch.randn(B, Nris, 1).to(device)
    
    # 打包成 list
    x_input = [theta_opt, w_r, w_i, hk_r, hk_i, hrk_r, hrk_i, 
               GB_r, GB_i, hs_r, hs_i, GSAT_r, GSAT_i]
    
    # 1. 测试 Teacher 模式 (p_theta = 0)
    net.train()
    out_teacher = net(x_input, p_theta=0.0)
    print(f"Teacher 模式输出尺寸: {out_teacher.shape}") # 预期 (16, 100)
    
    # 2. 测试 Random 模式 (p_theta = 1.0)
    out_random = net(x_input, p_theta=1.0)
    print(f"Random 模式输出尺寸: {out_random.shape}")   # 预期 (16, 100)
    
    print("模型维度验证通过！")