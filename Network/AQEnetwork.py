from copy import deepcopy
import numpy as np
import pandas
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import torch
import pprint
import sys
from pathlib import Path
import math


DISABLE_TQDM = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device", flush=True)

def load_complex(dataset_dir, variable_name_real, variable_name_imag):
    return (np.loadtxt(dataset_dir + variable_name_real + ".csv", delimiter=',') +
            1j * np.loadtxt(dataset_dir + variable_name_imag + ".csv", delimiter=','))

# Hau = (K, Nt)       AP  -> UE
# Har = (Nris, Nt)    AP  -> RIS
# Hru = (K, Nris)     RIS -> UE
# W   = (M, K)    UE stream -> AP

class DNN(nn.Module):
    def __init__(self, K_UE,K_SUE, Nt, N_RIS, Nc_enc):
        super(DNN, self).__init__()
        self.K_SUE = K_SUE
        self.K_UE = K_UE
        self.Nt = Nt
        self.N_RIS = N_RIS
        self.Nc_enc = Nc_enc
        ## height: Hout = floor((Hin + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
        ## width:  Wout = floor((Win + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
        p = 0.5 # dropout probability
        H = N_RIS
        self.linear_encoder = nn.Sequential(
            nn.Linear(2*((K_UE+K_SUE)*Nt + (K_UE+K_SUE)*N_RIS + (K_UE+K_SUE) + N_RIS*Nt + N_RIS) + N_RIS, 32*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(32*H),
            nn.Linear(32*H, 16*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(16*H),
            nn.Linear(16*H, 8*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(8*H),
            nn.Linear(8*H, 4*H),
            nn.ReLU(),
            nn.Dropout(p),
            nn.BatchNorm1d(4*H),
            nn.Linear(4*H, H),
        )
    @staticmethod
    def _random_theta_like(theta, mode="uniform"):
        """
        Generate random theta with same shape/device/dtype as theta.
        mode:
          - "uniform": Uniform[-pi, pi)
          - "normal":  Normal(0, 1) (not recommended for angles unless you wrap)
        """
        if mode == "uniform":
            return (2 * math.pi) * torch.rand_like(theta) - math.pi
        elif mode == "normal":
            return torch.randn_like(theta)
        else:
            raise ValueError(f"Unknown theta random mode: {mode}")

    @staticmethod
    def _apply_input_dropout(x_oracle, x_fallback, p_drop: float, training: bool):
        """
        x_oracle:   tensor [B, ...] (e.g., theta_opt or W_opt part)
        x_fallback: tensor [B, ...] (e.g., random theta or W_init)
        p_drop: probability to replace oracle with fallback.
        training: only apply dropout in training mode (recommended).
        """
        if (not training) or (p_drop <= 0.0):
            return x_oracle

        if p_drop >= 1.0:
            return x_fallback

        B = x_oracle.shape[0]
        # mask=1 -> keep oracle; mask=0 -> drop -> use fallback
        keep = (torch.rand(B, device=x_oracle.device) > p_drop).float().view(B, *([1] * (x_oracle.dim() - 1)))
        return keep * x_oracle + (1.0 - keep) * x_fallback
    def forward(self, x, p_theta: float = 0.0, p_w: float = 0.0, theta_random_mode: str = "uniform", w_init_fn=None):
        # --- read oracle inputs ---
        theta_oracle = x[0].float()                         # [B, N_RIS]
        # w_r_oracle   = x[1].float()                         # [B, ..., ...]
        # w_i_oracle   = x[2].float()

        # --- build fallback theta (random) ---
        theta_rand = self._random_theta_like(theta_oracle, mode=theta_random_mode)

        # --- apply theta input dropout ---
        theta = self._apply_input_dropout(theta_oracle, theta_rand, p_theta, self.training)
        # # --- build fallback W (init) ---
        # if p_w > 0.0:
        #     if w_init_fn is not None:
        #         w_r_init, w_i_init = w_init_fn(x)
        #     else:
        #         w_r_init, w_i_init = w_r_oracle, w_i_oracle
        #     w_r_init = w_r_init.to(device=w_r_oracle.device, dtype=w_r_oracle.dtype)
        #     w_i_init = w_i_init.to(device=w_i_oracle.device, dtype=w_i_oracle.dtype)

        #     w_r = self._apply_input_dropout(w_r_oracle, w_r_init, p_w, self.training)
        #     w_i = self._apply_input_dropout(w_i_oracle, w_i_init, p_w, self.training)
        # else:
        #     w_r, w_i = w_r_oracle, w_i_oracle
        # --- the rest unchanged ---
        hk_r   = torch.flatten(x[3], 1).float() # (K+SK,Nt)
        hk_i   = torch.flatten(x[4], 1).float() # (K+SK,Nt)
        hrk_r  = torch.flatten(x[5], 1).float() # (K+SK,Nris)
        hrk_i  = torch.flatten(x[6], 1).float() # (K+SK,Nris)
        GB_r   = torch.flatten(x[7], 1).float() # (N_ris,Nt)
        GB_i   = torch.flatten(x[8], 1).float() # (N_ris,Nt)
        hs_r   = torch.flatten(x[9], 1).float() # (K+SK,1)
        hs_i   = torch.flatten(x[10], 1).float()# (K+SK,1)
        GSAT_r = torch.flatten(x[11], 1).float()# (Nris,1)
        GSAT_i = torch.flatten(x[12], 1).float()# (Nris,1)

        # flatten w_r/w_i AFTER dropout
        # w_r = torch.flatten(w_r, 1)
        # w_i = torch.flatten(w_i, 1)

        x_in = torch.cat((theta, 
                          hk_r, hk_i, hrk_r, hrk_i,
                          GB_r, GB_i, hs_r, hs_i,
                          GSAT_r, GSAT_i), dim=1)

        theta_out = self.linear_encoder(x_in)
        return theta_out


class RisCnnNet(nn.Module):
    def __init__(self, K_UE, K_SUE, Nt, N_ris_rows, N_ris_cols):
        super(RisCnnNet, self).__init__()
        
        self.K = K_UE + K_SUE
        self.Nt = Nt
        self.rows = N_ris_rows
        self.cols = N_ris_cols
        self.N_ris = N_ris_rows * N_ris_cols
        
        # ==========================================
        # 1. CNN 分支：处理 RIS 相关信道 + 输入的 Theta 提示
        # ==========================================
        # 输入通道数计算：
        # GB:    (Nt) * 2(实虚)    = 2*Nt
        # hrk:   (K)  * 2(实虚)    = 2*K
        # GSAT:  (1)  * 2(实虚)    = 2
        # Theta: (1)  * 2(cos/sin) = 2  <--- 新增：输入的初始/Teacher相位
        # 总通道数: 2*(Nt + K + 1 + 1)
        cnn_in_channels = 2 * (Nt + self.K + 1 + 1)
        
        self.cnn_encoder = nn.Sequential(
            # Layer 1: 升维提取特征
            nn.Conv2d(cnn_in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 残差连接或深层特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 压缩
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # CNN 输出扁平化维度
        self.cnn_out_dim = 64 * self.rows * self.cols

        # ==========================================
        # 2. MLP 分支：处理直连信道 (hk, hs)
        # ==========================================
        mlp_in_dim = 2 * self.K * Nt + 2 * self.K
        
        self.direct_link_encoder = nn.Sequential(
            nn.Linear(mlp_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ==========================================
        # 3. 融合层 (Fusion & Output)
        # ==========================================
        fusion_dim = self.cnn_out_dim + 256
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.N_ris) # 输出 theta 相位
        )

    # --- 复用你原有的辅助函数 ---
    @staticmethod
    def _random_theta_like(theta, mode="uniform"):
        if mode == "uniform":
            return (2 * math.pi) * torch.rand_like(theta) - math.pi
        elif mode == "normal":
            return torch.randn_like(theta)
        else:
            raise ValueError(f"Unknown theta random mode: {mode}")

    @staticmethod
    def _apply_input_dropout(x_oracle, x_fallback, p_drop: float, training: bool):
        if (not training) or (p_drop <= 0.0):
            return x_oracle
        if p_drop >= 1.0:
            return x_fallback
        
        B = x_oracle.shape[0]
        # 生成 Mask: 1=Keep Oracle, 0=Use Fallback
        # 注意维度匹配，这里生成 (B, 1) 的 mask，会自动广播
        keep = (torch.rand(B, device=x_oracle.device) > p_drop).float().view(B, 1)
        return keep * x_oracle + (1.0 - keep) * x_fallback

    def forward(self, x, p_theta: float = 0.0, p_w: float = 0.0, theta_random_mode: str = "uniform"):
        # x 是 list，结构：[theta_opt, w_r, w_i, hk_r, ..., GSAT_i]
        bs = x[3].shape[0] 

        # ==========================================
        # A. 处理 Theta 输入 (Simulated Annealing)
        # ==========================================
        theta_oracle = x[0].float() # (Batch, N_ris)
        
        # 生成随机噪声/初始化作为 Fallback
        theta_rand = self._random_theta_like(theta_oracle, mode=theta_random_mode)
        
        # 应用 Input Dropout (模拟退火核心)
        # p_theta 从 0.0 (全Teacher) -> 1.0 (全Random)
        theta_in = self._apply_input_dropout(theta_oracle, theta_rand, p_theta, self.training)
        
        # [关键步骤] 将 Theta 转换为空间特征图 (Batch, 2, Rows, Cols)
        # 使用 cos/sin 编码比直接用角度值更好，避免周期性不连续问题
        theta_cos = torch.cos(theta_in).view(bs, 1, self.rows, self.cols)
        theta_sin = torch.sin(theta_in).view(bs, 1, self.rows, self.cols)
        
        # ==========================================
        # B. 准备 CNN 输入 (RIS信道 + Theta)
        # ==========================================
        
        # 1. HRK (RIS -> User) -> (B, 2K, R, C)
        hrk_r = x[5].view(bs, self.K, self.rows, self.cols)
        hrk_i = x[6].view(bs, self.K, self.rows, self.cols)
        
        # 2. GB (BS -> RIS) -> (B, 2Nt, R, C)
        # 注意转置：原始(N_ris, Nt) -> (Nt, N_ris) -> (Nt, R, C)
        GB_r = x[7].transpose(1, 2).view(bs, self.Nt, self.rows, self.cols)
        GB_i = x[8].transpose(1, 2).view(bs, self.Nt, self.rows, self.cols)

        # 3. GSAT (Sat -> RIS) -> (B, 2, R, C)
        GSAT_r = x[11].transpose(1, 2).view(bs, 1, self.rows, self.cols)
        GSAT_i = x[12].transpose(1, 2).view(bs, 1, self.rows, self.cols)

        # 4. 拼接所有特征到 Channel 维度
        ris_features = torch.cat([
            theta_cos, theta_sin, # 2 通道 (Teacher/Previous Guess)
            GB_r, GB_i,           # 2*Nt 通道
            hrk_r, hrk_i,         # 2*K 通道
            GSAT_r, GSAT_i        # 2 通道
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
        # CNN 提取
        feat_map = self.cnn_encoder(ris_features)
        feat_cnn = torch.flatten(feat_map, 1)
        
        # MLP 提取
        feat_direct = self.direct_link_encoder(direct_features)
        
        # 融合回归
        combined = torch.cat([feat_cnn, feat_direct], dim=1)
        theta_out = self.regressor(combined)
        
        return theta_out
# 概率退火以实现warmup到deployment的转变
def p_theta_schedule(epoch, E_warm=10, E_ramp=60,
                     p_start=0.0, p_max=1.0, mode="linear"):
    """
    epoch: 0-based
    """
    if epoch < E_warm:
        return p_start

    t = (epoch - E_warm) / max(1, E_ramp)
    t = max(0.0, min(1.0, t))

    if mode == "linear":
        p = p_start + (p_max - p_start) * t
    elif mode == "cos":
        p = p_start + (p_max - p_start) * (1 - math.cos(math.pi * t)) / 2
    else:
        raise ValueError("mode must be 'linear' or 'cos'")

    return float(p)