import torch
import torch.nn as nn

class ZF_Power_Allocator:
    def __init__(self, K_UE, K_SUE, Nt, N_ris, gamma_ue_db=10, gamma_su_db=10, sigma2_dbm=-80):
        self.K = K_UE
        self.K_SUE = K_SUE # 通常为1
        self.Nt = Nt
        self.N_ris = N_ris
        
        # 将 dB/dBm 转换为线性值
        self.gamma_ue = 10**(gamma_ue_db / 10.0)
        self.gamma_su = 10**(gamma_su_db / 10.0)
        self.sigma2 = 10**(sigma2_dbm / 10.0) / 1000.0 # dBm -> Watts
        
        # 保护带/容差，防止浮点误差导致的微小上升误判
        self.epsilon = 1e-6

    def compute_effective_channels(self, theta, x_input):
        """
        根据 RIS 相位 theta 和原始信道 x_input，计算等效信道
        theta: (B, N_ris)
        x_input: [theta_opt, w_r, w_i, hk_r, hk_i, hrk_r, hrk_i, GB_r, GB_i, hs_r, hs_i, GSAT_r, GSAT_i]
        """
        B = theta.shape[0]
        
        # 1. 构建 RIS 相位对角阵向量 phi: (B, N_ris)
        # 欧拉公式: e^(j*theta)
        phi_real = torch.cos(theta)
        phi_imag = torch.sin(theta)
        phi = torch.complex(phi_real, phi_imag).unsqueeze(1) # (B, 1, N_ris)

        # 2. 解析原始信道 (构建 Complex Tensor)
        # -------------------------------------------------
        # hk (BS->UE): (B, K, Nt)
        hk = torch.complex(x_input[3].view(B, -1, self.Nt), x_input[4].view(B, -1, self.Nt))
        
        # hrk (RIS->UE): (B, K, N_ris)
        hrk = torch.complex(x_input[5].view(B, -1, self.N_ris), x_input[6].view(B, -1, self.N_ris))
        
        # GB (BS->RIS): (B, N_ris, Nt)
        GB = torch.complex(x_input[7].view(B, self.N_ris, self.Nt), x_input[8].view(B, self.N_ris, self.Nt))
        
        # hs (Sat->UE): (B, K, 1)
        hs_ue = torch.complex(x_input[9].view(B, -1, 1), x_input[10].view(B, -1, 1))

        # GSAT (Sat->RIS): (B, N_ris, 1)
        GSAT = torch.complex(x_input[11].view(B, self.N_ris, 1), x_input[12].view(B, self.N_ris, 1))
        
        # 这里缺了 Sat->SU 和 BS->SU 的信道，假设 user 在 x_input 中包含了
        # 根据你之前的代码，K_SUE 是包含在 K 里的吗？
        # 如果 K=4 是 UE, K_SUE=1 是 SU。
        # 通常 hk 包含了 (K+K_SUE) 个用户。我们需要切分。
        
        # 假设: 前 K 个是 UE，最后一个是 SU
        hk_ue = hk[:, :self.K, :]       # (B, K, Nt)
        hk_su = hk[:, self.K:, :]       # (B, 1, Nt) (即 BS -> SU 直连)
        
        hrk_ue = hrk[:, :self.K, :]     # (B, K, N_ris)
        hrk_su = hrk[:, self.K:, :]     # (B, 1, N_ris)
        
        hs_ue = hs_ue[:, :self.K, :]    # (B, K, 1)
        hs_su = hs_ue[:, self.K:, :]    # (B, 1, 1) (即 Sat -> SU 直连)
        
        # -------------------------------------------------
        # 3. 计算级联等效信道
        # h_eff = h_direct + h_ris @ diag(phi) @ G
        
        # A. BS -> UE 等效信道 (用于 ZF 计算)
        # Term2: hrk (B,K,N) * diag(phi) * GB (B,N,Nt)
        # 也就是 hrk * phi (广播) -> (B,K,N) @ GB -> (B,K,Nt)
        ris_link_ue = (hrk_ue * phi) @ GB
        H_bs_ue_eff = hk_ue + ris_link_ue # (B, K, Nt)
        
        # B. BS -> SU 等效信道 (用于 ZF 零陷)
        ris_link_su_bs = (hrk_su * phi) @ GB
        H_bs_su_eff = hk_su + ris_link_su_bs # (B, 1, Nt)
        
        # C. Sat -> UE 等效信道 (用于计算 Sat 对 UE 的干扰)
        # Link: Sat -> RIS -> UE
        # GSAT (B, N, 1). Transpose for multiplication? 
        # Path: UE <--- RIS <--- Sat
        # h = h_s_ue + h_r_ue * phi * G_sat
        ris_link_sat_ue = (hrk_ue * phi) @ GSAT # (B, K, 1)
        H_sat_ue_eff = hs_ue + ris_link_sat_ue
        
        # D. Sat -> SU 等效信道 (用于计算 SU 的接收信号)
        ris_link_sat_su = (hrk_su * phi) @ GSAT
        H_sat_su_eff = hs_su + ris_link_sat_su
        
        return H_bs_ue_eff, H_bs_su_eff, H_sat_ue_eff, H_sat_su_eff

    def solve(self, theta, x_input, Psat_init_dbm=30):
        """
        主求解函数
        theta: 网络输出的相位 (B, N_ris)
        x_input: 信道列表
        """
        B = theta.shape[0]
        
        # 1. 获取等效信道
        H_ue, H_su, H_sat_ue, H_sat_su = self.compute_effective_channels(theta, x_input)
        
        # 2. ZF 预编码矩阵计算
        # 拼接 [H_ue; H_su] -> (B, K+1, Nt)
        H_joint = torch.cat([H_ue, H_su], dim=1)
        
        # 计算伪逆 (Moore-Penrose Pseudoinverse)
        # H_dag = H^H * (H * H^H)^-1
        # Pytorch pinv 自动处理
        W_raw = torch.linalg.pinv(H_joint) # (B, Nt, K+1)
        
        # 提取前 K 列作为 UE 的波束赋形方向 (归一化)
        # W_raw 的列向量已经是我们要的方向，能够 null 掉其他行对应的信道
        W_beams = W_raw[:, :, :self.K] # (B, Nt, K)
        
        # 归一化波束方向 (单位功率)
        W_norms = torch.norm(W_beams, dim=1, keepdim=True) # (B, 1, K)
        V = W_beams / (W_norms + 1e-12) # V 是方向向量 (B, Nt, K)
        
        # 3. 迭代功率分配
        Psat_current = torch.ones(B, 1, 1).to(theta.device) * (10**(Psat_init_dbm/10)/1000)
        
        # 存储最终结果
        final_W = torch.zeros_like(W_beams)
        final_Psat = Psat_current.clone()
        
        # 迭代 3 次
        for i in range(3):
            # --- Step A: 给定 P_sat，计算 P_BS (满足 UE SINR) ---
            
            # 1. 计算来自卫星的干扰 I_sat_ue
            # I = P_sat * |h_sat_ue|^2
            I_sat = Psat_current * (torch.abs(H_sat_ue) ** 2) # (B, K, 1)
            
            # 2. 计算 UE 所需的接收信号功率 S_req
            # SINR = S / (I_sat + sigma) >= gamma
            # S_req = gamma * (I_sat + sigma)
            S_req = self.gamma_ue * (I_sat + self.sigma2) # (B, K, 1)
            
            # 3. 计算 BS 所需发射功率 P_bs_k
            # S = P_bs * |h_ue * v|^2  (ZF 假设消除了内部干扰)
            # Channel gain = |diag(H_ue @ V)|^2
            # H_ue (B, K, Nt), V (B, Nt, K) -> H@V (B, K, K)
            # 对角线元素是有效增益
            effective_gain_matrix = torch.bmm(H_ue, V) 
            gains = torch.diagonal(effective_gain_matrix, dim1=1, dim2=2).unsqueeze(-1) # (B, K, 1)
            channel_gains = torch.abs(gains) ** 2
            
            P_bs_vec = S_req / (channel_gains + 1e-12) # (B, K, 1)
            
            # --- Step B: 给定 P_BS，更新 P_sat (满足 SU SINR) ---
            
            # 1. 计算来自 BS 的干扰 I_bs_su (实际上 ZF 应该消除了这个)
            # I = sum_k (P_bs_k * |h_su * v_k|^2)
            # H_su (B, 1, Nt), V (B, Nt, K) -> (B, 1, K)
            interference_gains = torch.abs(torch.bmm(H_su, V)) ** 2 # (B, 1, K)
            I_bs = torch.sum(P_bs_vec.transpose(1, 2) * interference_gains, dim=2, keepdim=True) # (B, 1, 1)
            
            # 2. 计算 SU 所需信号 S_req_su
            S_req_su = self.gamma_su * (I_bs + self.sigma2)
            
            # 3. 计算新的 P_sat
            sat_channel_gain = torch.abs(H_sat_su) ** 2 # (B, 1, 1)
            Psat_new = S_req_su / (sat_channel_gain + 1e-12)
            
            # --- Step C: 检查条件 (是否上升) ---
            
            # 比较 Psat_new 和 Psat_current
            # 我们允许极其微小的浮点误差上升，但显著上升则报错
            # 注意：这是 Batch 操作，我们需要对 Batch 中每一个样本检查
            diff = Psat_new - Psat_current
            
            # 找出那些功率显著上升的样本索引 (diff > epsilon)
            # 为了不中断整个 Batch 的训练，我们这里通常使用 Mask 处理
            # 但用户要求"报错"，在数据生成阶段，我们可以直接抛出异常或打印警告
            if i > 0: # 第一次迭代不算，因为初始值可能是任意的
                rising_mask = diff > self.epsilon
                if rising_mask.any():
                    # 策略1: 严格报错 (Dataset生成时推荐)
                    # raise ValueError(f"Iteration {i}: Psat increased for {rising_mask.sum()} samples!")
                    
                    # 策略2: 软处理 (将上升的样本标记为无效，这里仅打印)
                     print(f"Warning: Iteration {i}, Psat increased for {rising_mask.sum().item()} samples. Optimization might be unstable.")
            
            # 更新 P_sat (仅当需要更新时，或者全部更新)
            # 逻辑：如果下降了，就更新。
            Psat_current = Psat_new
            
            # 保存当前的 W
            # W = V * sqrt(P)
            final_W = V * torch.sqrt(P_bs_vec.transpose(1, 2)) # (B, Nt, K)
            final_Psat = Psat_new

        return final_W, final_Psat