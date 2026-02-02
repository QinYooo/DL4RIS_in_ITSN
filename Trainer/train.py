import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# 引入之前定义的类 (假设它们在同一个文件中或已import)
from Network.ZF_Power_Allocater import ZF_Power_Allocator
from Network.RISCNN import RisCnnNet

class IntegratedRISModel(nn.Module):
    def __init__(self, cnn_net, zf_solver):
        super(IntegratedRISModel, self).__init__()
        self.cnn_net = cnn_net
        self.zf_solver = zf_solver

    def forward(self, x_input, p_theta=0.0, theta_random_mode="uniform"):
        # 1. 神经网络预测 RIS 相位
        # theta_pred: (Batch, N_ris)
        theta_pred_raw = self.cnn_net(x_input, p_theta=p_theta, theta_random_mode=theta_random_mode)
        
        # 归一化相位到 [-pi, pi] (虽然 cos/sin 不关心这个，但为了物理意义明确)
        theta_pred = torch.atan2(torch.sin(theta_pred_raw), torch.cos(theta_pred_raw))

        # 2. 进入可微 ZF 层计算 W 和 P_sat
        # 注意：这里我们使用 try-except 的逻辑在 Dataset 生成时很有用，
        # 但在训练(Backprop)时，如果报错会打断训练。
        # 策略：训练时我们假设物理约束大致满足，或者允许 Power 变大作为惩罚。
        # 我们修改 ZF Solver 的逻辑，使其只返回 Power，不报错，而是返回巨大的 Loss。
        
        W_opt, Psat_opt = self.zf_solver.solve(theta_pred, x_input)
        
        # 3. 计算总功耗 (Batch, )
        # Power_BS = sum(|W|^2)
        P_bs = torch.sum(torch.abs(W_opt)**2, dim=(1, 2)) 
        P_sat = Psat_opt.squeeze()
        
        P_total = P_bs + P_sat
        
        return theta_pred, P_total, P_bs, P_sat

class RisHybridLoss(nn.Module):
    def __init__(self, lambda_power=0.1):
        super(RisHybridLoss, self).__init__()
        self.lambda_power = lambda_power # 权衡 模仿损失 和 功耗损失 的权重
        
    def forward(self, theta_pred, theta_teacher, power_pred):
        # 1. 监督损失 (Supervised Loss): 
        # 最大化预测相位和Teacher相位的余弦相似度
        # Loss = 1 - mean(cos(theta_pred - theta_teacher))
        # 当两者完全重合时，cos=1, Loss=0；反向时 cos=-1, Loss=2
        phase_diff = theta_pred - theta_teacher
        loss_phase = 1.0 - torch.mean(torch.cos(phase_diff))
        
        # 2. 功耗损失 (Unsupervised Power Loss)
        # 直接最小化总功耗
        loss_power = torch.mean(power_pred)
        
        # 3. 总损失
        # 在训练初期主要靠 loss_phase 引导，后期 loss_power 帮助微调
        total_loss = loss_phase + self.lambda_power * loss_power
        
        return total_loss, loss_phase, loss_power

class RisTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = RisHybridLoss(lambda_power=0.05) # Power 的数值通常较大，权重给小点

    def p_theta_schedule(self, epoch, total_epochs):
        """
        退火策略：
        - 前 20% Epoch: p_theta = 0 (完全 Teacher Forcing, 快速学会模仿)
        - 中间 60% Epoch: 线性增加到 1.0
        - 后 20% Epoch: p_theta = 1 (完全 Inference 模式, 适应真实场景)
        """
        warmup_end = int(0.2 * total_epochs)
        anneal_end = int(0.8 * total_epochs)
        
        if epoch < warmup_end:
            return 0.0
        elif epoch > anneal_end:
            return 1.0
        else:
            return (epoch - warmup_end) / (anneal_end - warmup_end)

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0
        epoch_power = 0
        
        # 计算当前 epoch 的 mask 概率
        current_p = self.p_theta_schedule(epoch, total_epochs)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [p={current_p:.2f}]", leave=False)
        
        for batch_idx, data_list in enumerate(pbar):
            # 1. 数据搬运到 GPU
            # data_list 里的每个 tensor 都要 .to(device)
            data_list = [d.to(self.device).float() for d in data_list]
            
            # theta_opt 是列表的第0个元素
            theta_teacher = data_list[0]
            
            # 2. 梯度清零
            self.optimizer.zero_grad()
            
            # 3. 前向传播
            # 即使在 Power 爆炸时，只要 PyTorch 计算图不断，梯度就能传回来
            theta_pred, P_total, _, _ = self.model(data_list, p_theta=current_p)
            
            # 4. 计算损失
            loss, l_phase, l_power = self.criterion(theta_pred, theta_teacher, P_total)
            
            # 5. 反向传播与优化
            loss.backward()
            
            # 梯度裁剪 (防止 ZF 求逆导致梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录日志
            epoch_loss += loss.item()
            epoch_power += torch.mean(P_total).item()
            
            # 更新进度条后缀
            pbar.set_postfix({'L_Ph': f"{l_phase.item():.4f}", 'P_Avg': f"{torch.mean(P_total).item():.2f}"})

        return epoch_loss / len(self.train_loader), epoch_power / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_power = 0
        val_phase_err = 0
        
        with torch.no_grad():
            for data_list in self.val_loader:
                data_list = [d.to(self.device).float() for d in data_list]
                theta_teacher = data_list[0]
                
                # 验证时 p_theta 必须为 1.0 (完全不看答案)
                theta_pred, P_total, _, _ = self.model(data_list, p_theta=1.0)
                
                val_power += torch.mean(P_total).item()
                val_phase_err += (1.0 - torch.mean(torch.cos(theta_pred - theta_teacher))).item()
                
        return val_phase_err / len(self.val_loader), val_power / len(self.val_loader)