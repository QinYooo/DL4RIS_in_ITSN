import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# import os
# import sys
# # 添加项目根目录到 Python 路径
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

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

    def forward_eval(self, x_input, p_theta=1.0, theta_random_mode="uniform"):
        """
        评估模式：返回完整信息 (theta, P_total, P_bs, P_sat, W, Psat)
        用于计算 SINR 违规率
        """
        theta_pred_raw = self.cnn_net(x_input, p_theta=p_theta, theta_random_mode=theta_random_mode)
        theta_pred = torch.atan2(torch.sin(theta_pred_raw), torch.cos(theta_pred_raw))

        W_opt, Psat_opt = self.zf_solver.solve(theta_pred, x_input)

        P_bs = torch.sum(torch.abs(W_opt)**2, dim=(1, 2))
        P_sat = Psat_opt.squeeze()
        P_total = P_bs + P_sat

        return theta_pred, P_total, P_bs, P_sat, W_opt, Psat_opt

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


class RisPowerLoss(nn.Module):
    """
    无监督功耗损失：仅最小化总功耗 P_total
    用于 Stage 2 的无标签训练
    """
    def __init__(self):
        super(RisPowerLoss, self).__init__()

    def forward(self, P_total):
        # 直接最小化平均总功耗
        loss_power = torch.mean(P_total)
        return loss_power

class SinrConstraintEvaluator:
    """
    统计 SINR 约束违规率
    需要计算每个用户的实际 SINR 并与阈值比较
    """
    def __init__(self, zf_solver, gamma_ue_db=10, gamma_su_db=10, sigma2_dbm=-80):
        self.zf_solver = zf_solver
        self.gamma_ue = 10**(gamma_ue_db / 10.0)  # 线性值
        self.gamma_su = 10**(gamma_su_db / 10.0)  # 线性值
        self.sigma2 = 1e12 * 10**(sigma2_dbm / 10.0) / 1000.0  # dBm -> Watts

    def compute_sinr(self, theta_pred, x_input, W_opt, Psat_opt):
        """
        计算实际 SINR
        返回: sinr_ue (Batch, K), sinr_su (Batch, SK)
        """
        B = theta_pred.shape[0]
        K = self.zf_solver.K
        K_SUE = self.zf_solver.K_SUE

        # 获取等效信道
        H_ue, H_su, H_sat_ue, H_sat_su = self.zf_solver.compute_effective_channels(
            theta_pred, x_input
        )

        # 提取波束方向 V (从 W_opt 提取，W_opt = V * sqrt(P))
        W_norms = torch.norm(W_opt, dim=1, keepdim=True)  # (B, 1, K)
        V = W_opt / (W_norms + 1e-12)  # (B, Nt, K)

        # 计算 UE 接收信号功率
        effective_gain = torch.bmm(H_ue, V)  # (B, K, K)
        channel_gains = torch.abs(torch.diagonal(effective_gain, dim1=1, dim2=2).unsqueeze(-1)) ** 2  # (B, K, 1)
        P_bs_vec = torch.sum(torch.abs(W_opt)**2, dim=1, keepdim=True).transpose(1, 2)  # (B, 1, K)
        S_ue = P_bs_vec * channel_gains  # (B, K, 1)

        # 计算 SU 接收信号功率
        sat_channel_gain = torch.abs(H_sat_su) ** 2  # (B, 1, 1)
        S_su = Psat_opt * sat_channel_gain  # (B, 1, 1)

        # 计算干扰
        # UE 受卫星干扰: I_sat_ue = P_sat * |H_sat_ue|^2
        I_sat_ue = Psat_opt * (torch.abs(H_sat_ue) ** 2)  # (B, K, 1)

        # SU 受基站干扰: I_bs_su
        interference_gains = torch.abs(torch.bmm(H_su, V)) ** 2  # (B, 1, K)
        I_bs_su = torch.sum(P_bs_vec * interference_gains, dim=2, keepdim=True)  # (B, 1, 1)

        # SINR 计算
        sinr_ue = S_ue / (I_sat_ue + self.sigma2 + 1e-12)  # (B, K, 1)
        sinr_su = S_su / (I_bs_su + self.sigma2 + 1e-12)   # (B, 1, 1)

        return sinr_ue.squeeze(-1), sinr_su.squeeze(-1)

    def get_violation_rate(self, theta_pred, x_input, W_opt, Psat_opt):
        """
        返回 SINR 约束违规率统计
        """
        sinr_ue, sinr_su = self.compute_sinr(theta_pred, x_input, W_opt, Psat_opt)

        # UE 违规判断
        ue_violation = (sinr_ue < self.gamma_ue).float()
        ue_violation_rate = ue_violation.mean().item()

        # SU 违规判断
        su_violation = (sinr_su < self.gamma_su).float()
        su_violation_rate = su_violation.mean().item()

        # 总违规率
        sinr_all = torch.cat([sinr_ue, sinr_su], dim=1)
        gamma_all = torch.cat([
            torch.full((sinr_ue.shape[0], sinr_ue.shape[1]), self.gamma_ue, device=sinr_ue.device),
            torch.full((sinr_su.shape[0], sinr_su.shape[1]), self.gamma_su, device=sinr_su.device)
        ], dim=1)
        total_violation = (sinr_all < gamma_all).float()
        total_violation_rate = total_violation.mean().item()

        return {
            'ue_violation_rate': ue_violation_rate,
            'su_violation_rate': su_violation_rate,
            'total_violation_rate': total_violation_rate,
            'avg_sinr_ue': sinr_ue.mean().item(),
            'avg_sinr_su': sinr_su.mean().item()
        }


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


class SemiSupervisedTrainer:
    """
    分阶段半监督训练器
    - Stage 1: 有标签退火训练 (labeled samples, stage1_epochs)
    - Stage 2: 无标签功耗最小化 (unlabeled_train samples, stage2_epochs)
    """
    def __init__(self, model, labeled_loader, unlabeled_train_loader, unlabeled_eval_loader,
                 optimizer, device, stage1_epochs=100, stage2_epochs=100, log_dir='./logs'):
        self.model = model
        self.labeled_loader = labeled_loader
        self.unlabeled_train_loader = unlabeled_train_loader
        self.unlabeled_eval_loader = unlabeled_eval_loader
        self.optimizer = optimizer
        self.device = device

        # 损失函数
        self.hybrid_criterion = RisHybridLoss(lambda_power=0.05)
        self.power_criterion = RisPowerLoss()

        # SINR 评估器
        self.sinr_evaluator = SinrConstraintEvaluator(model.zf_solver)

        # 训练参数
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs

        # TensorBoard 日志
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

    def p_theta_schedule_stage1(self, epoch):
        """
        Stage 1 退火策略：
        - 前 20% (0-19): p_theta = 0.0
        - 中间 60% (20-79): 线性增加 0.0 -> 1.0
        - 后 20% (80-99): p_theta = 1.0
        确保最后一个 epoch p_theta = 1.0
        """
        warmup_end = int(0.2 * self.stage1_epochs)
        anneal_end = int(0.8 * self.stage1_epochs)

        if epoch < warmup_end:
            return 0.0
        elif epoch >= anneal_end:
            return 1.0
        else:
            # 线性插值
            progress = (epoch - warmup_end) / (anneal_end - warmup_end)
            return progress

    def train_stage1_epoch(self, epoch):
        """
        Stage 1 训练：有标签监督退火
        """
        self.model.train()
        epoch_loss = 0
        epoch_power = 0
        epoch_phase_loss = 0

        p_theta = self.p_theta_schedule_stage1(epoch)
        global_step = epoch * len(self.labeled_loader)

        pbar = tqdm(self.labeled_loader,
                   desc=f"Stage1 Epoch {epoch+1}/{self.stage1_epochs} [p={p_theta:.2f}]",
                   leave=False)

        for batch_idx, data_list in enumerate(pbar):
            data_list = [d.to(self.device).float() for d in data_list]
            theta_teacher = data_list[0]

            self.optimizer.zero_grad()

            # 前向传播 (使用退火 p_theta)
            theta_pred, P_total, _, _ = self.model(data_list, p_theta=p_theta)

            # 计算混合损失
            loss, l_phase, l_power = self.hybrid_criterion(theta_pred, theta_teacher, P_total)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_power += torch.mean(P_total).item()
            epoch_phase_loss += l_phase.item()

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'P': f"{torch.mean(P_total).item():.2f}"})

            # TensorBoard 记录 (每个 batch)
            step = global_step + batch_idx
            self.writer.add_scalar('Stage1/Train/Loss', loss.item(), step)
            self.writer.add_scalar('Stage1/Train/PhaseLoss', l_phase.item(), step)
            self.writer.add_scalar('Stage1/Train/PowerLoss', l_power.item(), step)
            self.writer.add_scalar('Stage1/Train/AvgPower', torch.mean(P_total).item(), step)
            self.writer.add_scalar('Stage1/Train/p_theta', p_theta, step)

        # Epoch 结束时记录平均指标
        avg_loss = epoch_loss / len(self.labeled_loader)
        avg_power = epoch_power / len(self.labeled_loader)
        avg_phase_loss = epoch_phase_loss / len(self.labeled_loader)

        self.writer.add_scalar('Stage1/Epoch/AvgLoss', avg_loss, epoch)
        self.writer.add_scalar('Stage1/Epoch/AvgPhaseLoss', avg_phase_loss, epoch)
        self.writer.add_scalar('Stage1/Epoch/AvgPower', avg_power, epoch)
        self.writer.add_scalar('Stage1/Epoch/p_theta', p_theta, epoch)

        return {
            'avg_loss': avg_loss,
            'avg_power': avg_power,
            'avg_phase_loss': avg_phase_loss,
            'p_theta': p_theta
        }

    def train_stage2_epoch(self, epoch):
        """
        Stage 2 训练：无标签功耗最小化
        强制 p_theta = 1.0 (不使用 teacher)
        """
        self.model.train()
        epoch_loss = 0
        epoch_power = 0

        # Stage 2 强制 p_theta = 1.0
        p_theta = 1.0
        global_step = (self.stage1_epochs + epoch) * len(self.unlabeled_train_loader)

        pbar = tqdm(self.unlabeled_train_loader,
                   desc=f"Stage2 Epoch {epoch+1}/{self.stage2_epochs} [p=1.0]",
                   leave=False)

        for batch_idx, data_list in enumerate(pbar):
            data_list = [d.to(self.device).float() for d in data_list]
            # 注意：无标签数据中 theta_opt 不参与训练 (因为 p_theta=1.0)

            self.optimizer.zero_grad()

            # 前向传播 (强制 p_theta=1.0)
            theta_pred, P_total, _, _ = self.model(data_list, p_theta=p_theta)

            # 仅使用功耗损失
            loss = self.power_criterion(P_total)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_power += torch.mean(P_total).item()

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'P': f"{torch.mean(P_total).item():.2f}"})

            # TensorBoard 记录 (每个 batch)
            step = global_step + batch_idx
            self.writer.add_scalar('Stage2/Train/Loss', loss.item(), step)
            self.writer.add_scalar('Stage2/Train/AvgPower', torch.mean(P_total).item(), step)

        # Epoch 结束时记录平均指标
        avg_loss = epoch_loss / len(self.unlabeled_train_loader)
        avg_power = epoch_power / len(self.unlabeled_train_loader)

        self.writer.add_scalar('Stage2/Epoch/AvgLoss', avg_loss, epoch)
        self.writer.add_scalar('Stage2/Epoch/AvgPower', avg_power, epoch)

        return {
            'avg_loss': avg_loss,
            'avg_power': avg_power,
            'p_theta': p_theta
        }

    def evaluate(self, eval_loader, stage_name, epoch=0, stage=0):
        """
        评估函数
        - 使用 unlabeled_eval 数据
        - model.eval() 且 p_theta=1.0
        - 返回：平均 P_total, SINR 违规率
        """
        self.model.eval()

        total_power = 0
        total_batches = 0

        # SINR 统计
        sinr_stats = {
            'ue_violation_rate': 0,
            'su_violation_rate': 0,
            'total_violation_rate': 0,
            'avg_sinr_ue': 0,
            'avg_sinr_su': 0
        }

        with torch.no_grad():
            for data_list in tqdm(eval_loader, desc=f"Evaluating {stage_name}", leave=False):
                data_list = [d.to(self.device).float() for d in data_list]

                # eval 时 p_theta = 1.0
                theta_pred, P_total, P_bs, P_sat, _, _ = self.model.forward_eval(data_list, p_theta=1.0)

                # 统计功耗
                total_power += torch.mean(P_total).item()
                total_batches += 1

                # 统计 SINR 违规率
                W_opt, Psat_opt = self.model.zf_solver.solve(theta_pred, data_list)
                violations = self.sinr_evaluator.get_violation_rate(
                    theta_pred, data_list, W_opt, Psat_opt
                )

                for key in sinr_stats:
                    sinr_stats[key] += violations[key]

        # 取平均
        results = {
            'avg_power': total_power / total_batches,
            **{k: v / total_batches for k, v in sinr_stats.items()}
        }

        # TensorBoard 记录评估结果
        prefix = f"Stage{stage}/Eval" if stage > 0 else "Eval"
        self.writer.add_scalar(f'{prefix}/AvgPower', results['avg_power'], epoch)
        self.writer.add_scalar(f'{prefix}/SINR_Violation_Total', results['total_violation_rate'], epoch)
        self.writer.add_scalar(f'{prefix}/SINR_Violation_UE', results['ue_violation_rate'], epoch)
        self.writer.add_scalar(f'{prefix}/SINR_Violation_SU', results['su_violation_rate'], epoch)
        self.writer.add_scalar(f'{prefix}/AvgSINR_UE', results['avg_sinr_ue'], epoch)
        self.writer.add_scalar(f'{prefix}/AvgSINR_SU', results['avg_sinr_su'], epoch)

        return results

    def run_training(self):
        """
        执行完整的两阶段训练流程
        """
        print("=" * 60)
        print("Starting Semi-Supervised Training")
        print("=" * 60)
        print(f"Stage 1: Labeled training ({len(self.labeled_loader.dataset)} samples, "
              f"{self.stage1_epochs} epochs)")
        print(f"Stage 2: Unlabeled training ({len(self.unlabeled_train_loader.dataset)} samples, "
              f"{self.stage2_epochs} epochs)")
        print(f"Evaluation: Unlabeled eval ({len(self.unlabeled_eval_loader.dataset)} samples)")
        print("=" * 60)

        # 在 Stage 1 开始前评估一次
        print("\nInitial evaluation:")
        eval_results = self.evaluate(self.unlabeled_eval_loader, "Initial", epoch=0, stage=0)
        print(f"  Avg Power: {eval_results['avg_power']:.4f}")
        print(f"  SINR Violation Rate: {eval_results['total_violation_rate']:.4f}")
        print(f"  Avg SINR UE: {eval_results['avg_sinr_ue']:.4f}")
        print(f"  Avg SINR SU: {eval_results['avg_sinr_su']:.4f}")

        # ==================== Stage 1 ====================
        print("\n" + "=" * 60)
        print("Stage 1: Supervised Annealing Training")
        print("=" * 60)

        stage1_history = []
        for epoch in range(self.stage1_epochs):
            metrics = self.train_stage1_epoch(epoch)
            stage1_history.append(metrics)

            # 每 10 个 epoch 评估一次
            if (epoch + 1) % 10 == 0 or epoch == self.stage1_epochs - 1:
                eval_results = self.evaluate(self.unlabeled_eval_loader, f"Stage1-Epoch{epoch+1}", epoch=epoch+1, stage=1)
                print(f"\nStage1 Epoch {epoch+1}: Train Loss={metrics['avg_loss']:.4f}, "
                      f"Train P={metrics['avg_power']:.4f}, p={metrics['p_theta']:.2f}")
                print(f"  Eval: P={eval_results['avg_power']:.4f}, "
                      f"Violation={eval_results['total_violation_rate']:.4f}")

        # Stage 1 结束时确保 p_theta=1.0
        print(f"\nStage 1 completed. Final p_theta = 1.0")

        # ==================== Stage 2 ====================
        print("\n" + "=" * 60)
        print("Stage 2: Unsupervised Power Minimization")
        print("=" * 60)

        stage2_history = []
        for epoch in range(self.stage2_epochs):
            metrics = self.train_stage2_epoch(epoch)
            stage2_history.append(metrics)

            # 每 10 个 epoch 评估一次
            if (epoch + 1) % 10 == 0 or epoch == self.stage2_epochs - 1:
                eval_results = self.evaluate(self.unlabeled_eval_loader, f"Stage2-Epoch{epoch+1}", epoch=self.stage1_epochs+epoch+1, stage=2)
                print(f"\nStage2 Epoch {epoch+1}: Train Loss={metrics['avg_loss']:.4f}, "
                      f"Train P={metrics['avg_power']:.4f}, p=1.0")
                print(f"  Eval: P={eval_results['avg_power']:.4f}, "
                      f"Violation={eval_results['total_violation_rate']:.4f}")

        # ==================== Final Evaluation ====================
        print("\n" + "=" * 60)
        print("Training Completed. Final Evaluation:")
        print("=" * 60)
        final_results = self.evaluate(self.unlabeled_eval_loader, "Final", epoch=self.stage1_epochs+self.stage2_epochs, stage=2)
        print(f"  Avg Power: {final_results['avg_power']:.4f}")
        print(f"  UE SINR Violation Rate: {final_results['ue_violation_rate']:.4f}")
        print(f"  SU SINR Violation Rate: {final_results['su_violation_rate']:.4f}")
        print(f"  Total SINR Violation Rate: {final_results['total_violation_rate']:.4f}")
        print(f"  Avg SINR UE: {final_results['avg_sinr_ue']:.4f}")
        print(f"  Avg SINR SU: {final_results['avg_sinr_su']:.4f}")

        # 关闭 TensorBoard writer
        self.writer.close()
        print(f"\nTensorBoard logs saved to: {self.log_dir}")
        print(f"View logs with: tensorboard --logdir={self.log_dir}")

        return {
            'stage1_history': stage1_history,
            'stage2_history': stage2_history,
            'final_results': final_results
        }