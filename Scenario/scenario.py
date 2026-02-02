import numpy as np
import matplotlib.pyplot as plt

class ITSNScenario:
    def __init__(self, rng_seed=None):
        # --- 基础参数设置 (对应 Prepare_parameter.m) ---
        if(rng_seed==None):
            np.random.seed(np.random.randint(1,999))
        else:
            np.random.seed(rng_seed)
        
        # 系统规模
        self.K = 4       # BS users
        self.SK = 1      # SAT users
        self.N_t = 36    # BS transmit antennas (URA 8x8)
        self.N_r = 1     # User receive antennas
        self.N_sat = 64  # Satellite antennas (URA 8x8)
        self.N_ris = 100 # RIS elements (URA 10x10)

        # RIS subsurface configuration
        # MATLAB代码中 Phi = 9*diag(ones(N, 1))，增益系数为9
        self.ris_amplitude_gain = 9  # 幅度增益（对应功率增益81）
        
        # 物理常数
        self.f = 2e9
        self.c = 3e8
        self.wavelength = self.c / self.f
        self.Bw = 10e6
        self.k_boltz = 1.38064852e-23
        self.T0 = 290
        self.F_dB = 3
        self.P_noise = self.k_boltz * self.T0 * self.Bw * (10**(self.F_dB/10))

        # 路径损耗与衰落参数
        self.P0_dB = -30
        self.BETA_r = 10**(20/10) # Rician factor for RIS links
        self.BETA_d = 10**(10/10)  # Rician factor for Direct links (提高到10dB)
        self.BETA_SAT = 10**(6/10)
        self.alpha_terrestrial = 3.75 # 地面链路路损指数
        self.alpha_ris = 2.0          # RIS相关链路路损指数

        # 功率参数 (根据MATLAB代码 Optimize.m)
        # P = 0.001 是功率缩放因子
        # P_sat = (db2pow(21) * P_noise / PL) / (N_sat^2)
        # 实际BS功率 = P * ||W||_F^2
        # 实际卫星功率 = P_sat * N_sat
        self.P_bs_scale = 0.001  # 基站功率缩放因子 (与MATLAB一致)

        # 卫星功率将在generate_channels中根据路径损耗动态计算

        # 最大功率预算 (用于约束)
        self.P_bs_max = 10.0  # 基站最大功率 10W
        self.P_sat_max = 20.0  # 卫星最大功率 20W (降低以减少RIS相位不良时的跳变)
        
        # 位置坐标 (单位: m) - 与MATLAB Prepare_parameter.m一致
        self.BS_POS = np.array([0, 0, 10])
        self.RIS_POS = np.array([25, 25, 10])  # 与MATLAB一致
        self.SAT_POS = np.zeros(3) # 动态更新

        # 用户分布参数 - 与MATLAB一致 (50x50m区域)
        self.USER_AREA_SIZE = 50  # 用户分布区域 50x50m (与MATLAB一致)
        self.MIN_USER_DISTANCE = 10  # 用户间最小距离

        # 用户位置容器
        self.USERS_POS = np.zeros((self.K, 3))
        self.SATUSERS_POS = np.zeros((self.SK, 3))

        # 初始化位置
        self.reset_user_positions()

        # 初始化NLOS缓存（慢衰落，episode内不变）
        self._nlos_cache = {}
        self._generate_nlos_cache()

        # 初始化卫星 (默认位置)
        self.update_satellite_position(ele=40, azi=80)

    def db2pow(self, db):
        return 10**(db/10.0)

    def reset_user_positions(self):
        """
        随机生成地面用户和卫星用户位置
        保证用户间有足够的空间间隔以维持信道差异性
        """
        # BS Users: 分布在150x150m区域，高度0~5m随机
        max_attempts = 100
        for attempt in range(max_attempts):
            xy = self.USER_AREA_SIZE * np.random.rand(self.K, 2)
            z = 5 * np.random.rand(self.K)  # 高度0~5m
            self.USERS_POS = np.column_stack((xy, z))

            # 检查用户间距离是否满足最小间隔要求
            valid = True
            for i in range(self.K):
                for j in range(i+1, self.K):
                    dist = np.linalg.norm(self.USERS_POS[i] - self.USERS_POS[j])
                    if dist < self.MIN_USER_DISTANCE:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                break

            if attempt == max_attempts - 1:
                print("[Warning] Could not satisfy minimum user distance constraint after 100 attempts")

        # Sat Users: 在RIS附近的20x20区域，保持靠近RIS
        offset = self.RIS_POS[:2] - np.array([10, 10])  # RIS为中心的20x20区域
        self.SATUSERS_POS = np.column_stack((
            offset + 20 * np.random.rand(self.SK, 2),
            np.zeros(self.SK)
        ))

    def _generate_nlos_cache(self):
        """
        生成并缓存NLOS分量（模拟慢衰落，episode内保持不变）
        """
        self._nlos_cache = {
            # BS -> UE (K个用户，每个N_t维)
            'H_BS2UE': [(np.random.randn(self.N_t) + 1j * np.random.randn(self.N_t)) / np.sqrt(2)
                        for _ in range(self.K)],
            # BS -> SUE (SK个用户，每个N_t维)
            'H_BS2SUE': [(np.random.randn(self.N_t) + 1j * np.random.randn(self.N_t)) / np.sqrt(2)
                         for _ in range(self.SK)],
            # RIS -> UE (K个用户，每个N_ris维)
            'H_RIS2UE': [(np.random.randn(self.N_ris) + 1j * np.random.randn(self.N_ris)) / np.sqrt(2)
                         for _ in range(self.K)],
            # RIS -> SUE (SK个用户，每个N_ris维)
            'H_RIS2SUE': [(np.random.randn(self.N_ris) + 1j * np.random.randn(self.N_ris)) / np.sqrt(2)
                          for _ in range(self.SK)],
        }

    def update_nlos_cache(self, alpha):
        """
        使用一阶高斯-马尔可夫模型更新NLOS缓存
        g_{t+1} = alpha * g_t + sqrt(1 - alpha^2) * w_t, w_t ~ CN(0,1)

        Args:
            alpha: 相关系数，alpha = J_0(2*pi*f_D*delta_t)
        """
        if not self._nlos_cache:
            return

        sqrt_term = np.sqrt(1 - alpha**2)

        for key in self._nlos_cache:
            for i in range(len(self._nlos_cache[key])):
                dim = self._nlos_cache[key][i].shape[0]
                w = (np.random.randn(dim) + 1j * np.random.randn(dim)) / np.sqrt(2)
                self._nlos_cache[key][i] = alpha * self._nlos_cache[key][i] + sqrt_term * w

    def update_satellite_position(self, ele, azi, orbit_height=500e3):
        """
        更新卫星位置 (用于构建数据集循环)
        ele, azi: 单位是度 (degree)
        坐标系转换遵循 MATLAB 代码逻辑:
        x = R * cos(ele) * sin(azi)
        y = R * cos(ele) * cos(azi)
        z = R * sin(ele)
        """
        rad_ele = np.deg2rad(ele)
        rad_azi = np.deg2rad(azi)
        
        self.SAT_POS = orbit_height * np.array([
            np.cos(rad_ele) * np.sin(rad_azi),
            np.cos(rad_ele) * np.cos(rad_azi),
            np.sin(rad_ele)
        ])

    def get_distance(self, pos1, pos2):
        """计算两组坐标点之间的欧氏距离"""
        # 如果是单点对多点，利用广播机制
        if pos1.ndim == 1: pos1 = pos1[np.newaxis, :]
        if pos2.ndim == 1: pos2 = pos2[np.newaxis, :]
        # pos1: (N, 3), pos2: (1, 3) or (M, 3) -> return (N, 1) or matching dim
        return np.sqrt(np.sum((pos1 - pos2)**2, axis=1))

    def _get_ula_response(self, num_antennas, theta_deg):
        """生成均匀线阵(ULA)导引向量"""
        # theta_deg: Angle of Departure/Arrival
        idx = np.arange(num_antennas)
        # exp(1j * (n-1) * pi * cos(theta))
        # 注意: MATLAB代码中是从1开始, Python从0开始, 物理意义一致
        resp = np.exp(1j * idx * np.pi * np.cos(np.deg2rad(theta_deg)))
        return resp

    def _get_ura_response(self, total_elements, ele_deg, azi_deg):
        """
        生成均匀面阵(URA)导引向量
        遵循 MATLAB 代码中的逻辑：
        Array factor = exp(1j * pi * ((i-1)*cos(ele)*cos(azi) + (j-1)*cos(ele)*sin(azi)))
        """
        sqrt_N = int(np.sqrt(total_elements))
        # 网格生成 (i, j) 对应 MATLAB循环
        # i对应x轴方向索引, j对应y轴方向索引 (参考MATLAB循环结构)
        i_idx = np.arange(sqrt_N)
        j_idx = np.arange(sqrt_N)
        
        # 利用广播生成矩阵
        # MATLAB logic: outer loop i, inner loop j.
        # A(j, i) = ...
        
        term1 = i_idx * np.cos(np.deg2rad(ele_deg)) * np.cos(np.deg2rad(azi_deg))
        term2 = j_idx * np.cos(np.deg2rad(ele_deg)) * np.sin(np.deg2rad(azi_deg))
        
        # Create grid. Note: In MATLAB code A_aod(j, i), j is inner loop.
        # term1 is controlled by i, term2 is controlled by j.
        T1, T2 = np.meshgrid(term1, term2) # T1 varies with columns(i), T2 varies with rows(j)
        
        A_matrix = np.exp(1j * np.pi * (T1 + T2))
        
        # MATLAB: reshape(A.', 1, N). 
        # A.' is transpose. reshape(..., 1, N) flattens it.
        # Python: A_matrix.T.flatten() mimics reshape(A.', ...)
        return A_matrix.T.flatten()

    def generate_channels(self):
        """
        对应 Build_channel.m 的核心逻辑
        返回一个包含所有信道矩阵的字典
        """
        channels = {}
        
        # 1. 计算距离
        dist_BS2UE = self.get_distance(self.USERS_POS, self.BS_POS)
        dist_BS2SUE = self.get_distance(self.SATUSERS_POS, self.BS_POS)
        dist_BS2RIS = self.get_distance(self.BS_POS, self.RIS_POS)[0]
        dist_UE2SAT = self.get_distance(self.USERS_POS, self.SAT_POS) # BSUE -> SAT
        dist_SUE2SAT = self.get_distance(self.SATUSERS_POS, self.SAT_POS) # SUE -> SAT
        dist_SAT2RIS = self.get_distance(self.SAT_POS, self.RIS_POS)[0]
        dist_UE2RIS = self.get_distance(self.USERS_POS, self.RIS_POS)
        dist_SUE2RIS = self.get_distance(self.SATUSERS_POS, self.RIS_POS)

        # ----------------------------------------------------------------
        # 1. BS -> UE (K x Nt) - BS使用URA
        # 使用共轭形式，确保 H @ W 给出正确的波束成形增益
        # ----------------------------------------------------------------
        H_BS2UE = np.zeros((self.K, self.N_t), dtype=complex)

        for k in range(self.K):
            # 计算BS到UE的方向向量
            vec_bs2ue = self.USERS_POS[k] - self.BS_POS
            dx, dy, dz = vec_bs2ue[0], vec_bs2ue[1], vec_bs2ue[2]

            # 计算方位角和俯仰角
            azi_bs = np.degrees(np.arctan2(dz, dy))
            ele_bs = np.abs(np.degrees(np.arctan2(dx, np.sqrt(dy**2 + dz**2))))

            # 使用URA导引向量
            pl_linear = np.sqrt(self.db2pow(self.P0_dB - 10 * self.alpha_terrestrial * np.log10(dist_BS2UE[k])))
            h_los = self._get_ura_response(self.N_t, ele_bs, azi_bs)
            # Rician信道: LOS + NLOS（无缓存时即时生成）
            if 'H_BS2UE' in self._nlos_cache:
                h_nlos = self._nlos_cache['H_BS2UE'][k]
            else:
                h_nlos = (np.random.randn(self.N_t) + 1j * np.random.randn(self.N_t)) / np.sqrt(2)
            h_combined = np.sqrt(self.BETA_d / (1 + self.BETA_d)) * h_los + np.sqrt(1 / (1 + self.BETA_d)) * h_nlos
            # 使用共轭形式
            H_BS2UE[k, :] = pl_linear * h_combined.conj()

        channels['H_BS2UE'] = H_BS2UE

        # ----------------------------------------------------------------
        # 2. BS -> SUE (SK x Nt) - BS使用URA
        # 使用共轭形式
        # ----------------------------------------------------------------
        H_BS2SUE = np.zeros((self.SK, self.N_t), dtype=complex)

        for k in range(self.SK):
            # 计算BS到SUE的方向向量
            vec_bs2sue = self.SATUSERS_POS[k] - self.BS_POS
            dx, dy, dz = vec_bs2sue[0], vec_bs2sue[1], vec_bs2sue[2]

            # 计算方位角和俯仰角
            azi_bs = np.degrees(np.arctan2(dz, dy))
            ele_bs = np.abs(np.degrees(np.arctan2(dx, np.sqrt(dy**2 + dz**2))))

            # 使用URA导引向量
            pl_linear = np.sqrt(self.db2pow(self.P0_dB - 10 * self.alpha_terrestrial * np.log10(dist_BS2SUE[k])))
            h_los = self._get_ura_response(self.N_t, ele_bs, azi_bs)
            # Rician信道: LOS + NLOS（无缓存时即时生成）
            if 'H_BS2SUE' in self._nlos_cache:
                h_nlos = self._nlos_cache['H_BS2SUE'][k]
            else:
                h_nlos = (np.random.randn(self.N_t) + 1j * np.random.randn(self.N_t)) / np.sqrt(2)
            h_combined = np.sqrt(self.BETA_d / (1 + self.BETA_d)) * h_los + np.sqrt(1 / (1 + self.BETA_d)) * h_nlos
            # 使用共轭形式
            H_BS2SUE[k, :] = pl_linear * h_combined.conj()

        channels['H_BS2SUE'] = H_BS2SUE

        # ----------------------------------------------------------------
        # 3. BS -> RIS (N_ris x N_t) - BS使用URA, LoS Dominant
        # 外积形式: G = a_ris_aoa @ a_bs_aod^H
        # ----------------------------------------------------------------
        # AoD at BS (向RIS方向发射)
        vec_bs2ris = self.RIS_POS - self.BS_POS
        dx_bs, dy_bs, dz_bs = vec_bs2ris[0], vec_bs2ris[1], vec_bs2ris[2]
        azi_bs_aod = np.degrees(np.arctan2(dz_bs, dy_bs))
        ele_bs_aod = np.abs(np.degrees(np.arctan2(dx_bs, np.sqrt(dy_bs**2 + dz_bs**2))))

        a_bs_aod = self._get_ura_response(self.N_t, ele_bs_aod, azi_bs_aod)

        # AoA at RIS
        dx, dy, dz = vec_bs2ris[0], vec_bs2ris[1], vec_bs2ris[2]
        angle_aoa_azi = np.degrees(np.arctan2(dz, dy))
        angle_aoa_ele = np.abs(np.degrees(np.arctan2(dx, np.sqrt(dy**2 + dz**2))))

        a_ris_aoa = self._get_ura_response(self.N_ris, angle_aoa_ele, angle_aoa_azi)

        # G = a_ris_aoa @ a_bs_aod^H (外积，使用共轭转置)
        G_BS_raw = np.outer(a_ris_aoa, a_bs_aod.conj())

        pl_ris = np.sqrt(self.db2pow(self.P0_dB - 10 * self.alpha_ris * np.log10(dist_BS2RIS)))
        channels['G_BS'] = pl_ris * G_BS_raw

        # ----------------------------------------------------------------
        # 4. SAT -> BSUE (K x N_sat)
        # 注意：信道定义使用共轭形式，使得 H @ W 直接给出正确的波束成形增益
        # 接收信号 y = h^H @ w，定义 H = h^H 使得 y = H @ w
        # ----------------------------------------------------------------
        H_SAT2UE = np.zeros((self.K, self.N_sat), dtype=complex)
        for k in range(self.K):
            vec_sat2ue = self.USERS_POS[k] - self.SAT_POS
            dx, dy, dz = vec_sat2ue[0], vec_sat2ue[1], vec_sat2ue[2]

            # Angle Calculation per MATLAB
            azi = np.degrees(np.arctan2(dz, dy))
            ele = np.abs(np.degrees(np.arctan2(dx, np.sqrt(dy**2 + dz**2))))

            a_sat_aod = self._get_ura_response(self.N_sat, ele, azi)
            pl_lin = np.sqrt(self.db2pow(self.P0_dB - 10 * self.alpha_ris * np.log10(dist_UE2SAT[k])))
            # 使用共轭形式，确保 H @ W 给出正确的波束成形增益
            H_SAT2UE[k, :] = pl_lin * a_sat_aod.conj()

        channels['H_SAT2UE'] = H_SAT2UE

        # ----------------------------------------------------------------
        # 5. SAT -> SUE (SK x N_sat) - 纯LoS信道（卫星通信为视距传播）
        # 使用共轭形式，确保 H @ W 给出正确的波束成形增益
        # ----------------------------------------------------------------
        H_SAT2SUE = np.zeros((self.SK, self.N_sat), dtype=complex)
        for k in range(self.SK):
            vec_sat2sue = self.SATUSERS_POS[k] - self.SAT_POS
            dx, dy, dz = vec_sat2sue[0], vec_sat2sue[1], vec_sat2sue[2]

            azi = np.degrees(np.arctan2(dz, dy))
            ele = np.abs(np.degrees(np.arctan2(dx, np.sqrt(dy**2 + dz**2))))

            # 纯LoS信道（卫星到卫星用户）
            a_sat_aod = self._get_ura_response(self.N_sat, ele, azi)

            pl_lin = np.sqrt(self.db2pow(self.P0_dB - 10 * self.alpha_ris * np.log10(dist_SUE2SAT[k])))
            # 使用共轭形式
            H_SAT2SUE[k, :] = pl_lin * a_sat_aod.conj()

        channels['H_SAT2SUE'] = H_SAT2SUE

        # ----------------------------------------------------------------
        # 6. SAT -> RIS (N_ris x N_sat)
        # 外积形式: G = a_ris_aoa @ a_sat_aod^H
        # 这样 G @ W_sat 给出正确的波束成形增益
        # ----------------------------------------------------------------
        vec_sat2ris = self.RIS_POS - self.SAT_POS

        # AoD at SAT
        dx, dy, dz = vec_sat2ris[0], vec_sat2ris[1], vec_sat2ris[2]
        azi_sat = np.degrees(np.arctan2(dz, dy))
        ele_sat = np.abs(np.degrees(np.arctan2(dx, np.sqrt(dy**2 + dz**2))))
        a_sat_aod = self._get_ura_response(self.N_sat, ele_sat, azi_sat)

        # AoA at RIS
        vec_ris2sat = self.SAT_POS - self.RIS_POS
        dx_r, dy_r, dz_r = vec_ris2sat[0], vec_ris2sat[1], vec_ris2sat[2]
        azi_ris = np.degrees(np.arctan2(dz_r, dy_r))
        ele_ris = np.abs(np.degrees(np.arctan2(dx_r, np.sqrt(dy_r**2 + dz_r**2))))
        a_ris_aoa = self._get_ura_response(self.N_ris, ele_ris, azi_ris)

        # G = a_ris_aoa @ a_sat_aod^H (外积，使用共轭转置)
        G_SAT_raw = np.outer(a_ris_aoa, a_sat_aod.conj())
        pl_lin = np.sqrt(self.db2pow(self.P0_dB - 10 * self.alpha_ris * np.log10(dist_SAT2RIS)))
        channels['G_SAT'] = pl_lin * G_SAT_raw

        # ----------------------------------------------------------------
        # 7. RIS -> UE (K x N_ris)
        # 使用共轭形式，确保级联信道计算正确
        # ----------------------------------------------------------------
        H_RIS2UE = np.zeros((self.K, self.N_ris), dtype=complex)
        for k in range(self.K):
            vec_ris2ue = self.USERS_POS[k] - self.RIS_POS
            dx, dy, dz = vec_ris2ue[0], vec_ris2ue[1], vec_ris2ue[2]

            azi = np.degrees(np.arctan2(dz, dy))
            ele = np.abs(np.degrees(np.arctan2(dx, np.sqrt(dy**2 + dz**2))))

            a_ris_aod = self._get_ura_response(self.N_ris, ele, azi)
            # Rician信道: LOS + NLOS（无缓存时即时生成）
            if 'H_RIS2UE' in self._nlos_cache:
                h_nlos = self._nlos_cache['H_RIS2UE'][k]
            else:
                h_nlos = (np.random.randn(self.N_ris) + 1j * np.random.randn(self.N_ris)) / np.sqrt(2)
            h_comb = np.sqrt(self.BETA_r / (1 + self.BETA_r)) * a_ris_aod + np.sqrt(1 / (1 + self.BETA_r)) * h_nlos

            pl_lin = np.sqrt(self.db2pow(self.P0_dB - 10 * self.alpha_ris * np.log10(dist_UE2RIS[k])))
            # 使用共轭形式
            H_RIS2UE[k, :] = pl_lin * h_comb.conj()

        channels['H_RIS2UE'] = H_RIS2UE

        # ----------------------------------------------------------------
        # 8. RIS -> SUE (SK x N_ris)
        # 使用共轭形式
        # ----------------------------------------------------------------
        H_RIS2SUE = np.zeros((self.SK, self.N_ris), dtype=complex)
        for k in range(self.SK):
            vec_ris2sue = self.SATUSERS_POS[k] - self.RIS_POS
            dx, dy, dz = vec_ris2sue[0], vec_ris2sue[1], vec_ris2sue[2]

            azi = np.degrees(np.arctan2(dz, dy))
            ele = np.abs(np.degrees(np.arctan2(dx, np.sqrt(dy**2 + dz**2))))

            a_ris_aod = self._get_ura_response(self.N_ris, ele, azi)
            # Rician信道: LOS + NLOS（无缓存时即时生成）
            if 'H_RIS2SUE' in self._nlos_cache:
                h_nlos = self._nlos_cache['H_RIS2SUE'][k]
            else:
                h_nlos = (np.random.randn(self.N_ris) + 1j * np.random.randn(self.N_ris)) / np.sqrt(2)
            h_comb = np.sqrt(self.BETA_r / (1 + self.BETA_r)) * a_ris_aod + np.sqrt(1 / (1 + self.BETA_r)) * h_nlos

            pl_lin = np.sqrt(self.db2pow(self.P0_dB - 10 * self.alpha_ris * np.log10(dist_SUE2RIS[k])))
            # 使用共轭形式
            H_RIS2SUE[k, :] = pl_lin * h_comb.conj()

        channels['H_RIS2SUE'] = H_RIS2SUE

        # ----------------------------------------------------------------
        # 9. W_sat (Sat Precoding/Beamforming Vector) - Pointing to Satellite User
        # ----------------------------------------------------------------
        # 卫星波束成形应指向卫星用户，而不是原点
        # 使用第一个卫星用户的位置作为波束指向
        # 注意：W_sat 归一化到单位 Frobenius 范数，实际发射功率由 P_sat 控制
        W_sat_raw = H_SAT2SUE.conj().T
        # 归一化到单位 Frobenius 范数 (与 evaluate_baseline.py 一致)
        W_sat = W_sat_raw / (np.linalg.norm(W_sat_raw, 'fro'))
        channels['W_sat'] = W_sat  # Column vector, ||W_sat||_F = 1

        # ----------------------------------------------------------------
        # 10. 初始化卫星功率以保证15dB SNR (与baseline一致)
        # ---------------------------------------------------------------

        # 初始化卫星功率：根据直接路径信道增益计算，确保15dB SNR
        # SNR = P_sat * |h_s_j @ W_sat|^2 / sigma2 >= 15dB
        SNR_target_dB = 17.0  # 目标SNR (用户要求15dB)
        SNR_target_linear = self.db2pow(SNR_target_dB)

        # 计算直接路径信道增益 (不考虑RIS，因为初始时Phi未知)
        # W_sat = H_SAT2SUE.conj().T，所以用 h @ w
        channel_gain = np.linalg.norm(H_SAT2SUE @ W_sat) ** 2

        # 计算所需功率：P_sat = SNR_target * sigma2 / channel_gain
        P_sat_required = SNR_target_linear * self.P_noise / (channel_gain)

        # 限制在合理范围内
        self.P_sat = np.clip(P_sat_required, 0.01, self.P_sat_max)

        return channels

    def compute_sat_power(self, Phi, channels, W_bs=None, sinr_threshold_db=10.0):
        """
        根据RIS相位和卫星用户SINR约束计算最小卫星功率

        卫星用户SINR公式 (考虑RIS):
        SINR_sat = P_sat * |h_sat_eff|^2 / (I_bs + noise)

        其中:
        - h_sat_eff = H_SAT2SUE @ W_sat + H_RIS2SUE @ Phi @ G_SAT @ W_sat (卫星到卫星用户的等效信道)
        - I_bs = BS对卫星用户的干扰 (如果有BS波束成形)

        Parameters:
        -----------
        Phi : np.ndarray, shape (N_ris, N_ris)
            RIS反射矩阵 (对角矩阵)
        channels : dict
            信道字典
        W_bs : np.ndarray, shape (N_t, K), optional
            BS波束成形矩阵 (用于计算BS对卫星用户的干扰)
        sinr_threshold_db : float
            卫星用户SINR阈值 (dB)

        Returns:
        --------
        P_sat : float
            满足卫星用户SINR约束的最小卫星功率
        sat_user_info : dict
            卫星用户信息 (信道增益、干扰等)
        """
        sinr_threshold_linear = self.db2pow(sinr_threshold_db)

        # 获取信道
        H_SAT2SUE = channels['H_SAT2SUE']  # (SK, N_sat)
        H_RIS2SUE = channels['H_RIS2SUE']  # (SK, N_ris)
        G_SAT = channels['G_SAT']          # (N_ris, N_sat)
        W_sat = channels['W_sat']          # (N_sat, 1)
        H_BS2SUE = channels['H_BS2SUE']    # (SK, N_t)

        # 计算卫星到卫星用户的等效信道 (包含RIS反射路径)
        # H_sat_eff = H_SAT2SUE + H_RIS2SUE @ Phi @ G_SAT  (SK, N_sat)
        H_sat_eff = H_SAT2SUE + H_RIS2SUE @ Phi @ G_SAT  # (SK, N_sat)

        # 计算每个卫星用户的信号功率: |h_k^H @ w_sat|^2
        # W_sat = h.T (因为 H_SAT2SUE 存储的是 h^*)，所以用 h^* @ W_sat
        sat_channel_gain = np.zeros(self.SK)
        for k in range(self.SK):
            # signal = |h^* @ w_sat|^2 = |h^* @ h.T|^2 = ||h||^4
            sat_channel_gain[k] = np.abs(H_sat_eff[k, :] @ W_sat.flatten()) ** 2

        # 用于调试输出 (保留旧的计算方式用于对比)
        h_direct = H_SAT2SUE @ W_sat  # (SK, 1)
        h_ris = H_RIS2SUE @ Phi @ G_SAT @ W_sat  # (SK, 1)

        # 计算BS对卫星用户的干扰 (需要考虑RIS反射路径!)
        # H_eff_sue = H_BS2SUE + H_RIS2SUE @ Phi @ G_BS
        G_BS = channels['G_BS']  # (N_ris, N_t)
        H_eff_sue = H_BS2SUE + H_RIS2SUE @ Phi @ G_BS  # (SK, N_t)

        bs_interference = np.zeros(self.SK)
        if W_bs is not None:
            # I_bs = sum_k |h_eff_sue^H @ w_k|^2 * P_bs_scale
            # W_bs 已经取共轭，所以用 h @ w
            for k in range(self.SK):
                # 对每个BS用户的波束成形向量计算干扰
                interf_sum = 0.0
                for j in range(W_bs.shape[1]):  # 遍历所有BS用户
                    interf_sum += np.abs(H_eff_sue[k, :] @ W_bs[:, j]) ** 2
                bs_interference[k] = self.P_bs_scale * interf_sum

        # 计算满足SINR约束的最小卫星功率
        # SINR_sat = P_sat * |h_sat_eff|^2 / (I_bs + noise) >= threshold
        # P_sat >= threshold * (I_bs + noise) / |h_sat_eff|^2

        # 对所有卫星用户取最大值 (保证所有用户都满足)
        P_sat_required = np.zeros(self.SK)
        for k in range(self.SK):
            total_interference = bs_interference[k] + self.P_noise
            P_sat_required[k] = sinr_threshold_linear * total_interference / (sat_channel_gain[k] + 1e-20)

        # 取最大值确保所有卫星用户满足SINR
        P_sat = np.max(P_sat_required)

        # 限制在合理范围内
        P_sat = np.clip(P_sat, 1e-6, self.P_sat_max)

        # 更新实例变量
        self.P_sat = P_sat

        sat_user_info = {
            'sat_channel_gain': sat_channel_gain,
            'bs_interference': bs_interference,
            'P_sat_required': P_sat_required,
            'h_direct_gain': np.abs(h_direct.flatten()) ** 2,
            'h_ris_gain': np.abs(h_ris.flatten()) ** 2,
            'sinr_sat': P_sat * sat_channel_gain / (bs_interference + self.P_noise)
        }

        return P_sat, sat_user_info

    # def plot_scenario(self):
    #     """可视化场景 (对应 Prepare_parameter.m 的绘图部分)"""
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # BS
    #     ax.scatter(self.BS_POS[0], self.BS_POS[1], self.BS_POS[2], c='r', marker='o', s=100, label='BS')
        
    #     # RIS
    #     ax.scatter(self.RIS_POS[0], self.RIS_POS[1], self.RIS_POS[2], c='b', marker='s', s=100, label='RIS')
        
    #     # USERS
    #     ax.scatter(self.USERS_POS[:,0], self.USERS_POS[:,1], self.USERS_POS[:,2], c='r', marker='x', s=50, label='UE_BS')
        
    #     # SAT USERS
    #     ax.scatter(self.SATUSERS_POS[:,0], self.SATUSERS_POS[:,1], self.SATUSERS_POS[:,2], c='g', marker='x', s=50, label='UE_SAT')
        
    #     # SATELLITE (Scale down distance for visualization if needed, or just plot direction)
    #     # Note: 500km is too far to plot on same scale as 50m. 
    #     # Plotting a proxy point for direction
    #     sat_dir = self.SAT_POS / np.linalg.norm(self.SAT_POS) * 100 # Scale to 100m
    #     ax.scatter(sat_dir[0], sat_dir[1], sat_dir[2], c='k', marker='^', s=100, label='SAT (Direction)')
        
    #     ax.set_xlabel('X (m)')
    #     ax.set_ylabel('Y (m)')
    #     ax.set_zlabel('Z (m)')
    #     ax.legend()
    #     plt.show()

# --- 使用示例：构建低轨卫星数据集 ---
if __name__ == "__main__":
    # 1. 初始化环境
    env = ITSNScenario(rng_seed=42)
    
    # 2. 模拟卫星运动并收集信道数据
    # 假设卫星过顶：仰角从 10度 -> 90度 -> 10度，方位角变化
    elevation_trajectory = np.concatenate([np.linspace(10, 90, 20), np.linspace(90, 10, 20)])
    azimuth_trajectory = np.linspace(80, 100, 40) # 假设方位角变化不大
    
    dataset = []
    
    print("开始生成轨道数据...")
    for t, (ele, azi) in enumerate(zip(elevation_trajectory, azimuth_trajectory)):
        # 更新卫星位置
        env.update_satellite_position(ele, azi)
        
        # 生成当前时刻所有信道
        channels = env.generate_channels()
        
        # 你可以在这里加入 DRL 需要的状态信息提取
        # 例如：将 current_state = {channels, ele, azi, t} 存入列表
        snapshot = {
            'time_step': t,
            'satellite_angle': (ele, azi),
            'channels': channels
        }
        dataset.append(snapshot)
        
    print(f"数据生成完毕，共 {len(dataset)} 个时间步。")
    print("W_sat shape:", dataset[0]['channels']['W_sat'].shape)
    
    # 可视化初始位置
    # env.plot_scenario()