"""
Baseline Optimizer: ZF + SDR Method
====================================
Python implementation of the MATLAB baseline algorithm for RIS-assisted ITSN.

This implements the Block Coordinate Descent (BCD) algorithm:
1. Fix RIS phase Phi, optimize BS beamforming W using Zero-Forcing
2. Fix W, optimize RIS phase Phi using Semi-Definite Relaxation (SDR)
3. Update satellite power to meet SINR constraint
4. Iterate until convergence

Reference: optimize_ris_zf_SDR.m
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Optional
import warnings
import mosek

class BaselineZFSDROptimizer:
    """
    Baseline optimizer using Zero-Forcing (ZF) beamforming and
    Semi-Definite Relaxation (SDR) for RIS phase optimization.
    """

    def __init__(
        self,
        K: int,  # Number of BS users
        J: int,  # Number of satellite users
        N_t: int,  # Number of BS antennas
        N_s: int,  # Number of satellite antennas
        N: int,  # Number of RIS elements
        P_max: float,  # Maximum BS power (W)
        sigma2: float,  # Noise power (W)
        gamma_k: np.ndarray,  # SINR thresholds for BS users (linear, not dB)
        gamma_j: float,  # SINR threshold for satellite users (linear)
        P_b: float = 0.001,  # BS power scaling factor
        P_s_init: float = 1.0,  # Initial satellite power scaling
        ris_amplitude_gain: float = 9.0,  # RIS amplitude gain
        N_iter: int = 20,  # Maximum iterations
        convergence_tol: float = 1e-2,  # Convergence tolerance for power
        verbose: bool = True
    ):
        self.K = K
        self.J = J
        self.N_t = N_t
        self.N_s = N_s
        self.N = N
        self.P_max = P_max
        self.P_sat_max = 100.0  # Maximum satellite power (W)
        self.sigma2 = sigma2
        self.gamma_k = gamma_k.reshape(-1, 1)  # (K, 1)
        self.gamma_j = gamma_j
        self.P_b = P_b
        self.P_s = P_s_init
        self.ris_gain = ris_amplitude_gain
        self.N_iter = N_iter
        self.convergence_tol = convergence_tol
        self.verbose = verbose


        # Initialize variables
        self.w = np.zeros((N_t, K), dtype=complex)  # BS beamforming
        self.Phi = self.ris_gain * np.eye(N, dtype=complex)
        # self.Phi = self.ris_gain * np.diag(np.exp(1j * np.random.uniform(0, 2*np.pi, N)))  # RIS phase matrix (initial: identity)
        self.W_sat = None  # Satellite beamforming (fixed input)

        # Tracking
        self.sum_rate_history = []
        self.power_history = []

    def optimize(
        self,
        h_k: np.ndarray,  # BS -> BS users (K, N_t)
        h_j: np.ndarray,  # BS -> SAT users (J, N_t)
        h_s_k: np.ndarray,  # SAT -> BS users (K, N_s)
        h_s_j: np.ndarray,  # SAT -> SAT users (J, N_s)
        h_k_r: np.ndarray,  # BS users -> RIS (K, N)
        h_j_r: np.ndarray,  # SAT users -> RIS (J, N)
        G_BS: np.ndarray,  # RIS -> BS (N, N_t)
        G_S: np.ndarray,  # RIS -> SAT (N, N_s)
        W_sat: np.ndarray  # Satellite beamforming (N_s, J)
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Run the BCD optimization algorithm.

        Returns:
        --------
        w_opt : np.ndarray, shape (N_t, K)
            Optimized BS beamforming matrix
        Phi_opt : np.ndarray, shape (N, N)
            Optimized RIS phase matrix (diagonal)
        info : dict
            Optimization information (sum_rate_history, power_history, etc.)
        """
        self.W_sat = W_sat
        self.sum_rate_history = []
        self.power_history = [0]  # P_sum[0] = 0 for convergence check

        # Initialize satellite power to ensure 17dB SNR
        # self._initialize_satellite_power(h_s_j)
        if self.verbose:
            print(f"\n[Initialization] Satellite power set to {self.P_s:.4f} W (target: 17dB SNR)")

        for iteration in range(self.N_iter):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.N_iter}")
                print(f"{'='*60}")

            # Step 1: Compute effective channels
            H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j = self._compute_effective_channels(
                h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S
            )

            if self.verbose:
                print("\n[Before ZF] Computing initial SINR...")
                self._print_sinr(H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j, self.w)

            # Step 2: Zero-Forcing Beamforming
            self.w = self._zero_forcing_beamforming(H_eff_k, H_eff_j)

            # Step 3: Power Allocation
            p = self._power_allocation(H_eff_k, H_sat_eff_k, self.w)

            # Step 4: Update beamforming with power
            # Scale each column of w by sqrt(p[k])
            self.w = self.w * np.sqrt(p).reshape(1, -1)  # Broadcasting: (N_t, K) * (1, K)

            # Step 5: Compute sum rate and SINR
            sum_rate, gamma_k_iter, gamma_j_iter = self._compute_sum_rate(
                H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j
            )
            self.sum_rate_history.append(sum_rate)

            if self.verbose:
                print(f"\n[After Power Allocation] Sum Rate: {sum_rate:.4f} bps/Hz")
                self._print_detailed_sinr(
                    H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j,
                    p, gamma_k_iter, gamma_j_iter
                )

            
            # Step 6: Optimize RIS phase
            self.Phi = self._optimize_ris_sdr(
                h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r,
                G_BS, G_S, gamma_k_iter, gamma_j_iter
            )

            # Step 7: Update effective channels with new Phi
            H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j = self._compute_effective_channels(
                h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S
            )
            if self.verbose:
                print("\n[After SDR] Computing initial SINR...")
                self._print_sinr(H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j, self.w)
            # Step 8: Update satellite power
            self._update_satellite_power(H_eff_j, H_sat_eff_j, gamma_j_iter)

            # Step 9: Compute total power
            self.P_bs = self.P_b * np.linalg.norm(self.w, 'fro') ** 2
            self.P_sat = self.P_s * np.linalg.norm(self.W_sat, 'fro') ** 2
            self.P_sum = self.P_bs + self.P_sat
            self.power_history.append(self.P_sum)
            
            self.all_sinr = self._print_sinr(H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j, self.w)
            if self.verbose:
                print(f"\nPower Summary:")
                print(f"  P_BS:  {self.P_bs:.4f} W")
                print(f"  P_SAT: {self.P_sat:.4f} W")
                print(f"  P_SUM: {self.P_sum:.4f} W")

            # Step 10: Check convergence
            if abs(self.power_history[-1] - self.power_history[-2]) < self.convergence_tol:
                if self.verbose:
                    print(f"\n>>> Converged at iteration {iteration + 1}")
                break

        # Prepare output
        info = {
            'sum_rate_history': self.sum_rate_history,
            'power_history': self.power_history[1:],  # Exclude initial 0
            'iterations': len(self.sum_rate_history),
            'final_P_bs': self.P_bs,
            'final_P_sat': self.P_sat,
            'final_P_sum': self.P_sum,
            'all_sinr': self.all_sinr
        }
        print(f"P_bs: {self.P_bs}, P_sat: {self.P_sat}, P_sum: {self.P_sum}")
        return self.w, self.Phi, info

        
    def evaluate_performance(
        self,
        w: np.ndarray,  # Fixed BS beamforming (N_t, K)
        Phi: np.ndarray,  # Fixed RIS phase matrix (N, N)
        h_k: np.ndarray,  # BS -> BS users (K, N_t)
        h_j: np.ndarray,  # BS -> SAT users (J, N_t)
        h_s_k: np.ndarray,  # SAT -> BS users (K, N_s)
        h_s_j: np.ndarray,  # SAT -> SAT users (J, N_s)
        h_k_r: np.ndarray,  # BS users -> RIS (K, N)
        h_j_r: np.ndarray,  # SAT users -> RIS (J, N)
        G_BS: np.ndarray,  # RIS -> BS (N, N_t)
        G_S: np.ndarray,  # RIS -> SAT (N, N_s)
        W_sat: np.ndarray  # Satellite beamforming (N_s, J)
    ) -> Dict:
        """
        Evaluate performance with fixed optimization (for robustness testing).

        This method computes power consumption and sum rate using fixed w and Phi,
        without re-optimizing. Used to test robustness when satellite moves.

        Returns:
        --------
        info : dict
            Performance metrics with same format as optimize()
        """
        # Store fixed optimization
        self.w = w
        self.Phi = Phi
        self.W_sat = W_sat

        # Compute effective channels with fixed Phi
        H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j = self._compute_effective_channels(
            h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S
        )

        # Compute sum rate and SINR
        sum_rate, gamma_k_iter, gamma_j_iter = self._compute_sum_rate(
            H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j
        )

        # Compute power consumption
        P_bs = self.P_b * np.linalg.norm(self.w, 'fro') ** 2
        P_sat = self.P_s * self.N_s
        P_sum = P_bs + P_sat

        # Prepare output (same format as optimize)
        info = {
            'sum_rate_history': [sum_rate],
            'power_history': [P_sum],
            'iterations': 0,  # No optimization performed
            'final_P_bs': P_bs,
            'final_P_sat': P_sat,
            'final_P_sum': P_sum,
            'gamma_k': gamma_k_iter,
            'gamma_j': gamma_j_iter
        }

        return info

    def _compute_effective_channels(
        self, h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute effective channels with RIS reflection."""
        # BS -> BS users effective channel
        H_eff_k = h_k + h_k_r @ self.Phi @ G_BS  # (K, N_t)

        # BS -> SAT users effective channel
        H_eff_j = h_j + h_j_r @ self.Phi @ G_BS  # (J, N_t)

        # SAT -> BS users effective channel
        H_sat_eff_k = h_s_k + h_k_r @ self.Phi @ G_S  # (K, N_s)

        # SAT -> SAT users effective channel
        H_sat_eff_j = h_s_j + h_j_r @ self.Phi @ G_S  # (J, N_s)

        return H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j

    def _zero_forcing_beamforming(
        self, H_eff_k: np.ndarray, H_eff_j: np.ndarray
    ) -> np.ndarray:
        """
        Zero-Forcing beamforming for all users (BS users + SAT users).

        Returns:
        --------
        w : np.ndarray, shape (N_t, K)
            ZF beamforming vectors (only for BS users)
        """
        # Combine all users' channels
        H_combine = np.vstack([H_eff_k, H_eff_j]).T  # (K+J, N_t)

        # ZF: W = H^H (H H^H)^{-1}
        H_H = H_combine.conj().T  # (N_t, K+J)
        HHH = H_H @ H_combine  # (K+J, K+J)

        # Check rank and add regularization if needed
        rank_H = np.linalg.matrix_rank(H_combine)
        if rank_H < self.K + self.J:
            if self.verbose:
                print(f"  [Warning] Channel matrix rank deficient: {rank_H} < {self.K + self.J}")
            reg_factor = 1e-6 * np.eye(self.K + self.J)
        else:
            reg_factor = 1e-6 * np.eye(self.K + self.J)

        # Compute ZF weights
        w_all = H_combine @ np.linalg.inv(HHH)  # (N_t, K+J)

        # Extract only BS users' beamforming vectors
        w = w_all[:, :self.K]  # (N_t, K)
        w = w /np.linalg.norm(w, 'fro')  # Normalize
        return w

    def _power_allocation(
        self, H_eff_k: np.ndarray, H_sat_eff_k: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """
        Power allocation to meet SINR constraints.
        Uses water-filling if power exceeds P_max.

        Returns:
        --------
        p : np.ndarray, shape (K,)
            Power allocation coefficients
        """
        p = np.zeros(self.K)

        for k in range(self.K):
            # Signal power (before power scaling)
            signal_power = self.P_b * np.abs(H_eff_k[k].conj() @ w[:, k]) ** 2

            # Interference from other BS users
            interference = 0
            for m in range(self.K):
                if m != k:
                    interference += self.P_b * np.abs(H_eff_k[k].conj() @ w[:, m]) ** 2

            # Interference from satellite
            for j in range(self.J):
                interference += self.P_s * np.abs(H_sat_eff_k[k].conj() @ self.W_sat[:, j]) ** 2

            # Required power to meet SINR threshold
            # gamma_k * (interference + sigma2) / signal_power = p[k]
            p[k] = max(self.gamma_k[k, 0] * (interference + self.sigma2) / signal_power, 0)

        # Check total power constraint
        total_power = np.linalg.norm(np.sqrt(p).reshape(-1, 1) * w.T, 'fro') ** 2

        if self.verbose:
            print(f"  [Power Allocation] Total power: {total_power:.4f} W (Max: {self.P_max:.4f} W)")

        if total_power > self.P_max:
            if self.verbose:
                print(f"  [Warning] Power exceeds limit, applying water-filling...")
            p = self._water_filling_power_allocation(H_eff_k, w, H_sat_eff_k)

        return p

    def _water_filling_power_allocation(
        self, H_eff_k: np.ndarray, w: np.ndarray, H_sat_eff_k: np.ndarray
    ) -> np.ndarray:
        """
        Water-filling power allocation under power constraint.
        """
        mu = 1e-3  # Initial water level
        step = 1e-2
        max_iter = 1000
        epsilon = 1e-6

        # Compute channel gains
        g = np.zeros(self.K)
        for k in range(self.K):
            g[k] = np.abs(H_eff_k[k].conj() @ w[:, k]) ** 2

            # Interference
            interference = self.sigma2
            for j in range(self.J):
                interference += np.abs(H_sat_eff_k[k].conj() @ self.W_sat[:, j]) ** 2

            g[k] = g[k] / (self.gamma_k[k, 0] * interference)

        # Sort gains
        idx = np.argsort(g)[::-1]  # Descending order
        g_sorted = g[idx]
        idx_map = np.zeros(self.K, dtype=int)
        idx_map[idx] = np.arange(self.K)

        # Water-filling iteration
        for iteration in range(max_iter):
            p = np.zeros(self.K)
            active_users = 0

            # Find active users
            for i in range(self.K):
                if 1 / g_sorted[i] < mu:
                    active_users += 1
                else:
                    break

            if active_users > 0:
                # Allocate power to active users
                for i in range(active_users):
                    p[idx_map[idx[i]]] = mu - 1 / g_sorted[i]

                # Compute total power
                total_power = np.sum(p)

                # Adjust water level
                if abs(total_power - self.P_max) < epsilon:
                    break
                elif total_power > self.P_max:
                    mu = mu - step
                    step = step / 2
                else:
                    mu = mu + step
            else:
                mu = mu + step

        return np.maximum(p, 0)
    import numpy as np

    
    def _compute_sum_rate(
        self, H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute sum rate and SINR for all users.
        """
        sum_rate = 0
        gamma_k_iter = np.zeros(self.K)
        gamma_j_iter = np.zeros(self.J)

        # BS users
        for k in range(self.K):
            signal = self.P_b * np.abs(H_eff_k[k].conj() @ self.w[:, k]) ** 2
            interference = self.sigma2

            for m in range(self.K):
                if m != k:
                    interference += self.P_b * np.abs(H_eff_k[k].conj() @ self.w[:, m]) ** 2

            for j in range(self.J):
                interference += self.P_s * np.abs(H_sat_eff_k[k].conj() @ self.W_sat[:, j]) ** 2

            gamma_k_iter[k] = signal / interference
            sum_rate += np.log2(1 + gamma_k_iter[k])

        # Satellite users
        for j in range(self.J):
            signal = self.P_s * np.abs(H_sat_eff_j[j].conj() @ self.W_sat[:, j]) ** 2
            interference = self.sigma2

            for k in range(self.K):
                interference += self.P_b * np.abs(H_eff_j[j].conj() @ self.w[:, k]) ** 2

            gamma_j_iter[j] = signal / interference
            sum_rate += np.log2(1 + gamma_j_iter[j])

        return sum_rate, gamma_k_iter, gamma_j_iter

    def _optimize_ris_sdr(
        self, h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r,
        G_BS, G_S, gamma_k_iter, gamma_j_iter
    ) -> np.ndarray:
        """
        Optimize RIS phase shifts using Semi-Definite Relaxation (SDR).
        """
        if self.verbose:
            print("\n[SDR] Optimizing RIS phase shifts...")

        # Prepare matrices (following MATLAB code structure)
        A, a_, a = self._prepare_A_matrices(h_k_r, G_BS, self.w, h_k)
        B, b_, b = self._prepare_B_matrices(h_k_r, G_S, self.W_sat, h_s_k)
        C, c_, c = self._prepare_C_matrices(h_j_r, G_BS, self.w, h_j)
        D, d_, d = self._prepare_D_matrices(h_j_r, G_S, self.W_sat, h_s_j)

        # Construct augmented matrices R
        Ra, Rb, Rc, Rd = self._construct_R_matrices(
            A, a_, a, B, b_, b, C, c_, c, D, d_, d
        )

        if self.verbose:
            print("  [SDR] After constructing augmented matrices R...")
            gamma_k_, gamma_j_ = self._compute_sinr_from_sdr_matrices(Ra, Rb, Rc, Rd, a, b, c, d, self.Phi)
            print(f"    [SDR] SINR from SDR: {10*np.log10(gamma_k_)} dB")
            print(f"    [SDR] SINR from SDR: {10*np.log10(gamma_j_)} dB")

        # Solve SDR using CVXPY
        V, alpha_t, beta_t, cvx_status = self._solve_sdr_cvxpy(
            Ra, Rb, Rc, Rd, a, b, c, d, gamma_k_iter, gamma_j_iter
        )

        if cvx_status in ['optimal', 'optimal_inaccurate']:
            gamma_k, gamma_j = self.sinr_check_using_V(
                Ra, Rb, Rc, Rd, a, b, c, d,
                V = V,
                scale=1e4,                 # 如果优化里用 scale=1e4，这里也要一样
                use_real_each_term=False,  # 模仿旧CVXPY：最后整体 real
                verbose=self.verbose
            )
            # Gaussian randomization to recover phase
            phi = self._gaussian_randomization(
                V, Ra, Rb, Rc, Rd, a, b, c, d
            ).reshape(-1)
            Phi_new = self.ris_gain * np.diag(phi)
            return Phi_new
        else:
            if self.verbose:
                print(f"  [Warning] SDR solver failed with status: {cvx_status}")
            return self.Phi  # Keep previous Phi
    import numpy as np

    def sinr_check_using_V(self, Ra, Rb, Rc, Rd, a, b, c, d, V = None,
                        scale=1.0, use_real_each_term=True,
                        eps_floor=1e-15, verbose=True):
        """
        用 V=vv^H + trace(RV) 形式，完全参照优化器来验算“优化前”的 SINR。
        不使用 quad()。

        参数：
        Phi: 若不传，默认 self.Phi
        scale: 若优化里对 R/a/b/c/d/noise 做了缩放，这里也传同样 scale
        use_real_each_term: True 表示每一项都取 real（更稳/更物理）
                            False 表示像你旧CVXPY那样最后整体取 real
        返回：
        gamma_k (K,), gamma_j (J,), 以及 V
        """
        if V is None:
            Phi = self.Phi
            # 1) 构造 v（按你指定的形式）
            phi = np.diag(Phi)[:self.N] / self.ris_gain
            v = np.concatenate([phi, [1.0]]).reshape(-1, 1)  # (N+1,1)

            # 2) 构造 V = v v^H
            V = v @ v.conj().T  # (N+1,N+1)

        # 可选：检查 diag(V)=1（理论上应接近 1）
        if verbose:
            diag_err = np.max(np.abs(np.diag(V) - 1))
            min_eig = np.min(np.linalg.eigvalsh((V + V.conj().T) / 2))
            print(f"[V check] diag_err={diag_err:.2e}, min_eig={min_eig:.2e}")

        # 3) 缩放（保持与优化一致）
        Ra_s, Rb_s, Rc_s, Rd_s = scale * Ra, scale * Rb, scale * Rc, scale * Rd
        a_s, b_s, c_s, d_s = scale * a, scale * b, scale * c, scale * d
        noise = scale * self.sigma2

        def tr_term(R, scalar):
            """trace(R@V)+scalar"""
            return np.trace(R @ V) + scalar

        def real_term(x):
            return np.real(x)

        K = self.K
        J = self.J
        gamma_k = np.zeros(K, dtype=float)
        gamma_j = np.zeros(J, dtype=float)

        # ===== BS 用户 SINR =====
        for k in range(K):
            # signal
            sig_raw = tr_term(Ra_s[k, k], a_s[k, k])
            sig = self.P_b * (real_term(sig_raw) if use_real_each_term else sig_raw)

            # interference from other BS users
            interf_bs_sum = 0.0 + 0.0j
            for m in range(K):
                if m != k:
                    x = tr_term(Ra_s[k, m], a_s[k, m])
                    interf_bs_sum += (real_term(x) if use_real_each_term else x)

            # interference from satellite
            sat_raw = tr_term(Rb_s[k], b_s[k])
            interf_sat = self.P_s * (real_term(sat_raw) if use_real_each_term else sat_raw)

            # total interference
            if use_real_each_term:
                interf_total = self.P_b * interf_bs_sum + interf_sat + noise
                interf_total_val = float(np.real(interf_total))
            else:
                # 模仿你旧CVXPY写法：最后整体 real
                interf_total_val = float(np.real(self.P_b * interf_bs_sum + interf_sat) + noise)

            interf_total_val = max(interf_total_val, eps_floor)

            sig_val = float(np.real(sig)) if not use_real_each_term else float(sig)
            gamma_k[k] = sig_val / interf_total_val

            if verbose:
                print(f"[BS k={k}] sig={sig_val:.3e}, "
                    f"intf_bs={float(np.real(self.P_b*interf_bs_sum)):.3e}, "
                    f"intf_sat={float(np.real(interf_sat)):.3e}, "
                    f"noise={noise:.3e}, SINR={10*np.log10(gamma_k[k]):.3e}, "
                    f"imag(sig_raw)={np.imag(sig_raw):.1e}")

        # ===== 卫星用户 SINR =====
        for j in range(J):
            sig_raw = tr_term(Rd_s, d_s)
            sig = self.P_s * (real_term(sig_raw) if use_real_each_term else sig_raw)

            interf_bs_to_sat = 0.0 + 0.0j
            for k in range(K):
                x = tr_term(Rc_s[k], c_s[k])
                interf_bs_to_sat += (real_term(x) if use_real_each_term else x)

            if use_real_each_term:
                interf_total_val = float(np.real(self.P_b * interf_bs_to_sat) + noise)
            else:
                interf_total_val = float(np.real(self.P_b * interf_bs_to_sat) + noise)

            interf_total_val = max(interf_total_val, eps_floor)

            sig_val = float(np.real(sig)) if not use_real_each_term else float(sig)
            gamma_j[j] = sig_val / interf_total_val

            if verbose:
                print(f"[SAT j={j}] sig={sig_val:.3e}, "
                    f"intf_bs={float(np.real(self.P_b*interf_bs_to_sat)):.3e}, "
                    f"noise={noise:.3e}, SINR={10*np.log10(gamma_j[j]):.3e}, "
                    f"imag(sig_raw)={np.imag(sig_raw):.1e}")

        return gamma_k, gamma_j

    def _prepare_A_matrices(self, h_k_r, G_BS, w, h_k):
        """Prepare A matrices for SDR (BS interference terms)."""
        A = np.zeros((self.K, self.K, self.N, self.N), dtype=complex)
        a_ = np.zeros((self.K, self.K, self.N), dtype=complex)
        a = np.zeros((self.K, self.K), dtype=complex)
        for k in range(self.K):
            for m in range(self.K):
                w_m = w[:, m:m+1] 
                Wm = w_m @ w_m.conj().T
                diag_h_kr = np.diag(self.ris_gain * h_k_r[k])
                A[k, m] = diag_h_kr.conj() @ G_BS.conj()@ Wm @ G_BS.T @ diag_h_kr.T
                a_[k, m] = diag_h_kr.conj() @ G_BS.conj()@ Wm @ h_k[k].T
                a[k, m] = h_k[k].conj() @ Wm @ h_k[k].T

        return A, a_, a

    def _prepare_B_matrices(self, h_k_r, G_S, W_sat, h_s_k):
        """Prepare B matrices for SDR (Satellite interference to BS users)."""
        B = np.zeros((self.K, self.N, self.N), dtype=complex)
        b_ = np.zeros((self.K, self.N), dtype=complex)
        b = np.zeros(self.K, dtype=complex)

        WW = W_sat @ W_sat.conj().T

        for k in range(self.K):
            diag_h_kr = np.diag(self.ris_gain * h_k_r[k])
            B[k] = diag_h_kr.conj() @ G_S.conj() @ WW @ G_S.T @ diag_h_kr.T
            b_[k] = diag_h_kr.conj() @ G_S.conj() @ WW @ h_s_k[k].T
            # b(k) = W_sat' * (h_s_k(k,:)' * h_s_k(k,:)) * W_sat
            b[k] = h_s_k[k].conj() @ WW @ h_s_k[k].T
            

        return B, b_, b

    def _prepare_C_matrices(self, h_j_r, G_BS, w, h_j):
        """Prepare C matrices for SDR (BS interference to SAT users)."""
        C = np.zeros((self.K, self.N, self.N), dtype=complex)  # Note: K instead of J for consistency
        c_ = np.zeros((self.K, self.N), dtype=complex)
        c = np.zeros(self.K, dtype=complex)

        for k in range(self.K):
            w_k = w[:, k:k+1]
            Wk = w_k @ w_k.conj().T
            diag_h_jr = np.diag(self.ris_gain * h_j_r[0])  # Assuming single SAT user, use h_j_r[0]
            C[k] = diag_h_jr.conj() @ G_BS.conj() @ Wk @ G_BS.T @ diag_h_jr.T
            c_[k] = diag_h_jr.conj() @ G_BS.conj() @ Wk @ h_j[0].T
            c[k] = h_j[0].conj() @ Wk @ h_j[0].T

        return C, c_, c

    def _prepare_D_matrices(self, h_j_r, G_S, W_sat, h_s_j):
        """Prepare D matrices for SDR (Satellite signal to SAT users)."""
        WW = W_sat @ W_sat.conj().T
        diag_h_jr = np.diag(self.ris_gain * h_j_r[0])

        D = diag_h_jr.conj() @ G_S.conj() @ WW @ G_S.T @ diag_h_jr.T
        d_ = diag_h_jr.conj() @ G_S.conj() @ WW @ h_s_j[0].T
        # d = W_sat' * (h_s_j' * h_s_j) * W_sat
        d = h_s_j[0].conj() @ WW @ h_s_j[0].T

        return D, d_, d

    def _construct_R_matrices(self, A, a_, a, B, b_, b, C, c_, c, D, d_, d):
        """Construct augmented matrices for SDR."""
        Ra = np.zeros((self.K, self.K, self.N + 1, self.N + 1), dtype=complex)
        Rb = np.zeros((self.K, self.N + 1, self.N + 1), dtype=complex)
        Rc = np.zeros((self.K, self.N + 1, self.N + 1), dtype=complex)

        for k in range(self.K):
            for m in range(self.K):
                Ra[k, m, :self.N, :self.N] = A[k, m]
                Ra[k, m, :self.N, self.N] = a_[k, m]
                Ra[k, m, self.N, :self.N] = a_[k, m].conj()
                Ra[k, m, self.N, self.N] = 0

            Rb[k, :self.N, :self.N] = B[k]
            Rb[k, :self.N, self.N] = b_[k]
            Rb[k, self.N, :self.N] = b_[k].conj()
            Rb[k, self.N, self.N] = 0
            Rc[k, :self.N, :self.N] = C[k]
            Rc[k, :self.N, self.N] = c_[k]
            Rc[k, self.N, :self.N] = c_[k].conj()
            Rc[k, self.N, self.N] = 0
        Rd = np.zeros((self.N + 1, self.N + 1), dtype=complex)
        Rd[:self.N, :self.N] = D
        Rd[:self.N, self.N] = d_
        Rd[self.N, :self.N] = d_.conj()
        Rd[self.N, self.N] = 0
        return Ra, Rb, Rc, Rd

    def _solve_sdr_cvxpy(self, Ra, Rb, Rc, Rd, a, b, c, d, gamma_k_iter, gamma_j_iter):
        
        """Solve SDR problem using CVXPY with robust solver selection."""
        def _hermitize(M):
            return M
        N = self.N
        K = self.K

        # --------- 0) 强制 Hermitian 化（非常重要，能减少虚部/数值崩溃） ----------
        # Ra: (K,K,N+1,N+1)
        for k in range(K):
            for m in range(K):
                Ra[k, m] = _hermitize(Ra[k, m])
        # Rb: (K,N+1,N+1), Rc: (K,N+1,N+1)
        for k in range(K):
            Rb[k] = _hermitize(Rb[k])
            Rc[k] = _hermitize(Rc[k])
        Rd = _hermitize(Rd)
        
        scale = 1e8
        Ra_s, Rb_s, Rc_s, Rd_s = scale * Ra, scale * Rb, scale * Rc, scale * Rd
        a_s, b_s, c_s, d_s = scale * a, scale * b, scale * c, scale * d
        noise = scale * self.sigma2

        # --------- 2) 变量 ----------
        V = cp.Variable((N + 1, N + 1), hermitian=True)
        alpha_t = cp.Variable(K, nonneg=True)
        beta_t = cp.Variable(nonneg=True)

        # --------- 3) 目标 ----------
        objective = cp.Maximize(cp.min(alpha_t) + 32.0 * beta_t)

        # --------- 4) 约束 ----------
        constraints = [
            V >> 0,
            cp.diag(V) == 1
        ]

        # BS 用户 SINR 约束
        for k in range(K):
            signal = self.P_b * cp.real(cp.trace(Ra_s[k, k] @ V) + a_s[k, k])

            interf_bs = 0
            for m in range(K):
                if m != k:
                    # 每一项都取 real，更稳
                    interf_bs += cp.real(cp.trace(Ra_s[k, m] @ V) + a_s[k, m])

            interf_sat = cp.real(cp.trace(Rb_s[k] @ V) + b_s[k])

            interf_total = self.P_b * interf_bs + self.P_s * interf_sat + noise

            constraints.append(
                signal >= gamma_k_iter[k] * interf_total + alpha_t[k]
            )

        # 卫星用户 SINR 约束（你这里 gamma_j_iter[0]）
        signal_sat = self.P_s * cp.real(cp.trace(Rd_s @ V) + d_s)

        interf_bs_to_sat = 0
        for k in range(K):
            interf_bs_to_sat += cp.real(cp.trace(Rc_s[k] @ V) + c_s[k])

        interf_to_sat = self.P_b * interf_bs_to_sat + noise

        constraints.append(
            signal_sat >= gamma_j_iter[0] * interf_to_sat + beta_t
        )

        problem = cp.Problem(objective, constraints)

        mosek_params = {
            # 收紧可行性容差，迫使求解器更精确
            mosek.dparam.intpnt_co_tol_pfeas: 1.0e-8,
            mosek.dparam.intpnt_co_tol_dfeas: 1.0e-8,
            mosek.dparam.intpnt_co_tol_rel_gap: 1.0e-8,
            mosek.dparam.intpnt_co_tol_infeas: 1.0e-8,
            # 强制使用对偶形式 (通常对 SDP 更快且更稳)
            mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
            # 如果数值问题依然存在，允许 aggressive scaling
            # mosek.iparam.intpnt_scaling: mosek.scaling.aggressive,
        }

        # --------- 6) 求解 ----------
        try:
            if self.verbose:
                print(f"[SDR] Solving with MOSEK...")
            
            # 调用 MOSEK 并传入参数
            problem.solve(solver=cp.MOSEK, verbose=self.verbose, mosek_params=mosek_params)

            status = problem.status
            if self.verbose:
                print(f"[SDR] MOSEK status: {status}")

            if status in ["optimal", "optimal_inaccurate"]:
                Vv = V.value

                # 检查秩 (Rank)
                eigvals = np.linalg.eigvalsh(Vv)
                rank_eff = np.sum(eigvals > 1e-4 * np.max(eigvals))
                
                if self.verbose:
                    print(f"[SDR] Result Rank: {rank_eff}")
                    print(f"[SDR] Max Eigenvalue ratio: {np.max(eigvals)/np.sum(eigvals):.4f}")

                return Vv, alpha_t.value, beta_t.value, status
            else:
                return None, None, None, status

        except Exception as e:
            if self.verbose:
                print(f"[SDR] MOSEK failed: {e}")
            return None, None, None, f"failed: {e}"
    def _optimize_ris_simple(
        self, h_k, h_j, h_s_k, h_s_j, h_k_r, h_j_r, G_BS, G_S
    ) -> np.ndarray:
        """
        Optimize RIS phase using simple random search method.
        Much faster than SDR but may be suboptimal.
        """
        if self.verbose:
            print("\n[Simple RIS] Optimizing RIS phase using random search...")

        phi = self.simple_ris_opt.optimize_random_search(
            h_k_r, h_j_r, G_BS, G_S,
            self.w, self.W_sat,
            h_k, h_j, h_s_k, h_s_j,
            self.P_b, self.P_s, self.sigma2,
            ris_gain=self.ris_gain
        )

        Phi_new = self.ris_gain * np.diag(phi)

        if self.verbose:
            print(f"  [Simple RIS] Optimization completed")

        return Phi_new

    def _gaussian_randomization(self, V, Ra, Rb, Rc, Rd, a, b, c, d, L=5000):
        """
        Gaussian randomization to recover phase from SDR solution.
        Fixed for numerical stability (negative eigenvalues).
        """
        if V is None:
            return np.ones(self.N, dtype=complex)

        # 1. 特征值分解
        # eigh 返回的是升序排列 (smallest to largest)
        eigvals, U = np.linalg.eigh(V)

        # 2. 关键修正：处理负特征值
        # 数值误差会导致微小的负特征值 (e.g. -1e-9)，导致 sqrt 产生 NaN
        eigvals = np.maximum(eigvals, 0.0)

        # 3. 准备随机化矩阵
        sqrt_Sigma = np.diag(np.sqrt(eigvals))
        
        # 4. 提取主要成分 (Principal Component) - 对应最大的特征值
        # 由于 eigh 是升序，最后一个就是最大的
        idx_max = -1 
        v_principal = U[:, idx_max] * np.sqrt(eigvals[idx_max])
        
        # 归一化相位 (对齐最后一个元素，使其相位为 0，即对应辅助变量 t=1)
        # 避免除以 0 的风险
        ref = v_principal[-1]
        if np.abs(ref) < 1e-10: 
            ref = 1.0
        v_principal_aligned = np.exp(1j * np.angle(v_principal / ref))

        # --- 初始化最优解为主要成分 ---
        max_F = self._compute_randomization_objective(v_principal_aligned, Ra, Rb, Rc, Rd, a, b, c, d)
        max_v = v_principal_aligned
        
        if self.verbose:
            print(f"  [Randomization] Principal Component SINR obj: {max_F:.4f}")

        # 5. 开始随机化循环
        # 生成 L 个随机向量 r ~ CN(0, I)
        # 投影回原空间: xi = U * sqrt(Sigma) * r
        # 这里的矩阵乘法可以批量处理以加速，为了代码清晰保持循环
        ratio = np.max(eigvals)/np.sum(eigvals)
        for l in range(L):
            # 生成标准复高斯噪声
            r = (np.random.randn(self.N + 1) + 1j * np.random.randn(self.N + 1)) / np.sqrt(2)
            
            # 构造候选解 xi
            # 注意：这里的数学含义是生成均值为 0，协方差为 V 的随机向量
            xi = (1-ratio) * U @ (sqrt_Sigma @ r) + v_principal
            
            # 提取相位
            ref = xi[-1]
            if np.abs(ref) < 1e-10: continue # 跳过异常值
            
            # 关键：SDR 的解 v = [phi; 1]，所以我们要把最后一个元素的相位旋转回 0 (即实数 1)
            # 这样前 N 个元素的相位就是 phi
            v_cand = np.exp(1j * np.angle(xi / ref))
            
            # 计算目标函数值
            F = self._compute_randomization_objective(v_cand, Ra, Rb, Rc, Rd, a, b, c, d)

            if F > max_F:
                max_F = F
                max_v = v_cand
                # if self.verbose and l % 100 == 0:
                #    print(f"    [Randomization] New best found at iter {l}: {max_F:.4f}")

        # 6. 提取最终相位 (前 N 个元素)
        # max_v 的形状可能是 (N+1,) 或 (N+1, 1)，reshape 确保安全
        phi = v_principal_aligned.flatten()[:self.N]

        return phi
    
    def _compute_randomization_objective(self, v, Ra, Rb, Rc, Rd, a, b, c, d):
        """Compute sum rate for a given phase vector (used in randomization)."""
        F = 0
        sum_partC = 0

        # BS users
        for k in range(self.K):
            sum_part = 0
            Ra_kk = Ra[k, k]
            Rb_k = Rb[k]
            Rc_k = Rc[k]
            c_k = c[k]

            sum_partC += np.abs(v.conj().T @ Rc_k @ v + c_k)

            for m in range(self.K):
                if m != k:
                    Ra_km = Ra[k, m]
                    a_km = a[k, m]
                    sum_part += np.abs(v.conj().T @ Ra_km @ v + a_km)

            a_kk = a[k, k]
            b_k = b[k]

            signal = self.P_b * np.abs(v.conj().T @ Ra_kk @ v + a_kk)
            interference = np.abs(self.P_b * sum_part + self.P_s * np.abs(v.conj().T @ Rb_k @ v + b_k) + self.sigma2)

            F += np.log2(1 + signal / (interference))
        # Satellite user
        signal_sat = self.P_s * np.abs(v.conj().T @ Rd @ v + d)
        interference_sat = np.abs(self.P_b * sum_partC + self.sigma2)

        F += np.log2(1 + signal_sat / (interference_sat))

        return F

    def _compute_sinr_from_sdr_matrices(self, Ra, Rb, Rc, Rd, a, b, c, d, Phi):
        """
        使用 SDR 矩阵参数计算各用户 SINR，用于验证 SDR 转换是否正确。

        Parameters:
        -----------
        Ra : np.ndarray, shape (K, K, N+1, N+1)
            BS 用户间干扰矩阵（包括 RIS 反射）
        Rb : np.ndarray, shape (K, N+1, N+1)
            卫星对 BS 用户的干扰矩阵（包括 RIS 反射）
        Rc : np.ndarray, shape (K, N+1, N+1)
            BS 对卫星用户的干扰矩阵（包括 RIS 反射）
        Rd : np.ndarray, shape (N+1, N+1)
            卫星用户信号矩阵（包括 RIS 反射）
        a : np.ndarray, shape (K, K)
            BS 用户间直射路径项
        b : np.ndarray, shape (K,)
            卫星对 BS 用户直射路径项
        c : np.ndarray, shape (K,)
            BS 对卫星用户直射路径项
        d : complex
            卫星用户直射路径项
        Phi : np.ndarray, shape (N, N)
            RIS 相位矩阵（对角矩阵）

        Returns:
        --------
        gamma_k : np.ndarray, shape (K,)
            BS 用户的 SINR（线性值）
        gamma_j : np.ndarray, shape (J,)
            卫星用户的 SINR（线性值）
        """
        # 构造增广相位向量 v = [phi; 1]
        phi = np.diag(Phi)[:self.N] / self.ris_gain  # 提取相位，归一化
        v = np.concatenate([phi, [1.0]]).reshape(-1, 1)  # (N+1, 1)

        gamma_k = np.zeros(self.K)
        gamma_j = np.zeros(self.J)

        # 计算 BS 用户 SINR
        for k in range(self.K):
            # 信号功率: P_b * |v^H @ Ra[k,k] @ v + a[k,k]|
            signal_term = v.conj().T @ Ra[k, k] @ v + a[k, k]
            signal = self.P_b * np.abs(signal_term)

            # BS 用户间干扰
            interference_bs = 0
            for m in range(self.K):
                if m != k:
                    interference_term = v.conj().T @ Ra[k, m] @ v + a[k, m]
                    interference_bs += np.abs(interference_term)

            # 卫星干扰: P_s * |v^H @ Rb[k] @ v + b[k]|
            sat_interference_term = v.conj().T @ Rb[k] @ v + b[k]
            interference_sat = self.P_s * np.abs(sat_interference_term)

            # 总干扰
            interference = self.P_b * interference_bs + interference_sat + self.sigma2

            # SINR
            gamma_k[k] = signal / interference

        # 计算卫星用户 SINR
        for j in range(self.J):
            # 信号功率: P_s * |v^H @ Rd @ v + d|
            signal_term = v.conj().T @ Rd @ v + d
            signal = self.P_s * np.abs(signal_term)

            # BS 干扰: P_b * sum_k |v^H @ Rc[k] @ v + c[k]|
            interference_bs = 0
            for k in range(self.K):
                interference_term = v.conj().T @ Rc[k] @ v + c[k]
                interference_bs += np.abs(interference_term)

            # 总干扰
            interference = self.P_b * interference_bs + self.sigma2

            # SINR
            gamma_j[j] = signal / interference

        return gamma_k, gamma_j

    def _update_satellite_power(self, H_eff_j, H_sat_eff_j, gamma_j_iter):
        """Update satellite power to meet SINR constraint."""
        for j in range(self.J):
            signal_power = np.abs(H_sat_eff_j[j].conj() @ self.W_sat[:, j]) ** 2
            interference_power = self.sigma2

            for k in range(self.K):
                interference_power += self.P_b * np.abs(H_eff_j[j].conj() @ self.w[:, k]) ** 2

            # Update satellite power: P_s = gamma_j * interference / signal
            # Add upper bound to prevent explosion
            P_s_required = self.gamma_j * interference_power / signal_power
            self.P_s = min(P_s_required, self.P_sat_max / self.N_s)  # Cap at max power per antenna

            if self.verbose and P_s_required > self.P_sat_max / self.N_s:
                print(f"  [Warning] Satellite power capped at {self.P_s:.4f} (required: {P_s_required:.4f})")

    def _print_sinr(self, H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j, w):
        """Print SINR for debugging."""
        # BS users
        all_sinr = []
        for k in range(self.K):
            interference = self.sigma2
            for m in range(self.K):
                if m != k:
                    interference += self.P_b * np.abs(H_eff_k[k].conj() @ w[:, m]) ** 2
            for j in range(self.J):
                interference += self.P_s * np.abs(H_sat_eff_k[k].conj() @ self.W_sat[:, j]) ** 2

            signal = self.P_b * np.abs(H_eff_k[k].conj() @ w[:, k]) ** 2
            SINR_dB = 10 * np.log10(signal / interference + self.sigma2)
            all_sinr.append(signal / interference + self.sigma2)
            # print(f"  BS UE({k+1}) SINR: {SINR_dB:.2f} dB")

        # Satellite users
        for j in range(self.J):
            interference = self.sigma2
            for k in range(self.K):
                interference += self.P_b * np.abs(H_eff_j[j].conj() @ w[:, k]) ** 2

            signal = self.P_s * np.abs(H_sat_eff_j[j].conj() @ self.W_sat[:, j]) ** 2
            SINR_dB = 10 * np.log10(signal / interference + self.sigma2)
            all_sinr.append(signal / interference + self.sigma2)
            print(f"  SAT UE({j+1}) SINR: {SINR_dB:.2f} dB")
        return all_sinr

    def _print_detailed_sinr(self, H_eff_k, H_eff_j, H_sat_eff_k, H_sat_eff_j, p, gamma_k_iter, gamma_j_iter):
        """Print detailed SINR after power allocation."""
        for k in range(self.K):
            SINR_dB = 10 * np.log10(gamma_k_iter[k] + 1e-12)
            print(f"  BS UE({k+1}) SINR: {SINR_dB:.2f} dB (Threshold: {10*np.log10(self.gamma_k[k,0]):.2f} dB)")

        for j in range(self.J):
            SINR_dB = 10 * np.log10(gamma_j_iter[j] + 1e-12)
            print(f"  SAT UE({j+1}) SINR: {SINR_dB:.2f} dB (Threshold: {10*np.log10(self.gamma_j):.2f} dB)")

    def _initialize_satellite_power(self, h_s_j: np.ndarray):
        """
        Initialize satellite power to ensure 17dB SNR at the start.
        This prevents power explosion in subsequent iterations.

        Target: SNR = P_s * |h_s_j @ W_sat|^2 / sigma2 >= 17dB
        """
        SNR_target_dB = 17.0
        SNR_target_linear = 10 ** (SNR_target_dB / 10)  # ~50.12

        # Compute channel gain for satellite user
        for j in range(self.J):
            channel_gain = np.abs(h_s_j[j].conj() @ self.W_sat[:, j]) ** 2

            # Required power: P_s = SNR_target * sigma2 / channel_gain
            P_s_required = SNR_target_linear * self.sigma2 / channel_gain

            # Cap at maximum power per antenna
            self.P_s = P_s_required

            # Ensure minimum power
            self.P_s = max(self.P_s, 0.01)