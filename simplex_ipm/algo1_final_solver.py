"""
Algorithm 1: Feasible-Start, Fixed σ Interior Point Method for QP on Cartesian Product of Simplices.

Core implementation following the LaTeX specification.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.linalg import cho_factor, cho_solve
from simplex_ipm.helper.block_ops import apply_E, apply_E_transpose, validate_blocks
from simplex_ipm.helper.utils import fraction_to_boundary, norm_inf


class FeasibleStartIPM:
    """
    Feasible-start primal-dual interior point method for QP on Cartesian product of simplices.
    
    Problem: min (1/2) x^T Q x + q^T x  s.t.  Ex = 1, x >= 0
    where blocks {I_k} partition indices (simplex per block).
    """
    
    def __init__(self, Q, q, blocks, cfg=None):
        """
        Initialize solver.
        
        Parameters:
        -----------
        Q : array-like, shape (n, n)
            Dense or sparse symmetric matrix (Q >= 0).
        q : array-like, shape (n,)
            Linear term.
        blocks : list of lists
            blocks[k] contains indices in block k (0-indexed).
        cfg : dict, optional
            Configuration parameters (see default_config()).
        """
        # Convert Q to numpy array or sparse matrix
        if scipy.sparse.issparse(Q):
            self.Q = Q
            self.Q_dense = None
            self.is_sparse = True
        else:
            self.Q = np.asarray(Q, dtype=float)
            # Symmetrize if needed
            if not np.allclose(self.Q, self.Q.T):
                self.Q = (self.Q + self.Q.T) / 2
            self.Q_dense = self.Q
            self.is_sparse = False
        
        self.q = np.asarray(q, dtype=float)
        self.n = len(self.q)
        
        # Validate Q shape
        if self.Q.shape != (self.n, self.n):
            raise ValueError(f"Q shape {self.Q.shape} does not match q length {self.n}")
        
        # Validate blocks
        is_valid, error_msg = validate_blocks(blocks, self.n)
        if not is_valid:
            raise ValueError(f"Invalid blocks: {error_msg}")
        self.blocks = blocks
        self.n_blocks = len(blocks)
        
        # Configuration
        self.cfg = self.default_config()
        if cfg is not None:
            self.cfg.update(cfg)
        
        # State variables (initialized in initialize_delta_rule)
        self.x = None
        self.y = None
        self.z = None
        self.mu = None
        
        # Factorization cache
        self.M_factor = None
        self.M_reg = None
    
    @staticmethod
    def default_config():
        """Return default configuration."""
        return {
            'sigma': 0.1,  # Centering parameter in (0, 0.5)
            'max_iter': 100,
            'eps_feas': 1e-8,
            'eps_comp': 1e-8,
            'eps_delta': 1e-8,  # For δ-rule
            'tau_delta': 1e-2,  # For δ-rule
            'tau_reg': None,  # None for adaptive, or fixed value
            'K_th': 200,  # Threshold for assembling Schur matrix
            'pcg_tol': 1e-10,
            'pcg_maxit': 1000,
            'verbosity': 1,  # 0=silent, 1=summary, 2=verbose
            'gamma': 0.99,  # Fraction-to-boundary safety factor
        }
    
    def initialize_delta_rule(self):
        """
        Initialize strictly interior point via δ-rule.
        
        Returns:
        --------
        x : ndarray, shape (n,)
            Initial primal iterate (uniform per block).
        y : ndarray, shape (|K|,)
            Initial dual iterate (per-block).
        z : ndarray, shape (n,)
            Initial dual slack (z >= δ * 1).
        mu : float
            Initial complementarity gap.
        """
        # ==========================================
        # Step 1: Initialization (δ-rule)
        # ==========================================
        # Step 1.1: Uniform primal (x^0_i = 1/|I_k| for i in I_k)
        x = np.zeros(self.n, dtype=float)
        for k, block in enumerate(self.blocks):
            block_size = len(block)
            if block_size == 0:
                raise ValueError(f"Block {k} is empty")
            x[block] = 1.0 / block_size
        
        # Step 1.2: Compute w = Q x^0 + q
        if self.is_sparse:
            w = self.Q.dot(x) + self.q
        else:
            w = self.Q @ x + self.q
        
        # Step 1.3: Compute per-block δ_k
        delta_per_block = np.zeros(self.n_blocks, dtype=float)
        for k, block in enumerate(self.blocks):
            w_block = w[block]
            w_max = np.max(w_block)
            w_min = np.min(w_block)
            w_range = w_max - w_min
            delta_k = max(self.cfg['eps_delta'], self.cfg['tau_delta'] * w_range)
            delta_per_block[k] = delta_k
        
        # Step 1.4: Dual start (y_k = -min(w_Ik) + δ_k, z = w + E^T y)
        y = np.zeros(self.n_blocks, dtype=float)
        
        for k, block in enumerate(self.blocks):
            w_block = w[block]
            w_min = np.min(w_block)
            delta_k = delta_per_block[k]
            y[k] = -w_min + delta_k
            
        # Compute z = w + E^T y
        Ety = apply_E_transpose(y, self.blocks, self.n)
        z = w + Ety

        # Step 1.5: Compute initial gap
        mu = np.dot(x, z) / self.n
        
        # Sanity checks
        if np.any(x <= 0):
            raise ValueError("Initial x is not strictly positive")
        if np.any(z <= 0):
            raise ValueError("Initial z is not strictly positive")
        
        # Check feasibility (should be exact for uniform x)
        Ex = apply_E(x, self.blocks)
        if not np.allclose(Ex, 1.0, atol=1e-10):
            raise ValueError(f"Initial x does not satisfy Ex = 1: {Ex}")
        
        # Check dual feasibility (should be approximately satisfied)
        Ety = apply_E_transpose(y, self.blocks, self.n)
        r_D = w + Ety - z
        if norm_inf(r_D) > 1e-6:
            if self.cfg['verbosity'] >= 2:
                print(f"Warning: Initial dual residual is large: {norm_inf(r_D)}")
        
        self.x = x
        self.y = y
        self.z = z
        self.mu = mu
        
        return x, y, z, mu
    
    def compute_residuals(self):
        """
        Compute primal, dual, and complementarity residuals.
        
        Returns:
        --------
        r_D : ndarray, shape (n,)
            Dual residual: Qx + q + E^T y - z
        r_P : ndarray, shape (|K|,)
            Primal residual: Ex - 1
        r_C : ndarray, shape (n,)
            Complementarity residual: XZ 1 - μ_target 1
        mu_target : float
            Target complementarity: σ * μ
        """
        # Dual residual: r_D = Qx + q + E^T y - z
        if self.is_sparse:
            Qx = self.Q.dot(self.x)
        else:
            Qx = self.Q @ self.x
        Ety = apply_E_transpose(self.y, self.blocks, self.n)
        r_D = Qx + self.q + Ety - self.z
        
        # Primal residual: r_P = Ex - 1
        Ex = apply_E(self.x, self.blocks)
        r_P = Ex - 1.0
        
        # Complementarity residual: r_C = XZ 1 - μ_target 1
        # Note: In feasible-start, we want to drive XZ to σμ
        mu_target = self.cfg['sigma'] * self.mu
        XZ = self.x * self.z
        r_C = XZ - mu_target
        
        return r_D, r_P, r_C, mu_target
    
    def build_H_and_factorize(self):
        """
        Build H = Q + X^{-1}Z and factorize using Cholesky.
        
        Returns:
        --------
        H_factor : object
            Factorization object (Cholesky factor or sparse solver).
        H_reg : float or None
            Regularization parameter used (if any).
        """
        # Safe division to avoid ZeroDivisionError
        x_safe = np.maximum(self.x, 1e-14)
        x_inv_z = self.z / x_safe
        
        if self.is_sparse:
            # Sparse case: H = Q + diag(z/x)
            X_inv_Z = scipy.sparse.diags(x_inv_z, format='csc')
            H = self.Q + X_inv_Z
            
            tau = self.cfg['tau_reg']
            if tau is None:
                # Adaptive regularization
                H_norm_inf = np.max(np.abs(H).sum(axis=1).A1)
                tau = max(0.0, min(1e-8, 1e-10 * H_norm_inf))
            
            if tau > 0.0:
                n = H.shape[0]
                H = H + tau * scipy.sparse.eye(n, format='csc')
                self.H_reg = tau
            else:
                self.H_reg = None
            
            H_csc = H if scipy.sparse.isspmatrix_csc(H) else H.tocsc()
            try:
                # For sparse symmetric, we can use splu or a sparse Cholesky if available.
                # SciPy doesn't have a built-in sparse Cholesky, so we use splu.
                self.H_factor = scipy.sparse.linalg.splu(
                    H_csc,
                    permc_spec=self.cfg.get('permc_spec', 'MMD_AT_PLUS_A'),
                )
            except Exception:
                bump = max(1e-12, (self.H_reg or 0.0) * 10 or 1e-12)
                H_csc = H_csc + bump * scipy.sparse.eye(H.shape[0], format="csc")
                self.H_reg = (self.H_reg or 0.0) + bump
                self.H_factor = scipy.sparse.linalg.splu(H_csc, permc_spec="MMD_AT_PLUS_A")
            
            return self.H_factor, self.H_reg
        
        else:
            # --- Dense path (Cholesky) ---
            H = self.Q + np.diag(x_inv_z)
                
            tau = self.cfg['tau_reg']
            if tau is None:
                H_norm_inf = norm_inf(H)
                tau = max(0.0, min(1e-8, 1e-10 * H_norm_inf))
            
            if tau > 0.0:
                H = H + tau * np.eye(self.n)
                self.H_reg = tau
            else:
                self.H_reg = None
            
            try:
                self.H_factor = cho_factor(H, lower=False)
            except np.linalg.LinAlgError:
                bump = max(1e-12, (self.H_reg or 0.0) * 10 or 1e-12)
                H = H + bump * np.eye(self.n)
                self.H_reg = (self.H_reg or 0.0) + bump
                self.H_factor = cho_factor(H, lower=False)
            
            return self.H_factor, self.H_reg 
    
    def solve_H_system(self, rhs):
        """
        Solve H * sol = rhs using cached factorization.
        
        Parameters:
        -----------
        rhs : array-like, shape (n,)
            Right-hand side.
        
        Returns:
        --------
        sol : ndarray, shape (n,)
            Solution to H * sol = rhs.
        """
        rhs = np.asarray(rhs, dtype=float)
        
        if self.is_sparse:
            # Sparse solve
            return self.H_factor.solve(rhs)
        else:
            # Dense solve via Cholesky
            return cho_solve(self.H_factor, rhs)
    
    def schur_rhs(self, r_P, r_D, r_C):
        """
        Compute Schur system RHS for S Δy = b with S = E H^{-1} E^T:
            b = r_P - E H^{-1} (r_D + X^{-1} r_C)

        Parameters:
        -----------
        r_P : array-like, shape (|K|,)
            Primal residual (Ex - 1).
        r_D : array-like, shape (n,)
            Dual residual (Qx + q + E^T y - z). Can be None (treated as zeros).
        r_C : array-like, shape (n,)
            Complementarity residual (x ⊙ z - μ_target).
        
        Returns:
        --------
        b : ndarray, shape (|K|,)
            Schur system RHS.
        """
        x_safe = np.maximum(self.x, 1e-14)
        x_inv_r_C = r_C / x_safe
        
        if r_D is None:
            rhs_core = x_inv_r_C
        else:
            rhs_core = r_D + x_inv_r_C
        
        H_inv_rhs = self.solve_H_system(rhs_core)
        
        # b = r_P - E H^{-1} (r_D + X^{-1} r_C)
        b = r_P - apply_E(H_inv_rhs, self.blocks)
        return b
    
    def schur_operator(self, v):
        """
        Apply Schur operator: S v = E * H^{-1} * (E^T v).
        
        Parameters:
        -----------
        v : array-like, shape (|K|,)
            Vector to apply operator to.
        
        Returns:
        --------
        Sv : ndarray, shape (|K|,)
            Result of S * v.
        """
        v = np.asarray(v, dtype=float)
        
        # Step 1: E^T v
        Ety = apply_E_transpose(v, self.blocks, self.n)
        
        # Step 2: H^{-1} * (E^T v)
        H_inv_Ety = self.solve_H_system(Ety)
        
        # Step 3: E * H^{-1} * (E^T v)
        Sv = apply_E(H_inv_Ety, self.blocks)
        
        return Sv
    
    def schur_solve(self, b, assemble_if_small=True):
        """
        Solve Schur system: S * Δy = b.
        
        If |K| <= K_th, assemble S and factorize.
        Otherwise, use PCG on the operator.
        
        Parameters:
        -----------
        b : array-like, shape (|K|,)
            Schur system RHS.
        assemble_if_small : bool, default=True
            If True and |K| <= K_th, assemble full matrix.
        
        Returns:
        --------
        d_y : ndarray, shape (|K|,)
            Solution Δy.
        """
        b = np.asarray(b, dtype=float)
        
        if assemble_if_small and self.n_blocks <= self.cfg['K_th']:
            # Assemble full Schur matrix S
            S = np.zeros((self.n_blocks, self.n_blocks), dtype=float)
            for j in range(self.n_blocks):
                e_j = np.zeros(self.n_blocks, dtype=float)
                e_j[j] = 1.0
                S[:, j] = self.schur_operator(e_j)
            
            # Factorize and solve
            try:
                S_factor = cho_factor(S, lower=False)
                d_y = cho_solve(S_factor, b)
            except np.linalg.LinAlgError:
                # If Cholesky fails, use regularized version
                S_reg = S + 1e-12 * np.eye(self.n_blocks)
                S_factor = cho_factor(S_reg, lower=False)
                d_y = cho_solve(S_factor, b)
        else:
            # Use PCG on operator
            # Create linear operator
            def S_op(v):
                return self.schur_operator(v)
            
            # Simple diagonal preconditioner (diagonal of S)
            # For simplicity, use identity preconditioner
            # In practice, could compute diagonal of S
            M_precond = None  # Identity preconditioner
            
            # Solve with PCG
            d_y, info = scipy.sparse.linalg.cg(
                scipy.sparse.linalg.LinearOperator(
                    (self.n_blocks, self.n_blocks),
                    matvec=S_op,
                    dtype=float
                ),
                b,
                tol=self.cfg['pcg_tol'],
                maxiter=self.cfg['pcg_maxit'],
                M=M_precond
            )
            
            if info != 0:
                if self.cfg['verbosity'] >= 1:
                    print(f"Warning: PCG did not converge (info={info})")
        
        return d_y
    
    def back_substitute_dx_dz(self, r_C, d_y, r_D=None):
        """
        Back-substitute to get Δx and Δz.
        
        Δx = H^{-1} * (-X^{-1} r_C - r_D - E^T * Δy)
        Δz = Q * Δx + E^T * Δy + r_D
        
        In feasible-start, r_D should be zero, but we include it for robustness.
        
        Parameters:
        -----------
        r_C : array-like, shape (n,)
            Complementarity residual.
        d_y : array-like, shape (|K|,)
            Dual step direction.
        r_D : array-like, shape (n,), optional
            Dual residual (if None, assumed zero).
        
        Returns:
        --------
        d_x : ndarray, shape (n,)
            Primal step direction.
        d_z : ndarray, shape (n,)
            Dual slack step direction.
        """
        r_C = np.asarray(r_C, dtype=float)
        d_y = np.asarray(d_y, dtype=float)
        
        # Safe division
        x_safe = np.maximum(self.x, 1e-14)
        x_inv_r_C = r_C / x_safe
        
        # Step 1: E^T * Δy
        Ety_dy = apply_E_transpose(d_y, self.blocks, self.n)
        
        # Step 2: Δx = H^{-1} * (-X^{-1} r_C - r_D - E^T * Δy)
        rhs_dx = -x_inv_r_C - Ety_dy
        if r_D is not None:
            rhs_dx = rhs_dx - r_D
        d_x = self.solve_H_system(rhs_dx)
        
        # Step 3: Δz = Q * Δx + E^T * Δy + r_D
        if self.is_sparse:
            Q_dx = self.Q.dot(d_x)
        else:
            Q_dx = self.Q @ d_x
        d_z = Q_dx + Ety_dy
        if r_D is not None:
            d_z = d_z + r_D
        
        return d_x, d_z
    
    def update_and_mu(self, alpha, d_x, d_y, d_z):
        """
        Update iterates and compute new complementarity gap.
        
        x <- x + α * Δx
        y <- y + α * Δy
        z <- z + α * Δz
        μ <- (x^T z) / n
        
        Parameters:
        -----------
        alpha : float
            Step size.
        d_x : array-like, shape (n,)
            Primal step direction.
        d_y : array-like, shape (|K|,)
            Dual step direction.
        d_z : array-like, shape (n,)
            Dual slack step direction.
        """
        self.x = self.x + alpha * d_x
        self.y = self.y + alpha * d_y
        self.z = self.z + alpha * d_z
        self.mu = np.dot(self.x, self.z) / self.n
    
    def converged(self, r_D, r_P):
        """
        Check convergence.
        
        Converged if:
        - ||r_P||_∞ <= ε_feas
        - ||r_D||_∞ <= ε_feas
        - μ <= ε_comp
        
        Parameters:
        -----------
        r_D : array-like, shape (n,)
            Dual residual.
        r_P : array-like, shape (|K|,)
            Primal residual.
        
        Returns:
        --------
        is_converged : bool
            True if converged.
        """
        norm_r_P = norm_inf(r_P)
        norm_r_D = norm_inf(r_D)
        
        feas_ok = norm_r_P <= self.cfg['eps_feas'] and norm_r_D <= self.cfg['eps_feas']
        comp_ok = self.mu <= self.cfg['eps_comp']
        
        return feas_ok and comp_ok
    
    def iterate_once(self):
        """
        Perform one iteration of Algorithm 1.
        
        Returns:
        --------
        info : dict
            Dictionary with iteration information:
            - iter: iteration number
            - mu: complementarity gap
            - norm_r_P: ||r_P||_∞
            - norm_r_D: ||r_D||_∞
            - alpha_pri: primal step size
            - alpha_dual: dual step size
            - alpha: overall step size
            - converged: whether converged
        """
        # ==========================================
        # Step 2.1: Residual computation
        # ==========================================
        r_D, r_P, r_C, mu_target = self.compute_residuals()
        
        # Debug: check if we're maintaining feasibility
        if self.cfg['verbosity'] >= 3:
            print(f"    Residuals: ||r_P||={norm_inf(r_P):.6e}, ||r_D||={norm_inf(r_D):.6e}, ||r_C||={norm_inf(r_C):.6e}")
            print(f"    mu={self.mu:.6e}, mu_target={mu_target:.6e}")
        
        # ==========================================
        # Step 2.2: System assembly
        # ==========================================
        # Build and factorize H = Q + X^{-1}Z
        H_factor, H_reg = self.build_H_and_factorize()
        
        # Compute Schur system RHS: b = -E H^{-1} (r_D + X^{-1} r_C) (and r_P for robustness)
        b = self.schur_rhs(r_P, r_D, r_C)
        
        # ==========================================
        # Step 2.3: Direction computation
        # ==========================================
        # Solve S Δy = b
        d_y = self.schur_solve(b)
        
        # Back-substitute to get Δx and Δz
        d_x, d_z = self.back_substitute_dx_dz(r_C, d_y, r_D=r_D)
        
        # Debug: check Newton step
        if self.cfg['verbosity'] >= 3:
            from simplex_ipm.helper.block_ops import apply_E
            Edx = apply_E(d_x, self.blocks)
            print(f"    Newton step: ||d_x||={norm_inf(d_x):.6e}, ||d_y||={norm_inf(d_y):.6e}, ||d_z||={norm_inf(d_z):.6e}")
            print(f"    E*d_x (should be -r_P): {Edx}, -r_P: {-r_P}")
            if self.z is not None and d_z is not None:
                print(f"    min(z): {np.min(self.z):.6e}, min(d_z): {np.min(d_z):.6e}")
                print(f"    Components where d_z < 0: {np.sum(d_z < 0)}/{len(d_z)}")
        
        # ==========================================
        # Step 2.4: Step-size selection
        # ==========================================
        # Fraction-to-boundary rule
        alpha_pri, alpha_dual, alpha = fraction_to_boundary(
            self.x, d_x, self.z, d_z, gamma=self.cfg['gamma']
        )
        
        # ==========================================
        # Step 2.5: Update
        # ==========================================
        self.update_and_mu(alpha, d_x, d_y, d_z)
        
        # ==========================================
        # Step 2.6: Stopping criterion
        # ==========================================
        # Check convergence
        r_D_new, r_P_new, _, _ = self.compute_residuals()

        # KKT consistency checks (diagnostic at high verbosity)
        if self.cfg['verbosity'] >= 3:
            Ety_dy_chk = apply_E_transpose(d_y, self.blocks, self.n)
            eq1 = (self.Q @ d_x + Ety_dy_chk - d_z) + r_D  # ~0
            eq2 = apply_E(d_x, self.blocks) + r_P          # ~0
            eq3 = self.z * d_x + self.x * d_z + r_C        # ~0
            print(f"    Check eq1 (dual lin): {norm_inf(eq1):.3e}")
            print(f"    Check eq2 (primal  ): {norm_inf(eq2):.3e}")
            print(f"    Check eq3 (compl. ): {norm_inf(eq3):.3e}")
            assert norm_inf(eq2) <= 1e-8, "E dx != -r_P (check Schur RHS sign)"

        is_converged = self.converged(r_D_new, r_P_new)
        
        info = {
            'mu': self.mu,
            'norm_r_P': norm_inf(r_P_new),
            'norm_r_D': norm_inf(r_D_new),
            'alpha_pri': alpha_pri,
            'alpha_dual': alpha_dual,
            'alpha': alpha,
            'converged': is_converged,
            'H_reg': H_reg,
        }
        
        return info
    
    def solve(self):
        """
        Run Algorithm 1 until convergence or max iterations.
        
        Returns:
        --------
        result : dict
            Dictionary with solution and statistics:
            - x: primal solution
            - y: dual solution
            - z: dual slack
            - mu: final complementarity gap
            - iter: number of iterations
            - converged: whether converged
            - history: list of iteration info dicts
        """
        # Initialize
        self.initialize_delta_rule()
        
        if self.cfg['verbosity'] >= 1:
            print("Starting Algorithm 1 (Feasible-Start, Fixed sigma)")
            print(f"  n = {self.n}, |K| = {self.n_blocks}")
            print(f"  sigma = {self.cfg['sigma']}, max_iter = {self.cfg['max_iter']}")
            print(f"  Initial mu = {self.mu:.6e}")
            if self.cfg['verbosity'] >= 2:
                print("  Iter |     mu     | ||r_P||_inf | ||r_D||_inf | alpha_pri | alpha_dual")
                print("  " + "-" * 60)
        
        history = []
        
        for iter_num in range(1, self.cfg['max_iter'] + 1):
            info = self.iterate_once()
            info['iter'] = iter_num
            history.append(info)
            
            if self.cfg['verbosity'] >= 2:
                print(f"  {iter_num:4d} | {info['mu']:9.2e} | {info['norm_r_P']:8.2e} | "
                      f"{info['norm_r_D']:8.2e} | {info['alpha_pri']:5.3f} | {info['alpha_dual']:5.3f}")
            
            if info['converged']:
                if self.cfg['verbosity'] >= 1:
                    print(f"\nConverged in {iter_num} iterations")
                    print(f"  Final mu = {self.mu:.6e}")
                    print(f"  Final ||r_P||_inf = {info['norm_r_P']:.6e}")
                    print(f"  Final ||r_D||_inf = {info['norm_r_D']:.6e}")
                break
        
        if not history[-1]['converged']:
            if self.cfg['verbosity'] >= 1:
                print(f"\nDid not converge in {self.cfg['max_iter']} iterations")
                print(f"  Final mu = {self.mu:.6e}")
                print(f"  Final ||r_P||_inf = {history[-1]['norm_r_P']:.6e}")
                print(f"  Final ||r_D||_inf = {history[-1]['norm_r_D']:.6e}")
        
        result = {
            'x': self.x.copy() if self.x is not None else None,
            'y': self.y.copy() if self.y is not None else None,
            'z': self.z.copy() if self.z is not None else None,
            'mu': self.mu,
            'iter': len(history),
            'converged': history[-1]['converged'] if history else False,
            'history': history,
        }
        
        return result

