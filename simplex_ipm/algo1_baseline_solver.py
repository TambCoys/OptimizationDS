import numpy as np
from simplex_ipm.helper.utils import norm_inf


class BaselineIPM:
    """
    Algoritmo 1 (PDF p.12): Feasible-Start Primal–Dual Interior-Point Method (σ fisso)
    
    Risolve:  
        min ½xᵀQx + qᵀx  
        s.t.  ∑_{i∈Ik} x_i = 1 ∀k,  x ≥ 0
    """
    
    def __init__(self, Q, q, blocks, cfg=None):
        self.Q = Q
        self.q = q
        self.blocks = blocks
        
        # Configuration
        self.cfg = self.default_config()
        if cfg is not None:
            self.cfg.update(cfg)

    @staticmethod
    def default_config():
        """Return default configuration."""
        return {
            'eps_feas': 1e-8,
            'eps_comp': 1e-8,
            'sigma': 0.01,    # σ fisso, leggermente aggressivo
            'eps': 1e-10,     # ε per δ
            'tau': 1e-3,      # τ per δ
            'max_iter': 100000
        }

    def solve(self):
        """
        Esegue l'algoritmo.
        Restituisce: dict con chiavi 'x', 'y', 'z', 'converged', 'iter', 'mu', 'history'
        """
        Q = self.Q
        q = self.q
        blocks = self.blocks
        
        eps_feas = self.cfg['eps_feas']
        eps_comp = self.cfg['eps_comp']
        sigma = self.cfg['sigma']
        eps = self.cfg['eps']
        tau = self.cfg['tau']
        max_iter = self.cfg['max_iter']
        
        n = Q.shape[0]
        K = len(blocks)

        # --- Costruzione E ---
        E = np.zeros((K, n))
        for k, Ik in enumerate(blocks):
            E[k, Ik] = 1.0
        onesK = np.ones(K)
        onesn = np.ones(n)

        # =========================
        # STEP 1: Inizializzazione
        # =========================
        x = np.zeros(n)
        for Ik in blocks:
            x[Ik] = 1.0 / len(Ik)

        w = Q @ x + q

        # δ = max(ε, τ · max_k range(w_Ik))
        range_max = 0.0
        for Ik in blocks:
            wb = w[Ik]
            if wb.size > 1:
                range_max = max(range_max, float(wb.max() - wb.min()))
        delta = max(eps, tau * range_max)

        # y^0_k = -min(w_Ik) + δ
        y = np.zeros(K)
        for k, Ik in enumerate(blocks):
            y[k] = -float(w[Ik].min()) + delta

        z = w + E.T @ y
        assert np.all(x > 0) and np.all(z > 0), "Inizializzazione non strettamente interna!"

        mu = float(x @ z) / n
        info = {"iters": 0, "mu": [mu], "rP_inf": [], "rD_inf": [], "delta": delta}

        tiny = 1e-16

        # =========================
        # STEP 2: Iterazioni
        # =========================
        for it in range(1, max_iter + 1):
            info["iters"] = it

            # 2.1 Residui
            rD = Q @ x + q + E.T @ y - z
            rP = E @ x - onesK
            info["rP_inf"].append(norm_inf(rP))
            info["rD_inf"].append(norm_inf(rD))

            # --- Criterio di arresto migliorato ---
            xz_max = float(np.max(x * z))
            if (info["rP_inf"][-1] <= eps_feas and
                info["rD_inf"][-1] <= eps_feas and
                (mu <= eps_comp or xz_max <= eps_comp)):
                break

            # 2.2 H e μ_target
            H = Q + np.diag(z / x) + delta * np.eye(n)
            mu_target = sigma * mu
            rC = x * z - mu_target * onesn
            rhs = -rD - (rC / x)

            # 2.3 Schur complement
            H_inv_rhs = np.linalg.solve(H, rhs)
            H_inv_ET  = np.linalg.solve(H, E.T)
            S = E @ H_inv_ET
            b = E @ H_inv_rhs + rP

            # 2.4 Direzioni
            dy = np.linalg.solve(S, b)
            dx = np.linalg.solve(H, rhs - E.T @ dy)
            dz = rD + Q @ dx + E.T @ dy # <-- FIX aggiungi rD


            # 2.5 Step-size
            def frac_to_boundary(v, dv):
                mask = dv < 0
                if np.any(mask):
                    return min(1.0, 0.99 * float(np.min(-v[mask] / dv[mask])))
                return 1.0

            alpha = min(frac_to_boundary(x, dx), frac_to_boundary(z, dz))

            # 2.6 Aggiornamento
            x += alpha * dx
            y += alpha * dy
            z += alpha * dz
            x = np.maximum(x, tiny)
            z = np.maximum(z, tiny)

            # 2.7 μ attuale
            mu = float(x @ z) / n
            info["mu"].append(mu)

        else:
            print(f"Warning: raggiunto max_iter={max_iter} senza convergenza.")

        # Convert to standard dictionary format to match final solver
        converged = False
        if len(info["rP_inf"]) > 0 and len(info["rD_inf"]) > 0:
            converged = (info["rP_inf"][-1] <= eps_feas and 
                        info["rD_inf"][-1] <= eps_feas and 
                        (mu <= eps_comp or xz_max <= eps_comp))

        # Create history list of dicts to match final solver format
        history = []
        for i in range(len(info["rP_inf"])):
            history.append({
                'iter': i + 1,
                'mu': info["mu"][i],
                'norm_r_P': info["rP_inf"][i],
                'norm_r_D': info["rD_inf"][i]
            })

        result = {
            'x': x,
            'y': y,
            'z': z,
            'converged': converged,
            'iter': info['iters'],
            'mu': mu,
            'history': history,
            'info': info  # Keep original info for backward compatibility
        }

        return result
