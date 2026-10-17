# multinomial_group_enet_torch.py
#
# Multinomial logistic regression with a group elastic net penalty, solved with PyTorch.
# Same model as multinomial_group_lasso.py (the skglm version), but solved by accelerated
# proximal gradient (FISTA). Every step is a few large matrix products, so it uses all the
# CPU cores it is given, and runs on a GPU with device="cuda".
#
# The model, with gene coefficients W (n_genes x n_classes), an unpenalized intercept b
# (one per class) and Y one-hot (n_cells x n_classes):
#   minimize  (1/n) * sum_i w_i * [ logsumexp(x_i W + b) - y_i . (x_i W + b) ]
#             + alpha * sum_j ( l1_ratio * ||W[j, :]||_2 + (1 - l1_ratio) / 2 * ||W[j, :]||_2^2 )
# Each gene's coefficients across all classes form one group: the lasso part drops whole
# genes, the ridge part spreads weight across correlated genes. w_i are optional cell weights
# (default 1), e.g. inverse class frequencies so that rare classes count as much as common ones.
#
# Centering: X is used as if each gene were centred (X - mean), without densifying it.
# This only shifts the intercept, so the model is the same, but the problem is much better
# conditioned and FISTA converges faster. fit() returns the intercept for the uncentred X,
# so predictions are simply X @ W + b.

import numpy as np
import scipy.sparse as sp
import torch


def _to_torch_csr(X, device, dtype):
    """scipy sparse matrix -> torch sparse CSR tensor."""
    X = sp.csr_matrix(X)
    return torch.sparse_csr_tensor(
        torch.from_numpy(X.indptr.astype(np.int64)),
        torch.from_numpy(X.indices.astype(np.int64)),
        torch.from_numpy(X.data),
        size=X.shape, dtype=dtype, device=device)


class MultinomialGroupEnet:
    """Holds one training matrix on the chosen device, and fits the model at any alpha / l1_ratio.

    X: cells x genes (scipy sparse or numpy), already scaled. Do NOT add an intercept column:
       the intercept is handled separately.
    Y: one-hot labels, cells x classes (numpy).
    device: "cpu" or "cuda". dtype: torch.float64 (default, exact) or torch.float32 (faster).
    dense: keep X as a dense matrix (faster matrix products if it fits in memory).
    sample_weight: optional weight per cell (numpy, length n). Default: every cell weight 1.
    """

    def __init__(self, X, Y, device="cpu", dtype=torch.float64, dense=False, sample_weight=None):
        self.device, self.dtype = device, dtype
        self.n, self.p = X.shape
        self.K = Y.shape[1]
        if sp.issparse(X):
            mu = np.asarray(X.mean(axis=0)).ravel()
            if dense:
                self.X = torch.as_tensor(X.toarray(), dtype=dtype, device=device)
                self.Xt = None
            else:
                self.X = _to_torch_csr(X, device, dtype)
                self.Xt = _to_torch_csr(X.T, device, dtype)    # X transpose, for X^T R
        else:
            mu = X.mean(axis=0)
            self.X = torch.as_tensor(X, dtype=dtype, device=device)
            self.Xt = None
        self.mu = torch.as_tensor(mu, dtype=dtype, device=device)         # gene means
        self.Y = torch.as_tensor(Y, dtype=dtype, device=device)
        w = np.ones(self.n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self.w = torch.as_tensor(w, dtype=dtype, device=device).unsqueeze(1)     # n x 1
        self.L = self._lipschitz()

    # ---- products with the (implicitly) centred X
    def _Xc_mm(self, W):
        """(X - 1 mu^T) @ W"""
        XW = torch.sparse.mm(self.X, W) if self.X.is_sparse_csr else self.X @ W
        return XW - self.mu @ W

    def _Xct_mm(self, R):
        """(X - 1 mu^T)^T @ R"""
        if self.Xt is not None:
            XtR = torch.sparse.mm(self.Xt, R)
        else:
            XtR = self.X.T @ R
        return XtR - torch.outer(self.mu, R.sum(dim=0))

    def _lipschitz(self, n_iter=30):
        """Step-size bound: 0.5 * ||D^1/2 Xc||_2^2 / n, with D the cell weights (largest singular
        value by power iteration), and 0.5 * mean(w) for the intercept part. 0.5 bounds the
        curvature of the softmax loss."""
        v = torch.randn(self.p, 1, dtype=self.dtype, device=self.device,
                        generator=torch.Generator(device=self.device).manual_seed(0))
        for _ in range(n_iter):
            v = self._Xct_mm(self.w * self._Xc_mm(v))
            v = v / torch.linalg.norm(v)
        smax2 = (self.w * self._Xc_mm(v) ** 2).sum()
        # the intercept column has sum(w) / n = mean(w)
        return 0.5 * max(smax2.item() / self.n, self.w.mean().item())

    # ---- loss pieces
    def _grad(self, W, b):
        """Gradient of the mean cross-entropy w.r.t. W and b, and the loss value."""
        S = self._Xc_mm(W) + b
        logZ = torch.logsumexp(S, dim=1, keepdim=True)
        P = torch.exp(S - logZ)
        R = self.w * (P - self.Y) / self.n
        loss = (self.w.squeeze(1) * (logZ.squeeze(1) - (self.Y * S).sum(dim=1))).mean()
        return self._Xct_mm(R), R.sum(dim=0), loss

    def _prox(self, V, step, l1, l2):
        """Group elastic net prox, row by row: block soft-threshold, then ridge shrink."""
        norms = torch.linalg.norm(V, dim=1, keepdim=True)
        shrink = torch.clamp(1 - step * l1 / torch.clamp(norms, min=1e-300), min=0.0)
        return V * shrink / (1 + step * l2)

    def _kkt(self, W, b, gW, gb, l1, l2):
        """Largest violation of the optimality conditions (same measure as skglm's subdiff_distance)."""
        G = gW + l2 * W
        norms_W = torch.linalg.norm(W, dim=1)
        norms_G = torch.linalg.norm(G, dim=1)
        zero = norms_W == 0
        viol = torch.where(
            zero,
            torch.clamp(norms_G - l1, min=0.0),
            torch.linalg.norm(G + l1 * W / torch.clamp(norms_W, min=1e-300).unsqueeze(1), dim=1))
        return max(viol.max().item(), torch.linalg.norm(gb).item())

    # ---- fitting
    def fit(self, alpha, l1_ratio, W0=None, b0=None, tol=1e-6, max_iter=20000, check_every=10):
        """Fit at one (alpha, l1_ratio), optionally warm-started from (W0, b0) as returned by fit().

        Returns W (genes x classes, numpy), b (intercept for the uncentred X, numpy) and a dict
        with the number of iterations, the final KKT violation and whether it converged.
        """
        l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
        step = 1.0 / self.L
        W = torch.zeros(self.p, self.K, dtype=self.dtype, device=self.device) if W0 is None \
            else torch.as_tensor(W0, dtype=self.dtype, device=self.device).clone()
        # internally the intercept belongs to the centred X: b_centred = b + mu @ W
        b = torch.zeros(self.K, dtype=self.dtype, device=self.device) if b0 is None \
            else torch.as_tensor(b0, dtype=self.dtype, device=self.device) + self.mu @ W
        Wy, by, t = W.clone(), b.clone(), 1.0
        kkt, it = np.inf, 0
        for it in range(1, max_iter + 1):
            gW, gb, _ = self._grad(Wy, by)
            W_new = self._prox(Wy - step * gW, step, l1, l2)
            b_new = by - step * gb
            # adaptive restart: drop the momentum if it points uphill
            if ((Wy - W_new) * (W_new - W)).sum() + ((by - b_new) * (b_new - b)).sum() > 0:
                t = 1.0
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            beta = (t - 1) / t_new
            Wy = W_new + beta * (W_new - W)
            by = b_new + beta * (b_new - b)
            W, b, t = W_new, b_new, t_new
            if it % check_every == 0:
                gW_, gb_, _ = self._grad(W, b)
                kkt = self._kkt(W, b, gW_, gb_, l1, l2)
                if kkt <= tol:
                    break
        b_uncentred = b - self.mu @ W
        return (W.cpu().numpy(), b_uncentred.cpu().numpy(),
                dict(n_iter=it, kkt=kkt, converged=kkt <= tol))

    def alpha_max(self, l1_ratio):
        """Smallest alpha at which every gene is dropped (intercept-only model)."""
        W = torch.zeros(self.p, self.K, dtype=self.dtype, device=self.device)
        # the best intercept-only model predicts the (weighted) class frequencies
        b = torch.log(((self.w * self.Y).sum(dim=0) / self.w.sum()).clamp(min=1e-12))
        gW, _, _ = self._grad(W, b)
        return torch.linalg.norm(gW, dim=1).max().item() / l1_ratio


def predict_proba(X, W, b):
    """Class probabilities for cells X (scipy sparse or numpy, scaled like the training data)."""
    S = X @ W + b
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(axis=1, keepdims=True)
