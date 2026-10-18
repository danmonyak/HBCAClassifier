# multinomial_group_lasso.py
#
# Multinomial (softmax) logistic regression with a shared group penalty, for skglm.
# Not used by the classifier scripts, which use the faster PyTorch version
# (hbca_classifier/group_enet.py). Kept to cross-check that solver.
#
# skglm (0.5) has a multi-task solver (MultiTaskBCD) and a row-wise group penalty (L2_1),
# but no multinomial loss. This module adds:
#   - MultinomialLogistic: the softmax cross-entropy loss, written as a skglm multi-task
#     datafit (same methods as skglm's QuadraticMultiTask), for dense or sparse (CSC) X.
#   - WeightedGroupElasticNet: a group lasso + ridge penalty on each feature's coefficients
#     across classes, with one weight per feature. Weight 0 leaves a feature unpenalized
#     (used for the intercept column).
#
# The model, with W of shape (n_features, n_classes) and Y one-hot (n_samples, n_classes):
#   minimize  (1/n) * sum_i [ logsumexp(x_i W) - y_i . (x_i W) ]
#             + alpha * sum_j w_j * ( l1_ratio * ||W[j, :]||_2 + (1 - l1_ratio) / 2 * ||W[j, :]||_2^2 )
# Each feature's coefficients across all classes form one group, so a gene is either
# used for every class or dropped for every class. This is the grouped multinomial
# elastic net (glmnet's type.multinomial = "grouped").

import numpy as np
from numba import float64
from numpy.linalg import norm

from skglm.datafits.base import BaseMultitaskDatafit
from skglm.penalties.base import BasePenalty
from skglm.utils.prox_funcs import BST


class MultinomialLogistic(BaseMultitaskDatafit):
    """Multinomial logistic (softmax cross-entropy) datafit, averaged over samples.

    Y is one-hot, shape (n_samples, n_classes). XW holds the class scores X @ W.
    """

    def __init__(self):
        pass

    def get_spec(self):
        return ()

    def params_to_dict(self):
        return dict()

    # ---------------- helpers
    def _softmax_minus_y(self, Y, XW, i):
        """softmax(XW[i]) - Y[i] for one sample: the gradient of its loss w.r.t. its scores."""
        row = XW[i, :]
        m = np.max(row)
        e = np.exp(row - m)
        return e / np.sum(e) - Y[i, :]

    def _residual(self, Y, XW):
        """softmax(XW) - Y for all samples, shape (n_samples, n_classes)."""
        out = np.empty_like(XW)
        for i in range(XW.shape[0]):
            out[i, :] = self._softmax_minus_y(Y, XW, i)
        return out

    # ---------------- step sizes
    # The Hessian of the loss w.r.t. one feature's coefficients is bounded by
    # ||X[:, j]||^2 / n times the largest eigenvalue of diag(p) - p p^T, which is at most 1/2.
    def get_lipschitz(self, X, Y):
        n_samples, n_features = X.shape
        lipschitz = np.zeros(n_features)
        for j in range(n_features):
            lipschitz[j] = 0.5 * norm(X[:, j]) ** 2 / n_samples
        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, Y):
        n_samples = Y.shape[0]
        n_features = len(X_indptr) - 1
        lipschitz = np.zeros(n_features)
        for j in range(n_features):
            nrm2 = 0.
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
            lipschitz[j] = 0.5 * nrm2 / n_samples
        return lipschitz

    def initialize(self, X, Y):
        pass

    def initialize_sparse(self, X_data, X_indptr, X_indices, Y):
        pass

    # ---------------- loss and gradients
    def value(self, Y, W, XW):
        """Mean cross-entropy at scores XW."""
        n_samples = Y.shape[0]
        total = 0.
        for i in range(n_samples):
            row = XW[i, :]
            m = np.max(row)
            total += m + np.log(np.sum(np.exp(row - m))) - np.sum(Y[i, :] * row)
        return total / n_samples

    def gradient_j(self, X, Y, W, XW, j):
        """Gradient w.r.t. W[j, :] (one value per class)."""
        n_samples = X.shape[0]
        return X[:, j] @ self._residual(Y, XW) / n_samples

    def gradient_j_sparse(self, X_data, X_indptr, X_indices, Y, XW, j):
        """Gradient w.r.t. W[j, :] when X is sparse (CSC): only rows where X[:, j] != 0."""
        n_samples, n_classes = Y.shape
        grad = np.zeros(n_classes)
        for idx in range(X_indptr[j], X_indptr[j + 1]):
            i = X_indices[idx]
            grad += X_data[idx] * self._softmax_minus_y(Y, XW, i)
        return grad / n_samples

    def full_grad_sparse(self, X_data, X_indptr, X_indices, Y, XW):
        """Gradient w.r.t. all of W when X is sparse (CSC)."""
        n_samples, n_classes = Y.shape
        n_features = X_indptr.shape[0] - 1
        R = self._residual(Y, XW)
        grad = np.zeros((n_features, n_classes))
        for j in range(n_features):
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                grad[j, :] += X_data[idx] * R[X_indices[idx], :]
        return grad / n_samples

    def intercept_update_step(self, Y, XW):
        """Only used if the solver fits its own intercept (we add an intercept column instead)."""
        return 0.5 * np.sum(self._residual(Y, XW), axis=0) / Y.shape[0]


class WeightedGroupElasticNet(BasePenalty):
    """Group elastic net on rows of W (a gene's coefficients across all classes):

        alpha * sum_j weights[j] * ( l1_ratio * ||W[j, :]||_2 + (1 - l1_ratio) / 2 * ||W[j, :]||_2^2 )

    The first part (group lasso) sets whole rows to 0, so a gene is dropped for every class.
    The second part (ridge) shrinks rows smoothly and spreads weight across correlated genes.
    l1_ratio = 1 is the pure group lasso, l1_ratio = 0 is pure ridge.
    A weight of 0 leaves that feature (row) unpenalized (used for the intercept).
    """

    def __init__(self, alpha, l1_ratio, weights):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.weights = weights.astype(np.float64)

    def get_spec(self):
        return (
            ('alpha', float64),
            ('l1_ratio', float64),
            ('weights', float64[:]),
        )

    def params_to_dict(self):
        return dict(alpha=self.alpha, l1_ratio=self.l1_ratio, weights=self.weights)

    def value(self, W):
        row_norms = np.sqrt(np.sum(W ** 2, axis=1))
        return self.alpha * np.sum(self.weights * (
            self.l1_ratio * row_norms + 0.5 * (1 - self.l1_ratio) * row_norms ** 2))

    def prox_1feat(self, value, stepsize, j):
        """Prox of one row: block soft-thresholding (lasso part), then shrinking (ridge part)."""
        l1 = self.alpha * self.l1_ratio * self.weights[j]
        l2 = self.alpha * (1 - self.l1_ratio) * self.weights[j]
        return BST(value, l1 * stepsize) / (1. + l2 * stepsize)

    def subdiff_distance(self, W, grad, ws):
        """Distance of -grad to the subdifferential, per feature in ws (used to check convergence)."""
        subdiff_dist = np.zeros_like(ws, dtype=grad.dtype)
        for idx, j in enumerate(ws):
            l1 = self.alpha * self.l1_ratio * self.weights[j]
            l2 = self.alpha * (1 - self.l1_ratio) * self.weights[j]
            # the ridge part is smooth, so add its gradient to the datafit gradient
            g = grad[idx, :] + l2 * W[j, :]
            if self.weights[j] == 0.:
                subdiff_dist[idx] = norm(g)
            elif not np.any(W[j, :]):
                subdiff_dist[idx] = max(0, norm(g) - l1)
            else:
                subdiff_dist[idx] = norm(g + l1 * W[j, :] / norm(W[j, :]))
        return subdiff_dist

    def is_penalized(self, n_features):
        return self.weights != 0
