"""Logistic head that turns a CSD style embedding into p(c = Van Gogh | x).

Fit on the calibration set (Van-Gogh positives vs other-painter/plain negatives)
in CSD-768 space by full-batch L-BFGS — deterministic, embeddings are small.
"""

import torch
import torch.nn.functional as F


class LogisticHead:
    """L2-regularized logistic regression p(c|e) = sigmoid(w^T e + b) on embeddings."""

    def __init__(self, dim: int, l2: float = 1e-2, device: str = "cpu"):
        self.dim = dim
        self.l2 = l2
        self.device = device
        self.w = torch.zeros(dim, device=device, dtype=torch.float64)
        self.b = torch.zeros((), device=device, dtype=torch.float64)
        self.fitted = False

    def fit(self, emb_pos: torch.Tensor, emb_neg: torch.Tensor, max_iter: int = 200) -> "LogisticHead":
        """Fit on positive (c=1) and negative (c=0) embeddings."""
        Xp = emb_pos.to(self.device, torch.float64)
        Xn = emb_neg.to(self.device, torch.float64)
        X = torch.cat([Xp, Xn], dim=0)
        y = torch.cat([torch.ones(Xp.shape[0], device=self.device, dtype=torch.float64), torch.zeros(Xn.shape[0], device=self.device, dtype=torch.float64)])
        w = torch.zeros(self.dim, device=self.device, dtype=torch.float64, requires_grad=True)
        b = torch.zeros((), device=self.device, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(X @ w + b, y) + self.l2 * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.w = w.detach()
        self.b = b.detach()
        self.fitted = True
        return self

    @torch.no_grad()
    def predict_proba(self, emb: torch.Tensor) -> torch.Tensor:
        """Embeddings (n,d) -> p(c|x) in (0,1), shape (n,)."""
        if not self.fitted:
            raise RuntimeError("LogisticHead.predict_proba before fit()")
        e = emb.to(self.device, torch.float64)
        return torch.sigmoid(e @ self.w + self.b)
