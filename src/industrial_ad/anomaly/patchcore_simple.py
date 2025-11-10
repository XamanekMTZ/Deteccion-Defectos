from __future__ import annotations
import torch, math
from typing import Tuple, Dict

class MemoryBank:
    """
    Banco de parches 'normales' (train) para k-NN.
    Implementación deliberadamente simple: concatena features y hace
    kNN con producto punto (cosine) o L2.
    """
    def __init__(self, max_patches: int = 20000, normalize: bool = True, metric: str = "cosine"):
        self.max_patches = max_patches
        self.normalize = normalize
        self.metric = metric
        self._bank = None  # (M, D)

    def build(self, feats_list: list[torch.Tensor]):
        """
        feats_list: lista de tensores (N_i, D), CPU o CUDA.
        """
        with torch.no_grad():
            bank = torch.cat(feats_list, dim=0)
            if self.normalize:
                bank = torch.nn.functional.normalize(bank, dim=1)
            # subsample aleatorio si excede
            if bank.shape[0] > self.max_patches:
                idx = torch.randperm(bank.shape[0])[: self.max_patches]
                bank = bank[idx]
            self._bank = bank.float().cpu()  # en CPU para ahorrar GPU en inferencia batch=1

    @torch.no_grad()
    def knn_score(self, query: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        query: (N, D)
        retorna: distancias min-k (N,)
        """
        assert self._bank is not None, "MemoryBank vacío. Llama a build() antes."
        Q = query.float().cpu()
        B = self._bank  # (M, D)
        if self.normalize:
            Q = torch.nn.functional.normalize(Q, dim=1)
            # para coseno, la 'distancia' = 1 - sim
            sims = Q @ B.t()  # (N, M)
            vals, _ = torch.topk(sims, k=k, dim=1, largest=True)
            # usamos 1 - promedio top-k como 'anomaly score'
            d = 1.0 - vals.mean(dim=1)
            return d
        else:
            # L2
            # ||q-b||^2 = ||q||^2 + ||b||^2 - 2 q·b
            q2 = (Q**2).sum(dim=1, keepdim=True)
            b2 = (B**2).sum(dim=1).unsqueeze(0)
            sims = q2 + b2 - 2 * (Q @ B.t())
            vals, _ = torch.topk(-sims, k=k, dim=1, largest=True)
            d = (-vals).mean(dim=1).clamp_min(0).sqrt()
            return d

def aggregate_patch_scores(patch_scores: torch.Tensor, grid: Tuple[int,int], out_hw: Tuple[int,int]) -> torch.Tensor:
    """
    patch_scores: (N,) corresponden a (Ph*Pw)
    grid: (Ph, Pw). out_hw: (H, W) deseado (tamaño imagen para heatmap).
    Retorna mapa 2D (H, W) reescalado con bilinear.
    """
    Ph, Pw = grid
    m = patch_scores.reshape(1,1,Ph,Pw)
    m_up = torch.nn.functional.interpolate(m, size=out_hw, mode="bilinear", align_corners=False)
    return m_up[0,0]
