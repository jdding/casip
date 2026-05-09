import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class MLPAttr(nn.Module):
    def __init__(
        self,
        attr_dims: list,
        embed_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        for n_vals in attr_dims:
            self.embeddings.append(nn.Embedding(n_vals, embed_dim))
        self.fc = nn.Sequential(
            nn.Linear(len(attr_dims) * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        batch_size, n_items, n_attrs = attrs.shape
        embeds = []
        for i in range(n_attrs):
            emb = self.embeddings[i](attrs[:, :, i])
            embeds.append(emb)
        x = torch.cat(embeds, dim=-1)
        return self.fc(x).squeeze(-1)


class AGKNet(nn.Module):
    def __init__(
        self,
        item_embedding_dim: int = 1024,
        hidden_dim: int = 64,
        base_sigma: float = 0.25,
        target_sigma: float = 0.25,
        lambda_reg: float = 20.0,
        lambda_fixed: float = None,
        n_attr_types: int = 4,
        llm_prior_matrix: torch.Tensor = None
    ):
        super().__init__()
        
        self.item_embedding_dim = item_embedding_dim
        
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        self.temp_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.attr_mlp = MLPAttr([46, 30, 50, 698], embed_dim=16, hidden_dim=64)
        
        init_val = inverse_softplus(torch.tensor(base_sigma))
        self.sigma_mlp[2].bias.data.fill_(init_val)
        
        self.target_sigma = target_sigma
        self.lambda_reg = lambda_reg
        self.lambda_fixed = lambda_fixed
        
        if llm_prior_matrix is not None:
            self.register_buffer('llm_prior', llm_prior_matrix)
        else:
            self.llm_prior = None
    
    def get_sigma_regularization_loss(self) -> torch.Tensor:
        if self.lambda_fixed is not None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        mean_sigma = self._get_mean_sigma()
        loss = (mean_sigma - self.target_sigma) ** 2
        return self.lambda_reg * loss
    
    def _get_mean_sigma(self) -> torch.Tensor:
        if self.lambda_fixed is not None:
            return torch.tensor(self.lambda_fixed, device=next(self.parameters()).device)
        return F.softplus(self.sigma_mlp[2].bias[0])
    
    def forward(
        self,
        history_items: torch.Tensor,
        history_attrs: torch.Tensor,
        history_mask: torch.Tensor,
        chunk_emb: torch.Tensor,
        chunk_attrs: torch.Tensor,
        gap_days: torch.Tensor,
        chunk_item_indices: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = history_items.shape[0]
        n_chunk = chunk_emb.shape[0]
        max_len = history_items.shape[1]
        
        if self.llm_prior is not None:
            history_embs = self.llm_prior[history_items]
        else:
            history_embs = chunk_emb[history_items]
        
        gap_days_flat = gap_days.view(batch_size, 1, 1)
        
        if self.lambda_fixed is not None:
            sigma = torch.full(
                (batch_size, 1, 1),
                float(self.lambda_fixed),
                device=gap_days.device,
                dtype=chunk_emb.dtype
            )
        else:
            sigma = self.sigma_mlp(gap_days_flat)
        
        history_embs = F.normalize(history_embs, p=2, dim=2)
        chunk_emb = F.normalize(chunk_emb, p=2, dim=1)
        dist = 1.0 - torch.matmul(history_embs, chunk_emb.T)

        kernel_vals = torch.exp(-dist / (2 * sigma ** 2))
        
        history_mask_exp = history_mask.unsqueeze(-1)
        kernel_vals = kernel_vals.masked_fill(history_mask_exp == 0, 0.0)
        weights = kernel_vals.sum(dim=1)
        
        return weights
    
    def forward_pos_neg(
        self,
        history_items: torch.Tensor,
        history_attrs: torch.Tensor,
        history_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        pos_attr: torch.Tensor,
        neg_embs: torch.Tensor,
        neg_attrs: torch.Tensor,
        gap_days: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = history_items.shape[0]
        candidate_embs = torch.cat([pos_emb.unsqueeze(1), neg_embs], dim=1)

        if self.llm_prior is not None:
            history_embs = self.llm_prior[history_items]
        else:
            history_embs = candidate_embs[:, 0:1, :].expand(batch_size, history_items.shape[1], -1)

        gap_days_flat = gap_days.view(batch_size, 1, 1)
        if self.lambda_fixed is not None:
            sigma = torch.full(
                (batch_size, 1, 1),
                float(self.lambda_fixed),
                device=gap_days.device,
                dtype=candidate_embs.dtype
            )
        else:
            sigma = self.sigma_mlp(gap_days_flat)

        history_embs = F.normalize(history_embs, p=2, dim=2)
        candidate_embs = F.normalize(candidate_embs, p=2, dim=2)
        dist = 1.0 - torch.matmul(history_embs, candidate_embs.transpose(1, 2))
        kernel_vals = torch.exp(-dist / (2 * sigma ** 2))
        kernel_vals = kernel_vals.masked_fill(history_mask.unsqueeze(-1) == 0, 0.0)
        scores = kernel_vals.sum(dim=1)

        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        return pos_scores, neg_scores
