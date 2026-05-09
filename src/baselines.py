import torch
import torch.nn as nn
import torch.nn.functional as F


class MostPopularBaseline(nn.Module):
    def __init__(self, popularity_scores: torch.Tensor, alpha: float = 1.0):
        super().__init__()
        self.register_buffer('popularity', popularity_scores)
        self.alpha = alpha

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
        if chunk_item_indices is None:
            chunk_item_indices = torch.arange(n_chunk, device=chunk_emb.device)
        return self.popularity[chunk_item_indices].unsqueeze(0).expand(batch_size, -1) * self.alpha


class MeanPoolBaseline(nn.Module):
    def __init__(
        self,
        item_embeddings: torch.Tensor,
        popularity_scores: torch.Tensor,
        alpha: float = 0.6
    ):
        super().__init__()
        self.register_buffer('item_embeddings', item_embeddings)
        self.register_buffer('popularity', popularity_scores)
        self.alpha = alpha

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
        batch_size, max_len = history_items.shape
        item_embs = self.item_embeddings[history_items]

        mask = history_mask.unsqueeze(-1)
        sum_embs = (item_embs * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        user_emb = sum_embs / count

        # Normalize user embedding
        user_emb_norm = torch.norm(user_emb, p=2, dim=1, keepdim=True).clamp(min=1e-10)
        user_emb = user_emb / user_emb_norm

        semantic_scores = user_emb @ chunk_emb.T
        if chunk_item_indices is None:
            chunk_item_indices = torch.arange(chunk_emb.shape[0], device=chunk_emb.device)
        pop_scores = self.popularity[chunk_item_indices].unsqueeze(0).expand_as(semantic_scores)

        # Weighted average: alpha * semantic + (1 - alpha) * pop
        return self.alpha * semantic_scores + (1 - self.alpha) * pop_scores


class KDEBaseline(nn.Module):
    def __init__(
        self,
        item_embeddings: torch.Tensor,
        popularity_scores: torch.Tensor,
        sigma: float = 0.25,
        alpha: float = 0.5
    ):
        super().__init__()
        self.register_buffer('item_embeddings', item_embeddings)
        self.register_buffer('popularity', popularity_scores)
        self.sigma = sigma
        self.alpha = alpha

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
        history_embs = self.item_embeddings[history_items]
        history_embs = F.normalize(history_embs, p=2, dim=2)
        chunk_emb = F.normalize(chunk_emb, p=2, dim=1)

        sim = torch.matmul(history_embs, chunk_emb.T)
        dist = 1.0 - sim
        kernel_vals = torch.exp(-dist / (2 * self.sigma ** 2))

        mask = history_mask.unsqueeze(-1)
        valid_counts = mask.sum(dim=1).clamp(min=1)
        scores = (kernel_vals * mask).sum(dim=1) / valid_counts

        if chunk_item_indices is None:
            chunk_item_indices = torch.arange(chunk_emb.shape[0], device=chunk_emb.device)
        pop_scores = self.popularity[chunk_item_indices].unsqueeze(0).expand_as(scores)

        # Weighted average: alpha * kde + (1 - alpha) * pop
        return self.alpha * scores + (1 - self.alpha) * pop_scores
