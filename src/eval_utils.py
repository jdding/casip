import torch
import numpy as np
from typing import Tuple, List, Dict


def compute_hit_at_k(
    scores: torch.Tensor,
    true_idx: torch.Tensor,
    k: int = 10
) -> Tuple[float, int]:
    sorted_indices = torch.argsort(scores, descending=True)
    hit = (sorted_indices[:k] == true_idx).any().item()
    return hit, 1


def compute_ndcg_at_k(
    scores: torch.Tensor,
    true_idx: torch.Tensor,
    k: int = 10
) -> Tuple[float, int]:
    sorted_indices = torch.argsort(scores, descending=True)
    rank = (sorted_indices == true_idx).nonzero()
    if len(rank) == 0:
        return 0.0, 1
    rank = rank[0][0].item()
    if rank >= k:
        return 0.0, 1
    ndcg = float(1.0 / np.log2(rank + 2))
    return ndcg, 1


def evaluate_full_ranking_chunked(
    model,
    eval_pairs: List[Dict],
    catalog_embs: torch.Tensor,
    catalog_attrs: torch.Tensor,
    article_to_emb_idx: Dict[int, int],
    device: torch.device,
    chunk_size: int = 50000,
    k: int = 10,
    verbose: bool = False,
    catalog_article_ids: List[int] = None,
    catalog_item_indices: torch.Tensor = None,
    eval_batch_size: int = 16
) -> Dict[str, float]:
    model.eval()
    total_hit = 0
    total_ndcg = 0
    processed_n = 0
    N_catalog = catalog_embs.shape[0]
    if catalog_article_ids is not None:
        candidate_article_to_local = {int(aid): i for i, aid in enumerate(catalog_article_ids)}
        total_n = sum(1 for p in eval_pairs if int(p['target_article_id']) in candidate_article_to_local)
    else:
        candidate_article_to_local = None
        total_n = sum(1 for p in eval_pairs if int(p['target_article_id']) in article_to_emb_idx)
    if catalog_item_indices is None:
        catalog_item_indices = torch.arange(N_catalog, dtype=torch.long)
    
    if verbose:
        print(f"Evaluating {total_n} users on {N_catalog} items in chunks of {chunk_size}...", flush=True)
    
    valid_pairs = []
    true_indices = []
    for pair in eval_pairs:
        target_article_id = int(pair['target_article_id'])
        if candidate_article_to_local is not None:
            if target_article_id not in candidate_article_to_local:
                continue
            true_idx = candidate_article_to_local[target_article_id]
        elif target_article_id in article_to_emb_idx:
            true_idx = article_to_emb_idx[target_article_id]
        else:
            continue
        valid_pairs.append(pair)
        true_indices.append(true_idx)

    next_report = 100
    k_eff = min(k, N_catalog)

    with torch.no_grad():
        for batch_start in range(0, len(valid_pairs), eval_batch_size):
            batch_pairs = valid_pairs[batch_start:batch_start + eval_batch_size]
            batch_true = torch.tensor(
                true_indices[batch_start:batch_start + eval_batch_size],
                device=device,
                dtype=torch.long
            )
            history_items = torch.stack([p['history_items'] for p in batch_pairs]).to(device)
            history_attrs = torch.stack([p['history_attrs'] for p in batch_pairs]).to(device)
            history_mask = torch.stack([p['history_mask'] for p in batch_pairs]).to(device)
            gap_days = torch.stack([p['gap_days'] for p in batch_pairs]).to(device)

            top_values = None
            top_indices = None
            for start in range(0, N_catalog, chunk_size):
                end = min(start + chunk_size, N_catalog)
                chunk_emb = catalog_embs[start:end].to(device)
                chunk_attr = catalog_attrs[start:end].to(device)
                chunk_indices = catalog_item_indices[start:end].to(device)
                
                scores = model(
                    history_items,
                    history_attrs,
                    history_mask,
                    chunk_emb,
                    chunk_attr,
                    gap_days,
                    chunk_indices
                )
                local_k = min(k_eff, scores.shape[1])
                values, indices = torch.topk(scores, k=local_k, dim=1)
                indices = indices + start

                if top_values is None:
                    top_values = values
                    top_indices = indices
                else:
                    merged_values = torch.cat([top_values, values], dim=1)
                    merged_indices = torch.cat([top_indices, indices], dim=1)
                    keep_k = min(k_eff, merged_values.shape[1])
                    top_values, keep = torch.topk(merged_values, k=keep_k, dim=1)
                    top_indices = torch.gather(merged_indices, dim=1, index=keep)

            matches = top_indices == batch_true.unsqueeze(1)
            hits = matches.any(dim=1)
            discounts = 1.0 / torch.log2(
                torch.arange(k_eff, device=device, dtype=torch.float32) + 2.0
            )
            ndcgs = (matches.float() * discounts.unsqueeze(0)).sum(dim=1)

            total_hit += int(hits.sum().item())
            total_ndcg += float(ndcgs.sum().item())
            processed_n += len(batch_pairs)

            if verbose and processed_n >= next_report:
                print(
                    f"  Processed {processed_n}/{total_n}, current Hit@{k}: {total_hit/max(processed_n, 1):.4f}",
                    flush=True
                )
                next_report += 100

    hit_mean = total_hit / max(processed_n, 1)
    ndcg_mean = total_ndcg / max(processed_n, 1)

    return {
        f'hit@{k}': hit_mean,
        f'ndcg@{k}': ndcg_mean,
        'n_samples': processed_n
    }


class IDBaselineDummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int, max_history: int = 20):
        self.n_samples = n_samples
        self.max_history = max_history
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'history_items': torch.zeros(self.max_history, dtype=torch.int64),
            'history_attrs': torch.zeros((self.max_history, 4), dtype=torch.int64),
            'history_mask': torch.zeros(self.max_history, dtype=torch.float32),
            'target_item': torch.tensor(0, dtype=torch.int64),
            'negatives': torch.zeros(63, dtype=torch.int64),
            'gap_days': torch.tensor(365.0, dtype=torch.float32)
        }


def compute_popularity_scores(
    transactions_df,
    article_to_idx: Dict[int, int]
) -> torch.Tensor:
    popularity = transactions_df['article_id'].value_counts()
    n_items = len(article_to_idx)
    scores = torch.zeros(n_items)

    for article_id, count in popularity.items():
        if article_id in article_to_idx:
            idx = article_to_idx[article_id]
            scores[idx] = count

    max_count = scores.max().clamp(min=1)
    return scores / max_count


def compute_popularity_scores_chunked(
    transactions_path: str,
    article_to_idx: Dict[int, int],
    chunksize: int = 1_000_000,
    start_date: str = None,
    end_date: str = None
) -> torch.Tensor:
    import pandas as pd

    scores = torch.zeros(len(article_to_idx))
    usecols = ['article_id'] if start_date is None and end_date is None else ['t_dat', 'article_id']
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None

    for chunk in pd.read_csv(transactions_path, usecols=usecols, chunksize=chunksize):
        if 't_dat' in chunk.columns:
            chunk['t_dat'] = pd.to_datetime(chunk['t_dat'])
            if start_ts is not None:
                chunk = chunk[chunk['t_dat'] >= start_ts]
            if end_ts is not None:
                chunk = chunk[chunk['t_dat'] <= end_ts]
        counts = chunk['article_id'].value_counts()
        for article_id, count in counts.items():
            article_id = int(article_id)
            if article_id in article_to_idx:
                scores[article_to_idx[article_id]] += int(count)
    max_count = scores.max().clamp(min=1)
    return scores / max_count
