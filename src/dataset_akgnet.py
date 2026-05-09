import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from .consts import MAX_HISTORY_LEN

ATTR_COLUMNS = ['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'department_no']


def date_to_ordinal(value) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(pd.Timestamp(value).toordinal())


class CrossSeasonDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        articles_path: str,
        split: str = 'train',
        device: torch.device = None
    ):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        if split == 'train':
            self.samples = self.data['train_samples']
            self.split_ref = None
        elif split == 'val':
            self.samples = self.data['val_samples']
            self.split_ref = self.data.get('val_ref')
        elif split == 'test':
            self.samples = self.data['test_samples']
            self.split_ref = self.data.get('test_ref')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.article_to_emb_idx = self.data['article_to_idx']
        self.idx_to_article = self.data['idx_to_article']
        self.n_items = len(self.article_to_emb_idx)
        
        self.articles_df = pd.read_csv(articles_path)
        self.article_to_row = {int(aid): idx for idx, aid in enumerate(self.articles_df['article_id'])}
        self.attr_mappings = self._build_attr_mappings(self.articles_df)
        self.attr_dim = len(ATTR_COLUMNS)
        self.attr_dims = [len(self.attr_mappings[col]) for col in ATTR_COLUMNS]
        self.article_attrs = self._build_catalog_attrs(self.articles_df)
        self.seasonal_catalog_indices = {}
        for ref, catalog in self.data.get('seasonal_catalogs', {}).items():
            self.seasonal_catalog_indices[ref] = torch.tensor(
                [self.article_to_emb_idx[int(aid)] for aid in catalog],
                dtype=torch.int64
            )
        
        self.device = device if device is not None else torch.device('cpu')
    
    def _build_attr_mappings(self, articles_df: pd.DataFrame) -> Dict[str, Dict[int, int]]:
        mappings = {}
        for col in ATTR_COLUMNS:
            unique_vals = sorted(articles_df[col].unique())
            mappings[col] = {v: i for i, v in enumerate(unique_vals)}
        return mappings
    
    def _build_catalog_attrs(self, articles_df: pd.DataFrame) -> torch.Tensor:
        n_catalog = len(self.article_to_emb_idx)
        attrs = torch.zeros((n_catalog, self.attr_dim), dtype=torch.int64)
        
        for article_id, emb_idx in self.article_to_emb_idx.items():
            if article_id in self.article_to_row:
                row_idx = self.article_to_row[article_id]
                row = articles_df.iloc[row_idx]
                for col_idx, col in enumerate(ATTR_COLUMNS):
                    val = row[col]
                    attrs[emb_idx, col_idx] = self.attr_mappings[col][val]
        
        return attrs
    
    def get_catalog_attrs(self) -> torch.Tensor:
        return self.article_attrs
    
    def get_seasonal_catalog(self, t_ref: str = None) -> List[int]:
        if t_ref is not None and 'seasonal_catalogs' in self.data:
            return self.data['seasonal_catalogs'][t_ref]
        if self.split_ref is not None and 'seasonal_catalogs' in self.data:
            return self.data['seasonal_catalogs'][self.split_ref]
        return self.data['seasonal_catalog_combined']
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _pad_history(self, history_items: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = torch.zeros(MAX_HISTORY_LEN, dtype=torch.int64)
        mask = torch.zeros(MAX_HISTORY_LEN, dtype=torch.float32)
        history_items = history_items[-MAX_HISTORY_LEN:]
        n = len(history_items)
        start = MAX_HISTORY_LEN - n
        for i, item in enumerate(history_items):
            padded[start + i] = self.article_to_emb_idx.get(item, 0)
        mask[start:] = 1.0
        return padded, mask
    
    def _get_history_attrs(self, history_items: List[int], articles_df: pd.DataFrame) -> torch.Tensor:
        padded_attrs = torch.zeros((MAX_HISTORY_LEN, self.attr_dim), dtype=torch.int64)
        history_items = history_items[-MAX_HISTORY_LEN:]
        n = len(history_items)
        start = MAX_HISTORY_LEN - n
        for i, item in enumerate(history_items):
            if item in self.article_to_row:
                row_idx = self.article_to_row[item]
                row = articles_df.iloc[row_idx]
                for col_idx, col in enumerate(ATTR_COLUMNS):
                    val = row[col]
                    padded_attrs[start + i, col_idx] = self.attr_mappings[col][val]
        return padded_attrs
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        history_items = sample['history_items']
        target_item = sample['target_item']
        gap_days = sample['gap_days']
        ref_ord = date_to_ordinal(sample.get('t_ref', self.split_ref))
        
        history_padded, history_mask = self._pad_history(history_items)
        target_emb_idx = self.article_to_emb_idx.get(target_item, 0)
        
        if 'negatives' in sample:
            neg_embs = torch.tensor(
                [self.article_to_emb_idx.get(neg, 0) for neg in sample['negatives']],
                dtype=torch.int64
            )
            neg_weights = torch.tensor(
                sample.get('negative_weights', [1.0] * len(sample['negatives'])),
                dtype=torch.float32
            )
        else:
            ref = sample.get('t_ref', self.split_ref)
            catalog_indices = self.seasonal_catalog_indices.get(ref)
            if catalog_indices is None:
                catalog_indices = torch.arange(self.n_items, dtype=torch.int64)
            pool = catalog_indices[catalog_indices != target_emb_idx]
            choice = torch.randint(0, len(pool), (63,), dtype=torch.int64)
            neg_embs = pool[choice]
            neg_weights = torch.ones(len(neg_embs), dtype=torch.float32)
        
        history_attrs = self._get_history_attrs(history_items, self.articles_df)
        
        gap_tensor = torch.tensor(gap_days, dtype=torch.float32)
        
        return {
            'history_items': history_padded,
            'history_attrs': history_attrs,
            'history_mask': history_mask,
            'target_item': torch.tensor(target_emb_idx, dtype=torch.int64),
            'negatives': neg_embs,
            'negative_weights': neg_weights,
            'gap_days': gap_tensor,
            'ref_ord': torch.tensor(ref_ord, dtype=torch.float32),
            'original_target_id': target_item
        }


def create_evaluation_pairs(
    dataset: CrossSeasonDataset,
    split: str = 'test'
) -> List[Dict]:
    if split == 'train':
        samples = dataset.data['train_samples']
    elif split == 'val':
        samples = dataset.data['val_samples']
    elif split == 'test':
        samples = dataset.data['test_samples']
    else:
        raise ValueError(f"Unknown split: {split}")
    
    pairs = []
    for sample in samples:
        history_items = sample['history_items']
        target_article_id = sample['target_item']
        gap_days = sample['gap_days']
        
        history_padded, history_mask = dataset._pad_history(history_items)
        history_attrs = dataset._get_history_attrs(history_items, dataset.articles_df)
        
        pairs.append({
            'history_items': history_padded,
            'history_attrs': history_attrs,
            'history_mask': history_mask,
            'gap_days': torch.tensor(gap_days, dtype=torch.float32),
            'target_article_id': target_article_id
        })
    return pairs
