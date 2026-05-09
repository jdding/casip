import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
from .consts import *


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, DATE_FMT)


def build_seasonal_catalog(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime
) -> List[int]:
    window = df[(df['t_dat'] >= start_date) & (df['t_dat'] <= end_date)]
    return sorted(window['article_id'].unique().tolist())


def extract_a_users(
    df: pd.DataFrame,
    t_train: datetime,
    min_gap_days: int = 365
) -> List[int]:
    all_users = df['customer_id'].unique()
    a_users = []
    
    for user_id in all_users:
        user_trans = df[df['customer_id'] == user_id].sort_values('t_dat')
        if len(user_trans) < 2:
            continue
        
        last_pre_b = user_trans[user_trans['t_dat'] < t_train].iloc[-1]
        gap_start = last_pre_b['t_dat']
        gap_days = (t_train - gap_start).days
        
        if gap_days > min_gap_days:
            has_post_b = len(user_trans[user_trans['t_dat'] >= t_train]) > 0
            if has_post_b:
                a_users.append(user_id)
    
    return a_users


def build_sample(
    user_trans: pd.DataFrame,
    user_id: int,
    t_train: datetime,
    seasonal_catalog: List[int],
    max_history: int = 20,
    num_negatives: int = 63
) -> Optional[Dict]:
    user_trans_sorted = user_trans.sort_values('t_dat')
    
    gap_start = user_trans_sorted[user_trans_sorted['t_dat'] < t_train].iloc[-1]['t_dat']
    gap_days = (t_train - gap_start).days
    
    history_trans = user_trans_sorted[user_trans_sorted['t_dat'] < gap_start]
    if len(history_trans) == 0:
        return None
    
    history_items = history_trans['article_id'].tolist()[-max_history:]
    if len(history_items) < len(history_trans):
        history_items = history_items[-max_history:]
    
    b_window_trans = user_trans_sorted[user_trans_sorted['t_dat'] >= t_train]
    first_target = b_window_trans.iloc[0]['article_id']
    
    if first_target not in seasonal_catalog:
        return None
    
    seasonal_set = set(seasonal_catalog)
    available_negatives = list(seasonal_set - {first_target})
    if len(available_negatives) < num_negatives:
        return None
    
    negatives = np.random.choice(available_negatives, num_negatives, replace=False).tolist()
    
    return {
        'customer_id': user_id,
        'history_items': history_items,
        'target_item': first_target,
        'gap_days': gap_days,
        'negatives': negatives
    }


def process_split(
    df: pd.DataFrame,
    articles_df: pd.DataFrame,
    t_train: datetime,
    min_gap_days: int = 365,
    window_days: int = 90,
    max_history: int = 20,
    num_negatives: int = 63
) -> Tuple[List[Dict], List[int]]:
    b_start = t_train - timedelta(days=window_days)
    b_end = t_train
    
    seasonal_catalog = build_seasonal_catalog(df, b_start, b_end)
    a_users = extract_a_users(df, t_train, min_gap_days)
    
    samples = []
    for user_id in a_users:
        user_trans = df[df['customer_id'] == user_id]
        sample = build_sample(
            user_trans, user_id, t_train, seasonal_catalog,
            max_history, num_negatives
        )
        if sample is not None:
            samples.append(sample)
    
    return samples, seasonal_catalog


def compute_gate0_metrics(
    train_samples: List[Dict],
    seasonal_catalog: List[int],
    validation_samples: List[Dict],
    test_samples: List[Dict],
    min_gap_days: int = 365
) -> Dict:
    n_train = len(train_samples)
    
    oov_count = 0
    for sample in train_samples:
        if sample['target_item'] not in set(sample['history_items']):
            oov_count += 1
    oov_rate = oov_count / max(n_train, 1)
    
    history_items_all = set()
    for sample in train_samples:
        history_items_all.update(sample['history_items'])
    seasonal_set = set(seasonal_catalog)
    overlap = len(seasonal_set & history_items_all)
    catalog_overlap_rate = overlap / max(len(seasonal_set), 1)
    
    n_val = len(validation_samples)
    n_test = len(test_samples)
    
    return {
        'n_train_a_users': n_train,
        'target_oov_rate': oov_rate,
        'catalog_overlap_rate': catalog_overlap_rate,
        'n_val_a_users': n_val,
        'n_test_a_users': n_test
    }


def build_cross_season_dataset(
    transactions_path: str,
    articles_path: str,
    output_path: str,
    train_dates: List[str] = None,
    val_date: str = "2020-05-01",
    test_date: str = "2020-09-01",
    min_gap_days: int = 365
) -> Dict:
    if train_dates is None:
        train_dates = ["2019-06-01", "2019-09-01", "2019-12-01", "2020-03-01"]
    
    print(f"Loading transactions from {transactions_path}...")
    df = pd.read_csv(transactions_path)
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    
    print(f"Loading articles from {articles_path}...")
    articles_df = pd.read_csv(articles_path)
    
    customer_ids = df['customer_id'].unique()
    customer_to_idx = {cid: i for i, cid in enumerate(customer_ids)}
    article_ids = df['article_id'].unique()
    article_to_idx = {aid: i for i, aid in enumerate(article_ids)}
    
    df['customer_id'] = df['customer_id'].map(customer_to_idx)
    df['article_id'] = df['article_id'].map(article_to_idx)
    
    all_samples = []
    all_seasonal_catalogs = []
    
    for date_str in train_dates:
        t_train = parse_date(date_str)
        print(f"Processing train split {date_str}...")
        samples, seasonal = process_split(
            df, articles_df, t_train, min_gap_days,
            window_days=90, max_history=MAX_HISTORY_LEN
        )
        all_samples.extend(samples)
        all_seasonal_catalogs.extend(seasonal)
    
    combined_seasonal = sorted(list(set(all_seasonal_catalogs)))
    
    print(f"Processing validation {val_date}...")
    t_val = parse_date(val_date)
    val_samples, val_seasonal = process_split(
        df, articles_df, t_val, min_gap_days,
        window_days=90, max_history=MAX_HISTORY_LEN
    )
    
    print(f"Processing test {test_date}...")
    t_test = parse_date(test_date)
    test_samples, test_seasonal = process_split(
        df, articles_df, t_test, min_gap_days,
        window_days=90, max_history=MAX_HISTORY_LEN
    )
    
    metrics = compute_gate0_metrics(
        all_samples, combined_seasonal, val_samples, test_samples, min_gap_days
    )
    
    article_to_row = {aid: idx for idx, aid in enumerate(article_ids)}
    
    result = {
        'train_samples': all_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'seasonal_catalog_combined': combined_seasonal,
        'article_to_idx': article_to_idx,
        'idx_to_article': {i: a for a, i in article_to_idx.items()},
        'customer_to_idx': customer_to_idx,
        'article_attrs': None,
        'gate0_metrics': metrics
    }
    
    print(f"Saving to {output_path}...")
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    gate0_report = {
        'gate_0_1_n_users': metrics['n_train_a_users'],
        'gate_0_1_passed': metrics['n_train_a_users'] > 3000,
        'gate_0_2_oov_rate': metrics['target_oov_rate'],
        'gate_0_2_passed': metrics['target_oov_rate'] > 0.80,
        'gate_0_3_overlap': metrics['catalog_overlap_rate'],
        'gate_0_3_passed': metrics['catalog_overlap_rate'] < 0.50,
        'gate_0_4_val_n': metrics['n_val_a_users'],
        'gate_0_4_passed': 6000 <= metrics['n_val_a_users'] <= 8000,
        'gate_0_5_test_n': metrics['n_test_a_users'],
        'gate_0_5_passed': 15000 <= metrics['n_test_a_users'] <= 18000,
        'all_passed': (
            metrics['n_train_a_users'] > 3000 and
            metrics['target_oov_rate'] > 0.80 and
            metrics['catalog_overlap_rate'] < 0.50 and
            6000 <= metrics['n_val_a_users'] <= 8000 and
            15000 <= metrics['n_test_a_users'] <= 18000
        )
    }
    
    with open(output_path.replace('.pkl', '_gate0_report.json'), 'w') as f:
        json.dump(gate0_report, f, indent=2)
    
    print("\n=== Gate 0 Report ===")
    for k, v in gate0_report.items():
        if k.endswith('_passed'):
            print(f"{k}: {'✅ PASS' if v else '❌ FAIL'}")
    print(f"All passed: {'✅ YES' if gate0_report['all_passed'] else '❌ NO'}")
    
    return result, gate0_report
