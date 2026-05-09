#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import rees46_exact_source_protected_list_fusion as exact
import rees46_no_training_baselines as base
import rees46_semantic_bridge_audit as semantic
import rees46_validation_safe_gate as gate


DEFAULT_EXISTING_WEIGHTS = OrderedDict([
    ('global_purchase_popularity', 0.025),
    ('context_event_weighted_replay', 0.975),
])

_WORKER_STATE = None
_WORKER_BUCKET_RANKS = None
_WORKER_SCHEME = None
_WORKER_SCOPE = None
_WORKER_EXISTING_WEIGHTS = None
_WORKER_ARGS = None
_WORKER_KEEP_ALL_ZERO = None


def parse_ints(text):
    return [int(x) for x in text.split(',') if x.strip()]


def parse_weights(text):
    if not text.strip():
        return DEFAULT_EXISTING_WEIGHTS.copy()
    weights = OrderedDict()
    for part in text.split(','):
        if not part.strip():
            continue
        name, value = part.split(':', 1)
        weights[name.strip()] = float(value)
    return weights


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def product_sort_key(product_id):
    return str(product_id)


def build_phase_state(artifact, phase):
    active_window = 'val' if phase == 'val' else 'test'
    catalogs = artifact['catalogs']['products']
    active_products = catalogs[active_window]
    counters = {
        'pre_all': base.counter_from_top(artifact['popularity_top'].get('pre_all', [])),
        'val_implicit': base.counter_from_top(artifact['popularity_top'].get('val_implicit', [])),
        'val_all': base.counter_from_top(artifact['popularity_top'].get('val_all', [])),
        'pre_purchase': base.counter_from_top(artifact['purchase_popularity_top'].get('pre_purchase', [])),
        'val_purchase': base.counter_from_top(artifact['purchase_popularity_top'].get('val_purchase', [])),
    }
    meta_for_existing = base.product_meta_maps(artifact['product_meta'])
    if phase == 'val':
        indexed = base.build_ranked_index(
            active_products,
            meta_for_existing,
            counters['pre_all'],
            counters['pre_all'],
            counters['pre_all'],
            counters['pre_purchase'],
        )
        global_counter = counters['pre_all']
        fallback_counter = counters['pre_all']
        sem_pop_purchase = counters['pre_purchase']
        sem_pop_implicit = counters['pre_all']
        sem_pop_all = counters['pre_all']
    else:
        indexed = base.build_ranked_index(
            active_products,
            meta_for_existing,
            counters['val_implicit'],
            counters['val_all'],
            counters['pre_all'],
            counters['val_purchase'],
        )
        global_counter = counters['val_implicit']
        fallback_counter = counters['pre_all']
        sem_pop_purchase = counters['val_purchase']
        sem_pop_implicit = counters['val_implicit']
        sem_pop_all = counters['val_all']
    return OrderedDict([
        ('active_products', active_products),
        ('indexed', indexed),
        ('global_counter', global_counter),
        ('fallback_counter', fallback_counter),
        ('sem_meta', semantic.normalize_meta(artifact['product_meta'])),
        ('sem_pop_purchase', sem_pop_purchase),
        ('sem_pop_implicit', sem_pop_implicit),
        ('sem_pop_pre', counters['pre_all']),
        ('sem_pop_all', sem_pop_all),
    ])


def build_bucket_ranks_for_phase(state, scheme, args):
    bucket_ranks, _ = semantic.build_bucket_ranks(
        state['active_products'],
        state['sem_meta'],
        [scheme],
        state['sem_pop_purchase'],
        state['sem_pop_implicit'],
        state['sem_pop_pre'],
        state['sem_pop_all'],
        args.bucket_topk,
    )
    return bucket_ranks


def feature_names():
    return [
        'existing_recip_rank',
        'existing_present',
        'semantic_recip_rank',
        'semantic_present',
        'both_sources',
        'semantic_only',
        'existing_only',
        'semantic_promotes_rank_key',
        'candidate_order_recip',
        'seen_product',
        'history_product',
        'context_product',
        'source_count',
    ]


def candidate_pool_and_features(record, state, bucket_ranks, scheme, scope, existing_weights, args):
    max_existing = max(args.existing_pool_k, args.slate_len)
    source_lists = gate.make_source_lists(
        record['profile'],
        state['indexed'],
        state['global_counter'],
        state['fallback_counter'],
        event_weights={'view': args.event_view_weight, 'cart': args.event_cart_weight},
    )
    existing_list = gate.blend_ranked_lists(
        source_lists,
        existing_weights,
        state['indexed']['rank_order'],
        args.source_topk,
        max_existing,
    )
    semantic_list = semantic.candidate_list_for_user(
        record['semantic_record'],
        scheme,
        scope,
        state['sem_meta'],
        bucket_ranks,
        args.semantic_pool_k,
    )
    existing_rank = {str(p): idx for idx, p in enumerate(existing_list[:args.existing_pool_k])}
    semantic_rank = {str(p): idx for idx, p in enumerate(semantic_list[:args.semantic_pool_k])}

    seen = set()
    candidates = []
    for idx in range(max(args.existing_pool_k, args.semantic_pool_k)):
        if idx < args.existing_pool_k and idx < len(existing_list):
            item = str(existing_list[idx])
            if item not in seen:
                candidates.append(item)
                seen.add(item)
        if idx < args.semantic_pool_k and idx < len(semantic_list):
            item = str(semantic_list[idx])
            if item not in seen:
                candidates.append(item)
                seen.add(item)
        if len(candidates) >= args.slate_len:
            break

    profile = record['profile']
    rows = []
    source_ids = []
    for idx, item in enumerate(candidates):
        er = existing_rank.get(item)
        sr = semantic_rank.get(item)
        in_existing = er is not None
        in_semantic = sr is not None
        source_count = int(in_existing) + int(in_semantic)
        rows.append([
            1.0 / float(er + 1) if in_existing else 0.0,
            1.0 if in_existing else 0.0,
            1.0 / float(sr + 1) if in_semantic else 0.0,
            1.0 if in_semantic else 0.0,
            1.0 if in_existing and in_semantic else 0.0,
            1.0 if in_semantic and not in_existing else 0.0,
            1.0 if in_existing and not in_semantic else 0.0,
            1.0 if in_semantic and (not in_existing or sr < er) else 0.0,
            1.0 / float(idx + 1),
            1.0 if item in profile['seen_products'] else 0.0,
            1.0 if item in profile['history_products'] else 0.0,
            1.0 if item in profile['feedback_products'] else 0.0,
            float(source_count) / 2.0,
        ])
        if in_existing and in_semantic:
            source_ids.append(3)
        elif in_semantic:
            source_ids.append(2)
        elif in_existing:
            source_ids.append(1)
        else:
            source_ids.append(0)
    return existing_list, semantic_list, candidates, rows, source_ids


def make_examples(records, state, bucket_ranks, scheme, scope, existing_weights, args, phase, keep_all_zero):
    if args.workers and args.workers > 1:
        return make_examples_parallel(records, state, bucket_ranks, scheme, scope, existing_weights, args, phase, keep_all_zero)
    examples = []
    skipped_all_zero = 0
    rng = random.Random(args.seed + (0 if phase == 'train' else 17))
    for idx, record in enumerate(records):
        if idx and idx % 1000 == 0:
            print(f'[{phase}] built {idx}/{len(records)} slates', flush=True)
        existing_list, _, candidates, features, source_ids = candidate_pool_and_features(
            record, state, bucket_ranks, scheme, scope, existing_weights, args
        )
        if not candidates:
            continue
        target_set = set(str(t) for t in record['targets'])
        labels = [1.0 if item in target_set else 0.0 for item in candidates]
        if sum(labels) <= 0.0 and (not keep_all_zero) and args.zero_slate_keep_prob < 1.0:
            if rng.random() > args.zero_slate_keep_prob:
                skipped_all_zero += 1
                continue
        examples.append(OrderedDict([
            ('user_id', record['user_id']),
            ('targets', [str(t) for t in record['targets']]),
            ('existing_list', [str(x) for x in existing_list[:max(args.ks)] if str(x)]),
            ('candidates', candidates),
            ('features', features),
            ('source_ids', source_ids),
            ('labels', labels),
        ]))
    print(f'[{phase}] examples={len(examples)} skipped_all_zero={skipped_all_zero}', flush=True)
    return examples


def init_example_worker(state, bucket_ranks, scheme, scope, existing_weights, args, keep_all_zero):
    global _WORKER_STATE, _WORKER_BUCKET_RANKS, _WORKER_SCHEME, _WORKER_SCOPE
    global _WORKER_EXISTING_WEIGHTS, _WORKER_ARGS, _WORKER_KEEP_ALL_ZERO
    _WORKER_STATE = state
    _WORKER_BUCKET_RANKS = bucket_ranks
    _WORKER_SCHEME = scheme
    _WORKER_SCOPE = scope
    _WORKER_EXISTING_WEIGHTS = existing_weights
    _WORKER_ARGS = args
    _WORKER_KEEP_ALL_ZERO = keep_all_zero


def build_example_worker(item):
    idx, record = item
    existing_list, _, candidates, features, source_ids = candidate_pool_and_features(
        record,
        _WORKER_STATE,
        _WORKER_BUCKET_RANKS,
        _WORKER_SCHEME,
        _WORKER_SCOPE,
        _WORKER_EXISTING_WEIGHTS,
        _WORKER_ARGS,
    )
    if not candidates:
        return None, 0
    target_set = set(str(t) for t in record['targets'])
    labels = [1.0 if item in target_set else 0.0 for item in candidates]
    if sum(labels) <= 0.0 and (not _WORKER_KEEP_ALL_ZERO) and _WORKER_ARGS.zero_slate_keep_prob < 1.0:
        rng = random.Random(_WORKER_ARGS.seed + idx)
        if rng.random() > _WORKER_ARGS.zero_slate_keep_prob:
            return None, 1
    return OrderedDict([
        ('user_id', record['user_id']),
        ('targets', [str(t) for t in record['targets']]),
        ('existing_list', [str(x) for x in existing_list[:max(_WORKER_ARGS.ks)] if str(x)]),
        ('candidates', candidates),
        ('features', features),
        ('source_ids', source_ids),
        ('labels', labels),
    ]), 0


def make_examples_parallel(records, state, bucket_ranks, scheme, scope, existing_weights, args, phase, keep_all_zero):
    examples = []
    skipped_all_zero = 0
    workers = min(int(args.workers), max(len(records), 1))
    ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else mp.get_context()
    print(f'[{phase}] building slates with workers={workers}', flush=True)
    with ctx.Pool(
        processes=workers,
        initializer=init_example_worker,
        initargs=(state, bucket_ranks, scheme, scope, existing_weights, args, keep_all_zero),
    ) as pool:
        for count, (example, skipped) in enumerate(pool.imap(build_example_worker, enumerate(records), chunksize=args.worker_chunksize), start=1):
            if example is not None:
                examples.append(example)
            skipped_all_zero += int(skipped)
            if count % 1000 == 0:
                print(f'[{phase}] built {count}/{len(records)} slates examples={len(examples)}', flush=True)
    print(f'[{phase}] examples={len(examples)} skipped_all_zero={skipped_all_zero}', flush=True)
    return examples


class SlateDataset(Dataset):
    def __init__(self, examples, slate_len, n_features):
        self.examples = examples
        self.slate_len = slate_len
        self.n_features = n_features

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        n = min(len(ex['candidates']), self.slate_len)
        x = torch.zeros(self.slate_len, self.n_features, dtype=torch.float32)
        source = torch.zeros(self.slate_len, dtype=torch.long)
        y = torch.zeros(self.slate_len, dtype=torch.float32)
        mask = torch.zeros(self.slate_len, dtype=torch.bool)
        if n:
            x[:n] = torch.tensor(ex['features'][:n], dtype=torch.float32)
            source[:n] = torch.tensor(ex['source_ids'][:n], dtype=torch.long)
            y[:n] = torch.tensor(ex['labels'][:n], dtype=torch.float32)
            mask[:n] = True
        return x, source, y, mask


class SlateReranker(nn.Module):
    def __init__(self, model_type, n_features, slate_len, hidden_dim, n_layers, dropout):
        super().__init__()
        self.model_type = model_type
        self.input = nn.Linear(n_features, hidden_dim)
        self.source_embedding = nn.Embedding(4, hidden_dim)
        self.position_embedding = nn.Embedding(slate_len, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        if model_type == 'prm':
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.context = nn.TransformerEncoder(layer, num_layers=n_layers)
        elif model_type == 'dlcm':
            self.context = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, source_ids, mask):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        h = self.input(x) + self.source_embedding(source_ids) + self.position_embedding(positions)
        h = self.norm(h)
        if self.model_type == 'prm':
            h = self.context(h, src_key_padding_mask=~mask)
        else:
            h_rev = torch.flip(h, dims=[1])
            h_rev, _ = self.context(h_rev)
            h = torch.flip(h_rev, dims=[1])
        scores = self.output(h).squeeze(-1)
        return scores.masked_fill(~mask, -1e9)


def masked_bce_loss(scores, labels, mask):
    valid_labels = labels[mask]
    valid_scores = scores[mask]
    positives = valid_labels.sum()
    negatives = valid_labels.numel() - positives
    pos_weight = (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=100.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(valid_scores, valid_labels)


def train_model(train_examples, dev_examples, args, n_features):
    device = torch.device(args.device)
    model = SlateReranker(args.model, n_features, args.slate_len, args.hidden_dim, args.layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        SlateDataset(train_examples, args.slate_len, n_features),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == 'cuda',
    )
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for x, source, y, mask in train_loader:
            x = x.to(device)
            source = source.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(x, source, mask)
            loss = masked_bce_loss(scores, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
        dev_summary, _, _ = evaluate_examples(model, dev_examples, args, n_features, phase=f'dev_epoch{epoch}', write_rows=False)
        history.append(OrderedDict([
            ('epoch', epoch),
            ('train_loss', float(total_loss / max(total_batches, 1))),
            ('dev_hit_counts', dev_summary['hit_counts']),
            ('dev_net_gain', dev_summary['net_gain']),
            ('dev_cannibalization_ratio', dev_summary['cannibalization_ratio']),
        ]))
        key = (
            dev_summary['hit_counts'].get(f'hit@{args.selection_k}', 0),
            dev_summary['net_gain'].get(f'@{args.selection_k}', 0),
            -float(dev_summary['mean_first_hit_rank'] or 1e9),
        )
        print(f"[train] epoch={epoch} loss={history[-1]['train_loss']:.6f} dev_hit@{args.selection_k}={key[0]}", flush=True)
        if best is None or key > best['key']:
            best = OrderedDict([
                ('key', key),
                ('epoch', epoch),
                ('state_dict', {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}),
                ('dev_summary', dev_summary),
            ])
    model.load_state_dict(best['state_dict'])
    return model, best, history


def first_hit_rank(items, targets):
    target_set = set(str(t) for t in targets)
    for idx, item in enumerate(items):
        if item in target_set:
            return idx
    return -1


def summarize_eval(rows, ks, latency_seconds):
    out = OrderedDict([
        ('n_users', len(rows)),
        ('hit_counts', OrderedDict()),
        ('hit_rates', OrderedDict()),
        ('gross_recovery', OrderedDict()),
        ('cannibalized_hit', OrderedDict()),
        ('net_gain', OrderedDict()),
        ('cannibalization_ratio', OrderedDict()),
        ('mean_first_hit_rank', None),
        ('latency_seconds_total', float(latency_seconds)),
        ('latency_ms_per_user', float(latency_seconds * 1000.0 / max(len(rows), 1))),
    ])
    ranks = [row['reranked_first_hit_rank'] for row in rows]
    found = [r for r in ranks if r >= 0]
    out['mean_first_hit_rank'] = float(sum(found) / len(found)) if found else None
    for k in ks:
        hit = sum(1 for row in rows if 0 <= row['reranked_first_hit_rank'] < k)
        base_hit = sum(1 for row in rows if 0 <= row['existing_first_hit_rank'] < k)
        gross = sum(1 for row in rows if row[f'gross@{k}'])
        cannibal = sum(1 for row in rows if row[f'cannibal@{k}'])
        out['hit_counts'][f'hit@{k}'] = int(hit)
        out['hit_rates'][f'hit@{k}'] = float(hit / max(len(rows), 1))
        out['gross_recovery'][f'@{k}'] = int(gross)
        out['cannibalized_hit'][f'@{k}'] = int(cannibal)
        out['net_gain'][f'@{k}'] = int(hit - base_hit)
        out['cannibalization_ratio'][f'@{k}'] = float(cannibal / gross) if gross else None
    return out


def evaluate_examples(model, examples, args, n_features, phase, write_rows=True):
    device = torch.device(args.device)
    loader = DataLoader(
        SlateDataset(examples, args.slate_len, n_features),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda',
    )
    model.eval()
    all_scores = []
    start = time.perf_counter()
    with torch.no_grad():
        for x, source, _, mask in loader:
            scores = model(x.to(device), source.to(device), mask.to(device))
            all_scores.extend(scores.cpu().numpy().tolist())
    latency = time.perf_counter() - start
    rows = []
    max_k = max(args.ks)
    for ex, scores in zip(examples, all_scores):
        n = len(ex['candidates'])
        candidates = ex['candidates'][:n]
        local_scores = scores[:n]
        ranked = [
            item for _, item in sorted(
                zip(local_scores, candidates),
                key=lambda pair: (-pair[0], product_sort_key(pair[1])),
            )
        ][:max_k]
        existing_rank = first_hit_rank(ex['existing_list'][:max_k], ex['targets'])
        rerank_rank = first_hit_rank(ranked, ex['targets'])
        row = OrderedDict([
            ('user_id', ex['user_id']),
            ('n_candidates', n),
            ('n_positive_candidates', int(sum(ex['labels']))),
            ('existing_first_hit_rank', int(existing_rank)),
            ('reranked_first_hit_rank', int(rerank_rank)),
        ])
        for k in args.ks:
            base_hit = 0 <= existing_rank < k
            new_hit = 0 <= rerank_rank < k
            row[f'gross@{k}'] = int((not base_hit) and new_hit)
            row[f'cannibal@{k}'] = int(base_hit and (not new_hit))
        rows.append(row)
    summary = summarize_eval(rows, args.ks, latency)
    print(f"[{phase}] users={summary['n_users']} latency_ms_per_user={summary['latency_ms_per_user']:.4f}", flush=True)
    return summary, rows if write_rows else None, latency


def write_outputs(summary, history, test_rows, args):
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f'rees46_{args.model}_slate_reranker_summary.json')
    row_path = os.path.join(args.output_dir, f'rees46_{args.model}_slate_reranker_test_per_user.csv')
    md_path = os.path.join(args.output_dir, f'rees46_{args.model}_slate_reranker_summary.md')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(test_rows).to_csv(row_path, index=False)
    lines = [
        f"# REES46 {args.model.upper()} Slate-Aware Reranker",
        '',
        f"- Candidate pool: existing top `{args.existing_pool_k}` + semantic top `{args.semantic_pool_k}`; slate length `{args.slate_len}`",
        f"- Semantic source: `{args.semantic_source}`",
        f"- Selected epoch: `{summary['selected_epoch']}`",
        f"- Test latency: `{summary['test_summary']['latency_ms_per_user']:.4f}` ms/user on `{args.device}`",
        '',
        '## Test Result',
        '',
        '| System | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    test = summary['test_summary']
    gross100 = test['gross_recovery'].get('@100', 0)
    cann100 = test['cannibalized_hit'].get('@100', 0)
    net100 = test['net_gain'].get('@100', 0)
    ratio100 = test['cannibalization_ratio'].get('@100')
    ratio_text = '' if ratio100 is None else f'{ratio100:.3f}'
    lines.append(
        f"| {args.model.upper()} | {test['hit_counts'].get('hit@10', 0)} | "
        f"{test['hit_counts'].get('hit@50', 0)} | {test['hit_counts'].get('hit@100', 0)} | "
        f"{test['hit_counts'].get('hit@500', 0)} | {gross100} | {cann100} | {net100:+d} | {ratio_text} |"
    )
    lines.extend([
        '',
        '## Training History',
        '',
        '| Epoch | Loss | Dev Hit@100 | Dev Net@100 | Dev Ratio@100 |',
        '|---:|---:|---:|---:|---:|',
    ])
    for row in history:
        ratio = row['dev_cannibalization_ratio'].get('@100')
        ratio_text = '' if ratio is None else f'{ratio:.3f}'
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | "
            f"{row['dev_hit_counts'].get('hit@100', 0)} | "
            f"{row['dev_net_gain'].get('@100', 0):+d} | {ratio_text} |"
        )
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return json_path, md_path, row_path


def main():
    parser = argparse.ArgumentParser(description='REES46 slate-aware PRM/DLCM reranker over a fixed existing+semantic candidate pool.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', choices=['prm', 'dlcm'], default='prm')
    parser.add_argument('--semantic-source', default='semid_category_brand_context')
    parser.add_argument('--existing-weights', default='')
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--selection-k', type=int, default=100)
    parser.add_argument('--val-context-frac', type=float, default=0.5)
    parser.add_argument('--val-target-types', default='')
    parser.add_argument('--source-topk', type=int, default=500)
    parser.add_argument('--existing-pool-k', type=int, default=500)
    parser.add_argument('--semantic-pool-k', type=int, default=500)
    parser.add_argument('--slate-len', type=int, default=1000)
    parser.add_argument('--bucket-topk', type=int, default=500)
    parser.add_argument('--event-view-weight', type=float, default=1.0)
    parser.add_argument('--event-cart-weight', type=float, default=3.0)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--dev-frac', type=float, default=0.2)
    parser.add_argument('--zero-slate-keep-prob', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--worker-chunksize', type=int, default=32)
    parser.add_argument('--max-users', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.ks = parse_ints(args.ks)
    seed_everything(args.seed)

    scheme, scope, _ = exact.split_semantic_source(args.semantic_source)
    existing_weights = parse_weights(args.existing_weights)
    artifact = semantic.load_artifact(args.artifact)
    users = artifact['users'][:args.max_users] if args.max_users else artifact['users']
    val_records = exact.make_records(
        users,
        'val',
        args.val_context_frac,
        [x.strip() for x in args.val_target_types.split(',') if x.strip()],
    )
    test_records = exact.make_records(users, 'test', args.val_context_frac, [])
    split = int(len(val_records) * (1.0 - args.dev_frac))
    train_records = val_records[:split]
    dev_records = val_records[split:]
    print(
        f"Loaded users={len(users)} train={len(train_records)} dev={len(dev_records)} test={len(test_records)} "
        f"model={args.model} device={args.device}",
        flush=True,
    )

    val_state = build_phase_state(artifact, 'val')
    test_state = build_phase_state(artifact, 'test')
    val_bucket_ranks = build_bucket_ranks_for_phase(val_state, scheme, args)
    test_bucket_ranks = build_bucket_ranks_for_phase(test_state, scheme, args)
    train_examples = make_examples(
        train_records,
        val_state,
        val_bucket_ranks,
        scheme,
        scope,
        existing_weights,
        args,
        'train',
        keep_all_zero=False,
    )
    dev_examples = make_examples(
        dev_records,
        val_state,
        val_bucket_ranks,
        scheme,
        scope,
        existing_weights,
        args,
        'dev',
        keep_all_zero=True,
    )
    test_examples = make_examples(
        test_records,
        test_state,
        test_bucket_ranks,
        scheme,
        scope,
        existing_weights,
        args,
        'test',
        keep_all_zero=True,
    )
    if not train_examples:
        raise RuntimeError('No positive training slates found in the fixed candidate pool.')

    n_features = len(feature_names())
    model, best, history = train_model(train_examples, dev_examples, args, n_features)
    test_summary, test_rows, _ = evaluate_examples(model, test_examples, args, n_features, phase='test', write_rows=True)
    summary = OrderedDict([
        ('args', vars(args)),
        ('feature_names', feature_names()),
        ('n_train_examples', len(train_examples)),
        ('n_dev_examples', len(dev_examples)),
        ('n_test_examples', len(test_examples)),
        ('selected_epoch', int(best['epoch'])),
        ('dev_summary', best['dev_summary']),
        ('test_summary', test_summary),
        ('training_history', history),
        ('protocol_note', 'Slate-aware baseline over the fixed existing+semantic candidate pool. Labels are validation-holdout/test purchases only for their split; semantic_all_oracle is not used.'),
    ])
    json_path, md_path, row_path = write_outputs(summary, history, test_rows, args)
    print(json.dumps({'summary': json_path, 'markdown': md_path, 'per_user': row_path}, indent=2))


if __name__ == '__main__':
    main()
