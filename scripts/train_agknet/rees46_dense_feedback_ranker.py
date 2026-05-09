#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import random
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
import pandas as pd

import rees46_validation_safe_gate as gate
import rees46_no_training_baselines as base


FEATURE_SOURCES = [
    'global_popularity',
    'context_feedback_replay',
    'seen_product_replay',
    'brand_overlap_seen_allowed',
    'brand_replacement_unseen',
    'category_overlap_seen_allowed',
    'category_or_brand_replacement_unseen',
    'category_code_family_replacement_unseen',
]

_WORKER_INDEXED = None
_WORKER_GLOBAL_COUNTER = None
_WORKER_FALLBACK_COUNTER = None
_WORKER_SOURCE_TOPK = None
_WORKER_CANDIDATE_TOPK = None
_WORKER_MAX_NEGATIVES = None
_WORKER_SEED = None
_WORKER_MODE = None
_WORKER_WEIGHTS = None
_WORKER_MEAN = None
_WORKER_STD = None


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def feature_names():
    names = []
    for source in FEATURE_SOURCES:
        names.append(f'{source}_recip_rank')
        names.append(f'{source}_present')
    names.extend([
        'source_count',
        'seen_product',
        'history_product',
        'context_product',
    ])
    return names


def rank_maps(source_lists, source_topk):
    maps = {}
    for name in FEATURE_SOURCES:
        maps[name] = {product_id: rank for rank, product_id in enumerate(source_lists.get(name, [])[:source_topk])}
    return maps


def candidate_products(source_lists, indexed, source_topk, candidate_topk):
    seen = set()
    for name in FEATURE_SOURCES:
        seen.update(source_lists.get(name, [])[:source_topk])
    return sorted(seen, key=lambda p: indexed['rank_order'].get(p, len(indexed['rank_order'])))[:candidate_topk]


def product_features(product_id, record, maps, source_topk):
    feats = []
    count = 0
    for name in FEATURE_SOURCES:
        rank = maps[name].get(product_id)
        if rank is None:
            feats.extend([0.0, 0.0])
        else:
            count += 1
            feats.extend([1.0 / float(rank + 1), 1.0])
    profile = record['profile']
    feats.extend([
        float(count) / float(max(len(FEATURE_SOURCES), 1)),
        1.0 if product_id in profile['seen_products'] else 0.0,
        1.0 if product_id in profile['history_products'] else 0.0,
        1.0 if product_id in profile['feedback_products'] else 0.0,
    ])
    return feats


def init_worker(indexed, global_counter, fallback_counter, source_topk, candidate_topk, max_negatives, seed, mode, weights, mean, std):
    global _WORKER_INDEXED, _WORKER_GLOBAL_COUNTER, _WORKER_FALLBACK_COUNTER
    global _WORKER_SOURCE_TOPK, _WORKER_CANDIDATE_TOPK, _WORKER_MAX_NEGATIVES, _WORKER_SEED
    global _WORKER_MODE, _WORKER_WEIGHTS, _WORKER_MEAN, _WORKER_STD
    _WORKER_INDEXED = indexed
    _WORKER_GLOBAL_COUNTER = global_counter
    _WORKER_FALLBACK_COUNTER = fallback_counter
    _WORKER_SOURCE_TOPK = source_topk
    _WORKER_CANDIDATE_TOPK = candidate_topk
    _WORKER_MAX_NEGATIVES = max_negatives
    _WORKER_SEED = seed
    _WORKER_MODE = mode
    _WORKER_WEIGHTS = weights
    _WORKER_MEAN = mean
    _WORKER_STD = std


def build_training_row(item):
    row_id, record = item
    source_lists = gate.make_source_lists(record['profile'], _WORKER_INDEXED, _WORKER_GLOBAL_COUNTER, _WORKER_FALLBACK_COUNTER)
    candidates = candidate_products(source_lists, _WORKER_INDEXED, _WORKER_SOURCE_TOPK, _WORKER_CANDIDATE_TOPK)
    candidate_set = set(candidates)
    target_set = set(str(x) for x in record['targets']) & candidate_set
    if not target_set:
        return [], []
    maps = rank_maps(source_lists, _WORKER_SOURCE_TOPK)
    rng = random.Random(_WORKER_SEED + row_id)
    negatives = [p for p in candidates if p not in target_set]
    rng.shuffle(negatives)
    negatives = negatives[:_WORKER_MAX_NEGATIVES]
    x_rows = [product_features(p, record, maps, _WORKER_SOURCE_TOPK) for p in sorted(target_set)]
    y_rows = [1.0] * len(x_rows)
    x_rows.extend(product_features(p, record, maps, _WORKER_SOURCE_TOPK) for p in negatives)
    y_rows.extend([0.0] * len(negatives))
    return x_rows, y_rows


def evaluate_one_record(item):
    row_id, record = item
    source_lists = gate.make_source_lists(record['profile'], _WORKER_INDEXED, _WORKER_GLOBAL_COUNTER, _WORKER_FALLBACK_COUNTER)
    candidates = candidate_products(source_lists, _WORKER_INDEXED, _WORKER_SOURCE_TOPK, _WORKER_CANDIDATE_TOPK)
    maps = rank_maps(source_lists, _WORKER_SOURCE_TOPK)
    if not candidates:
        return OrderedDict([
            ('row_id', row_id),
            ('user_id', record['user_id']),
            ('first_hit_rank', -1),
            ('n_candidates', 0),
        ])
    x = np.asarray([product_features(p, record, maps, _WORKER_SOURCE_TOPK) for p in candidates], dtype=np.float32)
    x = (x - _WORKER_MEAN) / _WORKER_STD
    scores = x @ _WORKER_WEIGHTS[:-1] + _WORKER_WEIGHTS[-1]
    ranked = [p for _, p in sorted(zip(scores.tolist(), candidates), key=lambda sp: (-sp[0], p_sort_key(sp[1])))]
    return OrderedDict([
        ('row_id', row_id),
        ('user_id', record['user_id']),
        ('first_hit_rank', gate.first_hit_rank(ranked, record['targets'])),
        ('n_candidates', len(candidates)),
    ])


def p_sort_key(product_id):
    return str(product_id)


def collect_training_examples(records, indexed, global_counter, fallback_counter, args):
    x_parts = []
    y_parts = []
    print(f'[train] building examples for {len(records)} users with workers={args.workers}', flush=True)
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(
            indexed,
            global_counter,
            fallback_counter,
            args.source_topk,
            args.candidate_topk,
            args.negatives_per_user,
            args.seed,
            'train',
            None,
            None,
            None,
        ),
    ) as pool:
        for count, (x_rows, y_rows) in enumerate(pool.imap(build_training_row, enumerate(records), chunksize=64), start=1):
            if x_rows:
                x_parts.extend(x_rows)
                y_parts.extend(y_rows)
            if count % 1000 == 0:
                print(f'[train] processed {count}/{len(records)} users examples={len(y_parts)}', flush=True)
    x = np.asarray(x_parts, dtype=np.float32)
    y = np.asarray(y_parts, dtype=np.float32)
    return x, y


def train_logistic(x, y, dev_records, indexed, global_counter, fallback_counter, args):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)

    rng = np.random.default_rng(args.seed)
    weights = np.zeros(x.shape[1], dtype=np.float32)
    best = None
    history = []
    pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1.0))
    sample_weight = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            xb = x[idx]
            yb = y[idx]
            wb = sample_weight[idx]
            pred = sigmoid(xb @ weights)
            err = (pred - yb) * wb
            grad = xb.T @ err / float(len(idx))
            grad[:-1] += args.l2 * weights[:-1]
            weights -= args.lr * grad.astype(np.float32)
        dev_summary, _ = evaluate_records(
            dev_records,
            indexed,
            global_counter,
            fallback_counter,
            weights,
            mean,
            std,
            args,
            phase=f'dev_epoch{epoch}',
            write_rows=False,
        )
        hit10 = dev_summary['hit_counts']['hit@10']
        history.append(OrderedDict([
            ('epoch', epoch),
            ('dev_hit_counts', dev_summary['hit_counts']),
            ('dev_mean_first_hit_rank', dev_summary['mean_first_hit_rank']),
        ]))
        print(f'[train] epoch={epoch} dev_hit10={hit10}', flush=True)
        key = (hit10, dev_summary['hit_counts'].get('hit@50', 0), -float(dev_summary['mean_first_hit_rank'] or 1e9))
        if best is None or key > best['key']:
            best = {
                'key': key,
                'epoch': epoch,
                'weights': weights.copy(),
                'mean': mean.copy(),
                'std': std.copy(),
                'dev_summary': dev_summary,
            }
    return best, history


def evaluate_records(records, indexed, global_counter, fallback_counter, weights, mean, std, args, phase, write_rows=True):
    rows = []
    print(f'[{phase}] evaluating {len(records)} users with workers={args.workers}', flush=True)
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(
            indexed,
            global_counter,
            fallback_counter,
            args.source_topk,
            args.candidate_topk,
            args.negatives_per_user,
            args.seed,
            'eval',
            weights,
            mean,
            std,
        ),
    ) as pool:
        for count, row in enumerate(pool.imap(evaluate_one_record, enumerate(records), chunksize=64), start=1):
            rows.append(row)
            if count % 1000 == 0:
                print(f'[{phase}] processed {count}/{len(records)} users', flush=True)
    summary = gate.summarize_rows(rows, args.ks)
    return summary, rows if write_rows else None


def load_artifact(path):
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    if artifact.get('artifact_version') != 'rees46_protocol_v1':
        raise RuntimeError(f'Unsupported artifact version: {artifact.get("artifact_version")}')
    return artifact


def make_indices(artifact):
    catalogs = artifact['catalogs']['products']
    counters = {
        'pre_all': base.counter_from_top(artifact['popularity_top'].get('pre_all', [])),
        'val_implicit': base.counter_from_top(artifact['popularity_top'].get('val_implicit', [])),
        'val_all': base.counter_from_top(artifact['popularity_top'].get('val_all', [])),
    }
    meta = base.product_meta_maps(artifact['product_meta'])
    val_indexed = base.build_ranked_index(
        catalogs['val'],
        meta,
        counters['pre_all'],
        counters['pre_all'],
        counters['pre_all'],
        base.counter_from_top([]),
    )
    test_indexed = base.build_ranked_index(
        catalogs['test'],
        meta,
        counters['val_implicit'],
        counters['val_all'],
        counters['pre_all'],
        base.counter_from_top([]),
    )
    return counters, val_indexed, test_indexed


def main():
    parser = argparse.ArgumentParser(description='Learned dense-feedback REES46 candidate reranker.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--source-topk', type=int, default=500)
    parser.add_argument('--candidate-topk', type=int, default=2500)
    parser.add_argument('--negatives-per-user', type=int, default=120)
    parser.add_argument('--val-context-frac', type=float, default=0.5)
    parser.add_argument('--dev-frac', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--max-users', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.ks = [int(x) for x in args.ks.split(',') if x.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    artifact = load_artifact(args.artifact)
    users = artifact['users']
    if args.max_users:
        users = users[:args.max_users]
    counters, val_indexed, test_indexed = make_indices(artifact)
    val_records = gate.make_phase_records(users, 'val', args.val_context_frac)
    test_records = gate.make_phase_records(users, 'test', args.val_context_frac)
    split = int(len(val_records) * (1.0 - args.dev_frac))
    train_records = val_records[:split]
    dev_records = val_records[split:]
    print(
        f'Loaded users={len(users)} train={len(train_records)} dev={len(dev_records)} '
        f'test={len(test_records)} workers={args.workers}',
        flush=True,
    )

    x, y = collect_training_examples(train_records, val_indexed, counters['pre_all'], counters['pre_all'], args)
    print(f'[train] examples={len(y)} positives={int(y.sum())} features={x.shape[1]}', flush=True)
    best, history = train_logistic(x, y, dev_records, val_indexed, counters['pre_all'], counters['pre_all'], args)
    test_summary, test_rows = evaluate_records(
        test_records,
        test_indexed,
        counters['val_implicit'],
        counters['pre_all'],
        best['weights'],
        best['mean'],
        best['std'],
        args,
        'test',
    )

    learned_weights = OrderedDict(
        list(zip(feature_names() + ['intercept'], [float(x) for x in best['weights'].tolist()]))
    )
    summary = OrderedDict([
        ('args', vars(args)),
        ('n_users_total', len(users)),
        ('n_train_users', len(train_records)),
        ('n_dev_users', len(dev_records)),
        ('n_test_users', len(test_records)),
        ('n_train_examples', int(len(y))),
        ('n_train_positive_examples', int(y.sum())),
        ('feature_names', feature_names()),
        ('selected_epoch', int(best['epoch'])),
        ('dev_summary', best['dev_summary']),
        ('test_summary', test_summary),
        ('training_history', history),
        ('learned_weights', learned_weights),
        ('reference_rule_test_hit_counts', OrderedDict([
            ('hit@10', 2910),
            ('hit@50', 5211),
            ('hit@100', 6199),
            ('hit@500', 8517),
        ])),
    ])

    summary_path = os.path.join(args.output_dir, 'rees46_dense_feedback_ranker_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    rows_path = os.path.join(args.output_dir, 'rees46_dense_feedback_ranker_test_per_user.csv')
    pd.DataFrame(test_rows).to_csv(rows_path, index=False)

    md_path = os.path.join(args.output_dir, 'rees46_dense_feedback_ranker_summary.md')
    lines = [
        '# REES46 Dense-Feedback Ranker',
        '',
        f"- Train/dev/test users: `{len(train_records)}/{len(dev_records)}/{len(test_records)}`",
        f"- Train examples / positives: `{len(y)}` / `{int(y.sum())}`",
        f"- Selected epoch: `{best['epoch']}`",
        f"- Candidate top-k: `{args.candidate_topk}`; source top-k: `{args.source_topk}`",
        '',
        '## Result',
        '',
        '| System | Hit@10 | Hit@50 | Hit@100 | Hit@500 |',
        '|---|---:|---:|---:|---:|',
    ]
    dev = best['dev_summary']['hit_counts']
    test = test_summary['hit_counts']
    ref = summary['reference_rule_test_hit_counts']
    lines.append(f"| learned dev | {dev['hit@10']} | {dev['hit@50']} | {dev['hit@100']} | {dev['hit@500']} |")
    lines.append(f"| learned test | {test['hit@10']} | {test['hit@50']} | {test['hit@100']} | {test['hit@500']} |")
    lines.append(f"| fixed rule test | {ref['hit@10']} | {ref['hit@50']} | {ref['hit@100']} | {ref['hit@500']} |")
    lines.extend([
        '',
        '## Training History',
        '',
        '| Epoch | Dev Hit@10 | Dev Hit@50 | Dev Hit@100 | Dev Hit@500 |',
        '|---:|---:|---:|---:|---:|',
    ])
    for row in history:
        hits = row['dev_hit_counts']
        lines.append(f"| {row['epoch']} | {hits['hit@10']} | {hits['hit@50']} | {hits['hit@100']} | {hits['hit@500']} |")
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(json.dumps({'summary': summary_path, 'markdown': md_path, 'per_user': rows_path}, indent=2))


if __name__ == '__main__':
    main()
