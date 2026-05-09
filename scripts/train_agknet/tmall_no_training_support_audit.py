#!/usr/bin/env python3
import argparse
import glob
import json
import os
import platform
import sys
import time
from collections import Counter, OrderedDict, defaultdict

import pandas as pd


PROXY_ACTIONS = {0, 1, 3}
PURCHASE_ACTION = 2
KS = (10, 50, 100, 500)


def parse_windows(text):
    return [(n, int(s), int(e)) for n, s, e in (x.split(':') for x in text.split(','))]


def find_file(input_dir, file_name):
    candidates = []
    for root, _, _ in os.walk(input_dir):
        candidates.extend(glob.glob(os.path.join(root, file_name)))
    return sorted(candidates)[0] if candidates else None


def assign_windows(values, windows):
    ts = pd.to_numeric(values, errors='coerce').fillna(-1).astype(int)
    out = pd.Series(pd.NA, index=values.index, dtype='object')
    for name, start, end in windows:
        out.loc[(ts >= start) & (ts < end)] = name
    return out


def action_weight(action_type):
    if action_type == PURCHASE_ACTION:
        return 4.0
    if action_type == 1:
        return 2.0
    if action_type == 3:
        return 1.5
    if action_type == 0:
        return 1.0
    return 0.0


def update_counter_from_group(counter, grouped, key_cols):
    for row in grouped.itertuples(index=False):
        key = tuple(str(getattr(row, c)) for c in key_cols)
        if len(key) == 1:
            key = key[0]
        counter[key] += float(row.score)


def top_items(counter, n):
    return [str(item) for item, _ in counter.most_common(n)]


def rank_user_counter(counter, global_scores, n):
    return [
        item for item, _ in sorted(
            counter.items(),
            key=lambda kv: (-kv[1], -global_scores.get(kv[0], 0.0), kv[0]),
        )[:n]
    ]


def hit_vector(candidates, targets):
    out = {}
    for k in KS:
        out[k] = int(bool(set(candidates[:k]) & targets))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/tmall')
    parser.add_argument('--output-dir', default='results/20260506_tmall_support_audit')
    parser.add_argument('--chunksize', type=int, default=1_000_000)
    parser.add_argument('--windows', default='pre:501:1001,gap:1001:1101,val:1101:1111,test:1111:1112')
    parser.add_argument('--gate', default='pre_any_gap_silent_val_proxy_test_purchase')
    parser.add_argument('--source-topk-per-bucket', type=int, default=200)
    parser.add_argument('--max-users', type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    user_log = find_file(args.input_dir, 'user_log_format1.csv')
    if not user_log:
        raise FileNotFoundError('Missing user_log_format1.csv')

    windows = parse_windows(args.windows)
    usecols = ['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type']
    user_counts = defaultdict(Counter)
    global_item_scores = Counter()
    cat_item_scores = defaultdict(Counter)
    brand_item_scores = defaultdict(Counter)
    cat_brand_item_scores = defaultdict(Counter)
    merchant_item_scores = defaultdict(Counter)
    rows_seen = 0
    rows_used = 0

    for chunk in pd.read_csv(user_log, usecols=usecols, chunksize=args.chunksize):
        rows_seen += len(chunk)
        chunk['window'] = assign_windows(chunk['time_stamp'], windows)
        chunk = chunk[chunk['window'].notna()].copy()
        if chunk.empty:
            continue
        rows_used += len(chunk)
        chunk['action_type'] = pd.to_numeric(chunk['action_type'], errors='coerce').fillna(-1).astype(int)
        chunk['n_proxy'] = chunk['action_type'].isin(PROXY_ACTIONS).astype(int)
        chunk['n_purchase'] = (chunk['action_type'] == PURCHASE_ACTION).astype(int)
        chunk['n_cart'] = (chunk['action_type'] == 1).astype(int)
        chunk['n_favorite'] = (chunk['action_type'] == 3).astype(int)
        chunk['n_all'] = 1
        for window, wdf in chunk.groupby('window', sort=False):
            grouped = wdf.groupby('user_id', sort=False)[['n_all', 'n_proxy', 'n_purchase', 'n_cart', 'n_favorite']].sum().reset_index()
            for row in grouped.itertuples(index=False):
                user_id = str(row.user_id)
                user_counts[user_id][f'{window}_all'] += int(row.n_all)
                user_counts[user_id][f'{window}_proxy'] += int(row.n_proxy)
                user_counts[user_id][f'{window}_purchase'] += int(row.n_purchase)
                user_counts[user_id][f'{window}_cart'] += int(row.n_cart)
                user_counts[user_id][f'{window}_favorite'] += int(row.n_favorite)
        deployable = chunk[chunk['window'].isin(['pre', 'val'])].copy()
        if deployable.empty:
            continue
        deployable['score'] = deployable['action_type'].map(action_weight)
        update_counter_from_group(global_item_scores, deployable.groupby('item_id', sort=False)['score'].sum().reset_index(), ['item_id'])
        for (cat_id, item_id), score in deployable.groupby(['cat_id', 'item_id'], sort=False)['score'].sum().items():
            cat_item_scores[str(cat_id)][str(item_id)] += float(score)
        for (brand_id, item_id), score in deployable.groupby(['brand_id', 'item_id'], sort=False)['score'].sum().items():
            brand_item_scores[str(brand_id)][str(item_id)] += float(score)
        for (cat_id, brand_id, item_id), score in deployable.groupby(['cat_id', 'brand_id', 'item_id'], sort=False)['score'].sum().items():
            cat_brand_item_scores[(str(cat_id), str(brand_id))][str(item_id)] += float(score)
        for (seller_id, item_id), score in deployable.groupby(['seller_id', 'item_id'], sort=False)['score'].sum().items():
            merchant_item_scores[str(seller_id)][str(item_id)] += float(score)

    users = pd.DataFrame([
        {
            'user_id': user_id,
            **{
                f'{window}_{key}': counts.get(f'{window}_{key}', 0)
                for window, _, _ in windows
                for key in ('all', 'proxy', 'purchase', 'cart', 'favorite')
            },
        }
        for user_id, counts in user_counts.items()
    ])
    gate_masks = {
        'pre_any_gap_silent_val_proxy_test_purchase':
            (users['pre_all'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1),
        'pre_purchase_gap_silent_val_proxy_test_purchase':
            (users['pre_purchase'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1),
        'pre_any_gap_silent_val_cartfav_test_purchase':
            (users['pre_all'] >= 1) & (users['gap_all'] == 0) & ((users['val_cart'] + users['val_favorite']) >= 1) & (users['test_purchase'] >= 1),
    }
    if args.gate not in gate_masks:
        raise ValueError(f'Unknown gate {args.gate}')
    gate_users = users[gate_masks[args.gate]]['user_id'].astype(str).tolist()
    if args.max_users:
        gate_users = gate_users[:args.max_users]
    gate_user_set = set(gate_users)

    user_targets = defaultdict(set)
    user_existing = defaultdict(Counter)
    user_cats = defaultdict(set)
    user_brands = defaultdict(set)
    user_cat_brands = defaultdict(set)
    user_merchants = defaultdict(set)

    for chunk in pd.read_csv(user_log, usecols=usecols, chunksize=args.chunksize):
        chunk['user_id_str'] = chunk['user_id'].astype(str)
        chunk = chunk[chunk['user_id_str'].isin(gate_user_set)].copy()
        if chunk.empty:
            continue
        chunk['window'] = assign_windows(chunk['time_stamp'], windows)
        chunk = chunk[chunk['window'].notna()].copy()
        if chunk.empty:
            continue
        chunk['action_type'] = pd.to_numeric(chunk['action_type'], errors='coerce').fillna(-1).astype(int)
        deployable = chunk[chunk['window'].isin(['pre', 'val'])]
        for row in deployable.itertuples(index=False):
            user_id = row.user_id_str
            item = str(row.item_id)
            score = action_weight(int(row.action_type))
            user_existing[user_id][item] += score
            if row.window == 'val' and int(row.action_type) in PROXY_ACTIONS:
                user_cats[user_id].add(str(row.cat_id))
                user_brands[user_id].add(str(row.brand_id))
                user_cat_brands[user_id].add((str(row.cat_id), str(row.brand_id)))
                user_merchants[user_id].add(str(row.seller_id))
        test_purchase = chunk[(chunk['window'] == 'test') & (chunk['action_type'] == PURCHASE_ACTION)]
        for row in test_purchase.itertuples(index=False):
            user_targets[row.user_id_str].add(str(row.item_id))

    global_top = top_items(global_item_scores, max(KS))
    bucket_cache = {}

    def bucket_candidates(source_name, keys, score_map):
        scores = Counter()
        for key in sorted(keys):
            cache_key = (source_name, key)
            if cache_key not in bucket_cache:
                bucket_cache[cache_key] = top_items(score_map.get(key, Counter()), args.source_topk_per_bucket)
            for rank, item in enumerate(bucket_cache[cache_key]):
                scores[item] += args.source_topk_per_bucket - rank
        return [item for item, _ in scores.most_common(max(KS))]

    source_hits = {name: Counter() for name in [
        'existing_user_replay',
        'global_deployable_popularity',
        'semantic_category_val_proxy',
        'semantic_brand_val_proxy',
        'semantic_category_brand_val_proxy',
        'semantic_merchant_val_proxy',
        'semantic_union_oracle',
        'existing_plus_semantic_oracle',
        'existing_plus_semantic_all',
    ]}
    existing_miss_recovered = Counter()
    users_eval = 0

    for user_id in gate_users:
        targets = user_targets.get(user_id, set())
        if not targets:
            continue
        users_eval += 1
        existing = rank_user_counter(user_existing[user_id], global_item_scores, max(KS))
        source_lists = {
            'existing_user_replay': existing,
            'global_deployable_popularity': global_top,
            'semantic_category_val_proxy': bucket_candidates('cat', user_cats[user_id], cat_item_scores),
            'semantic_brand_val_proxy': bucket_candidates('brand', user_brands[user_id], brand_item_scores),
            'semantic_category_brand_val_proxy': bucket_candidates('cat_brand', user_cat_brands[user_id], cat_brand_item_scores),
            'semantic_merchant_val_proxy': bucket_candidates('merchant', user_merchants[user_id], merchant_item_scores),
        }
        combined_scores = Counter()
        for source in ('existing_user_replay', 'semantic_category_val_proxy', 'semantic_brand_val_proxy', 'semantic_category_brand_val_proxy', 'semantic_merchant_val_proxy'):
            for rank, item in enumerate(source_lists[source]):
                combined_scores[item] += max(KS) - rank
        source_lists['existing_plus_semantic_all'] = [item for item, _ in combined_scores.most_common(max(KS))]
        existing_hit = hit_vector(existing, targets)
        semantic_sources = [
            'semantic_category_val_proxy',
            'semantic_brand_val_proxy',
            'semantic_category_brand_val_proxy',
            'semantic_merchant_val_proxy',
        ]
        for source, candidates in source_lists.items():
            hits = hit_vector(candidates, targets)
            for k, v in hits.items():
                source_hits[source][k] += v
                if source.startswith('semantic') and v and not existing_hit[k]:
                    existing_miss_recovered[(source, k)] += 1
        for k in KS:
            semantic_oracle_hit = any(set(source_lists[source][:k]) & targets for source in semantic_sources)
            existing_plus_oracle_hit = existing_hit[k] or semantic_oracle_hit
            source_hits['semantic_union_oracle'][k] += int(semantic_oracle_hit)
            source_hits['existing_plus_semantic_oracle'][k] += int(existing_plus_oracle_hit)

    rows = []
    for source, counts in source_hits.items():
        row = OrderedDict([('source', source)])
        for k in KS:
            row[f'hit@{k}'] = int(counts[k])
        rows.append(row)

    payload = OrderedDict([
        ('args', vars(args)),
        ('provenance', OrderedDict([('cwd', os.getcwd()), ('argv', sys.argv), ('hostname', platform.node()), ('python', sys.version.split()[0])])),
        ('rows_seen', rows_seen),
        ('rows_used_in_windows', rows_used),
        ('gate_users', len(gate_users)),
        ('users_eval', users_eval),
        ('source_hits', rows),
        ('existing_miss_recovered', OrderedDict((f'{s}@{k}', int(v)) for (s, k), v in sorted(existing_miss_recovered.items()))),
        ('runtime_seconds', float(time.time() - started)),
    ])
    out_json = os.path.join(args.output_dir, 'tmall_support_audit_summary.json')
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'tmall_support_audit_summary.csv'), index=False)

    lines = [
        '# Tmall No-Training Support Audit',
        '',
        f'- Gate: `{args.gate}`',
        f'- Gate users: `{len(gate_users)}`',
        f'- Evaluated users with test purchases: `{users_eval}`',
        '',
        '## Source Hits',
        '',
        '| Source | Hit@10 | Hit@50 | Hit@100 | Hit@500 |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f"| {row['source']} | {row['hit@10']} | {row['hit@50']} | {row['hit@100']} | {row['hit@500']} |")
    lines.extend(['', '## Existing Miss Recovery', '', '| Source@K | Recovered existing misses |', '|---|---:|'])
    for key, value in payload['existing_miss_recovered'].items():
        lines.append(f'| {key} | {value} |')
    with open(os.path.join(args.output_dir, 'tmall_support_audit_summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(out_json)


if __name__ == '__main__':
    main()
