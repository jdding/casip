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


def hit(candidates, targets, k):
    return int(bool(set(candidates[:k]) & targets))


def rank_counter(counter, global_scores, n):
    return [
        item for item, _ in sorted(
            counter.items(),
            key=lambda kv: (-kv[1], -global_scores.get(kv[0], 0.0), kv[0]),
        )[:n]
    ]


def top_items(counter, n):
    return [str(item) for item, _ in counter.most_common(n)]


def make_promoted(existing, semantic, sem_n):
    protected = list(existing[:10])
    blocked = set(protected)
    inserted = []
    for item in semantic:
        if item not in blocked:
            inserted.append(item)
            blocked.add(item)
        if len(inserted) >= sem_n:
            break
    out = protected + inserted
    for item in existing[10:]:
        if item not in blocked:
            out.append(item)
            blocked.add(item)
        if len(out) >= max(KS):
            break
    return out[:max(KS)]


def build_eval_state(user_log, windows, chunksize, gate, source_topk_per_bucket):
    usecols = ['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type']
    user_counts = defaultdict(Counter)
    global_item_scores = Counter()
    cat_item_scores = defaultdict(Counter)
    brand_item_scores = defaultdict(Counter)
    cat_brand_item_scores = defaultdict(Counter)

    for chunk in pd.read_csv(user_log, usecols=usecols, chunksize=chunksize):
        chunk['window'] = assign_windows(chunk['time_stamp'], windows)
        chunk = chunk[chunk['window'].notna()].copy()
        if chunk.empty:
            continue
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
        for item_id, score in deployable.groupby('item_id', sort=False)['score'].sum().items():
            global_item_scores[str(item_id)] += float(score)
        for (cat_id, item_id), score in deployable.groupby(['cat_id', 'item_id'], sort=False)['score'].sum().items():
            cat_item_scores[str(cat_id)][str(item_id)] += float(score)
        for (brand_id, item_id), score in deployable.groupby(['brand_id', 'item_id'], sort=False)['score'].sum().items():
            brand_item_scores[str(brand_id)][str(item_id)] += float(score)
        for (cat_id, brand_id, item_id), score in deployable.groupby(['cat_id', 'brand_id', 'item_id'], sort=False)['score'].sum().items():
            cat_brand_item_scores[(str(cat_id), str(brand_id))][str(item_id)] += float(score)

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
    masks = {
        'pre_any_gap_silent_val_proxy_test_purchase':
            (users['pre_all'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1),
        'pre_purchase_gap_silent_val_proxy_test_purchase':
            (users['pre_purchase'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1),
    }
    gate_users = users[masks[gate]]['user_id'].astype(str).tolist()
    gate_user_set = set(gate_users)

    bucket_cache = {}
    def bucket_list(kind, key):
        cache_key = (kind, key)
        if cache_key in bucket_cache:
            return bucket_cache[cache_key]
        source = {'cat': cat_item_scores, 'brand': brand_item_scores, 'cat_brand': cat_brand_item_scores}[kind]
        bucket_cache[cache_key] = top_items(source.get(key, Counter()), source_topk_per_bucket)
        return bucket_cache[cache_key]

    rows = []
    user_existing = defaultdict(Counter)
    user_targets = defaultdict(set)
    user_pre_items = defaultdict(set)
    user_history_items = defaultdict(set)
    user_cats = defaultdict(set)
    user_brands = defaultdict(set)
    user_cat_brands = defaultdict(set)

    for chunk in pd.read_csv(user_log, usecols=usecols, chunksize=chunksize):
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
            user_existing[user_id][item] += action_weight(int(row.action_type))
            user_history_items[user_id].add(item)
            if row.window == 'pre':
                user_pre_items[user_id].add(item)
            if row.window == 'val' and int(row.action_type) in PROXY_ACTIONS:
                user_cats[user_id].add(str(row.cat_id))
                user_brands[user_id].add(str(row.brand_id))
                user_cat_brands[user_id].add((str(row.cat_id), str(row.brand_id)))
        test_purchase = chunk[(chunk['window'] == 'test') & (chunk['action_type'] == PURCHASE_ACTION)]
        for row in test_purchase.itertuples(index=False):
            user_targets[row.user_id_str].add(str(row.item_id))

    for user_id in gate_users:
        targets = user_targets.get(user_id, set())
        if not targets:
            continue
        existing = rank_counter(user_existing[user_id], global_item_scores, max(KS))
        sem_scores = {
            'cat': Counter(),
            'brand': Counter(),
            'cat_brand': Counter(),
        }
        for cat in sorted(user_cats[user_id]):
            for rank, item in enumerate(bucket_list('cat', cat)):
                sem_scores['cat'][item] += source_topk_per_bucket - rank
        for brand in sorted(user_brands[user_id]):
            for rank, item in enumerate(bucket_list('brand', brand)):
                sem_scores['brand'][item] += source_topk_per_bucket - rank
        for cat_brand in sorted(user_cat_brands[user_id]):
            for rank, item in enumerate(bucket_list('cat_brand', cat_brand)):
                sem_scores['cat_brand'][item] += source_topk_per_bucket - rank
        semantic = {name: [item for item, _ in scores.most_common(max(KS))] for name, scores in sem_scores.items()}
        rows.append({
            'user_id': user_id,
            'targets': targets,
            'existing': existing,
            'semantic': semantic,
            'pre_items': set(user_pre_items[user_id]),
            'history_items': set(user_history_items[user_id]),
            'features': {
                'n_cat_brand': len(user_cat_brands[user_id]),
                'n_brand': len(user_brands[user_id]),
                'n_cat': len(user_cats[user_id]),
            },
        })
    return rows


def evaluate_policies(rows, policy_specs):
    metrics = []
    for source, sem_n, gate_name, gate_fn in policy_specs:
        counts = Counter()
        for row in rows:
            existing = row['existing']
            targets = row['targets']
            if gate_fn(row['features']):
                promoted = make_promoted(existing, row['semantic'][source], sem_n)
                opened = 1
            else:
                promoted = existing
                opened = 0
            counts['open'] += opened
            for k in KS:
                base = hit(existing, targets, k)
                new = hit(promoted, targets, k)
                counts[f'base@{k}'] += base
                counts[f'hit@{k}'] += new
                counts[f'gross@{k}'] += int(new and not base)
                counts[f'cannibal@{k}'] += int(base and not new)
        n = max(len(rows), 1)
        row = OrderedDict([
            ('policy', f'{source}__sem{sem_n}__{gate_name}'),
            ('source', source),
            ('sem_n', sem_n),
            ('gate', gate_name),
            ('open_rate', counts['open'] / n),
        ])
        for k in KS:
            gross = counts[f'gross@{k}']
            cann = counts[f'cannibal@{k}']
            row[f'base@{k}'] = counts[f'base@{k}']
            row[f'hit@{k}'] = counts[f'hit@{k}']
            row[f'gross@{k}'] = gross
            row[f'cannibal@{k}'] = cann
            row[f'net@{k}'] = gross - cann
            row[f'ratio@{k}'] = cann / gross if gross else None
        metrics.append(row)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/tmall')
    parser.add_argument('--output-dir', default='results/20260506_tmall_casp_policy_probe')
    parser.add_argument('--chunksize', type=int, default=1_000_000)
    parser.add_argument('--validation-windows', default='pre:501:1001,gap:1001:1101,val:1101:1106,test:1106:1111')
    parser.add_argument('--test-windows', default='pre:501:1001,gap:1001:1101,val:1101:1111,test:1111:1112')
    parser.add_argument('--gate', default='pre_any_gap_silent_val_proxy_test_purchase')
    parser.add_argument('--min-validation-net', type=int, default=5)
    parser.add_argument('--min-validation-net50', type=int, default=0)
    parser.add_argument('--max-validation-ratio', type=float, default=0.5)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    user_log = find_file(args.input_dir, 'user_log_format1.csv')
    if not user_log:
        raise FileNotFoundError('Missing user_log_format1.csv')

    gate_fns = [('all', lambda f: True)]
    for key in ('n_cat_brand', 'n_brand', 'n_cat'):
        for threshold in (1, 2, 3, 5, 10, 20):
            gate_fns.append((f'{key}_le_{threshold}', lambda f, key=key, threshold=threshold: f[key] <= threshold))
    policy_specs = [
        (source, sem_n, gate_name, gate_fn)
        for source in ('cat_brand', 'brand', 'cat')
        for sem_n in (10, 25, 50, 100)
        for gate_name, gate_fn in gate_fns
    ]

    validation_rows = build_eval_state(user_log, parse_windows(args.validation_windows), args.chunksize, args.gate, 200)
    val_metrics = evaluate_policies(validation_rows, policy_specs)
    feasible = [
        row for row in val_metrics
        if row['hit@10'] >= row['base@10']
        and row['net@50'] >= args.min_validation_net50
        and row['net@100'] >= args.min_validation_net
        and (row['ratio@100'] is None or row['ratio@100'] <= args.max_validation_ratio)
    ]
    if feasible:
        selected = sorted(feasible, key=lambda r: (r['net@100'], -r['ratio@100'] if r['ratio@100'] is not None else 0, -r['open_rate']), reverse=True)[0]
    else:
        selected = sorted(val_metrics, key=lambda r: (r['net@100'], r['hit@100']), reverse=True)[0]

    selected_spec = None
    for spec in policy_specs:
        if f'{spec[0]}__sem{spec[1]}__{spec[2]}' == selected['policy']:
            selected_spec = spec
            break
    test_rows = build_eval_state(user_log, parse_windows(args.test_windows), args.chunksize, args.gate, 200)
    test_metrics = evaluate_policies(test_rows, [selected_spec])
    selected_test = test_metrics[0]

    pd.DataFrame(val_metrics).to_csv(os.path.join(args.output_dir, 'tmall_casp_validation_grid.csv'), index=False)
    pd.DataFrame([selected]).to_csv(os.path.join(args.output_dir, 'tmall_casp_selected_validation.csv'), index=False)
    pd.DataFrame([selected_test]).to_csv(os.path.join(args.output_dir, 'tmall_casp_selected_test.csv'), index=False)
    payload = OrderedDict([
        ('args', vars(args)),
        ('provenance', OrderedDict([('cwd', os.getcwd()), ('argv', sys.argv), ('hostname', platform.node()), ('python', sys.version.split()[0])])),
        ('n_validation_users', len(validation_rows)),
        ('n_test_users', len(test_rows)),
        ('selected_validation', selected),
        ('selected_test', selected_test),
        ('feasible_validation_policies', len(feasible)),
        ('runtime_seconds', time.time() - started),
    ])
    with open(os.path.join(args.output_dir, 'tmall_casp_policy_summary.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    lines = [
        '# Tmall CASP Policy Probe',
        '',
        f"- Validation users: `{len(validation_rows)}`",
        f"- Test users: `{len(test_rows)}`",
        f"- Feasible validation policies: `{len(feasible)}`",
        f"- Selected policy: `{selected['policy']}`",
        '',
        '## Selected Validation',
        '',
        '| Hit@10 | Hit@100 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 | Open |',
        '|---:|---:|---:|---:|---:|---:|---:|',
        f"| {selected['hit@10']} | {selected['hit@100']} | {selected['gross@100']} | {selected['cannibal@100']} | {selected['net@100']} | {selected['ratio@100']} | {selected['open_rate']:.3f} |",
        '',
        '## Selected Test',
        '',
        '| Hit@10 | Hit@100 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 | Open |',
        '|---:|---:|---:|---:|---:|---:|---:|',
        f"| {selected_test['hit@10']} | {selected_test['hit@100']} | {selected_test['gross@100']} | {selected_test['cannibal@100']} | {selected_test['net@100']} | {selected_test['ratio@100']} | {selected_test['open_rate']:.3f} |",
    ]
    with open(os.path.join(args.output_dir, 'tmall_casp_policy_summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(os.path.join(args.output_dir, 'tmall_casp_policy_summary.json'))


if __name__ == '__main__':
    main()
