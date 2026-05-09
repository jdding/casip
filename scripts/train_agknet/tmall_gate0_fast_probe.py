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


ACTION_NAMES = {0: 'click', 1: 'cart', 2: 'purchase', 3: 'favorite'}
PROXY_ACTIONS = {0, 1, 3}
PURCHASE_ACTION = 2
TARGET_LEVELS = ('item', 'merchant', 'category', 'brand')


def parse_windows(text):
    windows = []
    for spec in text.split(','):
        name, start, end = spec.strip().split(':')
        windows.append((name, int(start), int(end)))
    return windows


def find_file(input_dir, file_name):
    if os.path.isfile(input_dir):
        return input_dir if os.path.basename(input_dir) == file_name else None
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


def target_col(level):
    return {
        'item': 'item_id',
        'merchant': 'seller_id',
        'category': 'cat_id',
        'brand': 'brand_id',
    }[level]


def safe_set(series):
    return set(series.dropna().astype(str).unique().tolist())


def update_user_counts(user_counts, grouped, prefix):
    for row in grouped.itertuples(index=False):
        user_id = str(row.user_id)
        user_counts[user_id][f'{prefix}_all'] += int(row.n_all)
        user_counts[user_id][f'{prefix}_proxy'] += int(row.n_proxy)
        user_counts[user_id][f'{prefix}_purchase'] += int(row.n_purchase)
        user_counts[user_id][f'{prefix}_cart'] += int(row.n_cart)
        user_counts[user_id][f'{prefix}_favorite'] += int(row.n_favorite)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/tmall')
    parser.add_argument('--user-log', default='')
    parser.add_argument('--train-labels', default='')
    parser.add_argument('--test-labels', default='')
    parser.add_argument('--output-dir', default='results/20260506_tmall_gate0_fast_probe')
    parser.add_argument('--chunksize', type=int, default=2_000_000)
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--windows', default='pre:501:1001,gap:1001:1101,val:1101:1111,test:1111:1112')
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    user_log = args.user_log or find_file(args.input_dir, 'user_log_format1.csv')
    train_labels = args.train_labels or find_file(args.input_dir, 'train_format1.csv')
    test_labels = args.test_labels or find_file(args.input_dir, 'test_format1.csv')
    if not user_log:
        raise FileNotFoundError('Missing user_log_format1.csv')

    windows = parse_windows(args.windows)
    usecols = ['user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id', 'time_stamp', 'action_type']
    user_counts = defaultdict(Counter)
    action_counts = Counter()
    window_action_counts = Counter()
    window_target_sets = defaultdict(set)
    window_purchase_target_sets = defaultdict(set)
    rows_seen = 0
    rows_used = 0

    for chunk in pd.read_csv(user_log, usecols=usecols, chunksize=args.chunksize):
        if args.max_rows and rows_seen >= args.max_rows:
            break
        if args.max_rows:
            chunk = chunk.head(max(args.max_rows - rows_seen, 0))
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
        action_counts.update(ACTION_NAMES.get(x, f'unknown_{x}') for x in chunk['action_type'].tolist())
        for (window, action_type), n in chunk.groupby(['window', 'action_type']).size().items():
            window_action_counts[(window, ACTION_NAMES.get(int(action_type), str(action_type)))] += int(n)
        for window, wdf in chunk.groupby('window', sort=False):
            grouped = wdf.groupby('user_id', sort=False)[['n_all', 'n_proxy', 'n_purchase', 'n_cart', 'n_favorite']].sum().reset_index()
            update_user_counts(user_counts, grouped, window)
            purchase = wdf[wdf['action_type'] == PURCHASE_ACTION]
            for level in TARGET_LEVELS:
                col = target_col(level)
                window_target_sets[(window, level)].update(safe_set(wdf[col]))
                window_purchase_target_sets[(window, level)].update(safe_set(purchase[col]))

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
    if users.empty:
        raise RuntimeError('No users found in configured windows')

    gate_specs = OrderedDict([
        ('pre_any_gap_silent_val_proxy_test_purchase',
         (users['pre_all'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1)),
        ('pre_any_gap_low3_val_proxy_test_purchase',
         (users['pre_all'] >= 1) & (users['gap_all'] <= 3) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1)),
        ('pre_purchase_gap_silent_val_proxy_test_purchase',
         (users['pre_purchase'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1)),
        ('pre_any_gap_silent_val_cartfav_test_purchase',
         (users['pre_all'] >= 1) & (users['gap_all'] == 0) & ((users['val_cart'] + users['val_favorite']) >= 1) & (users['test_purchase'] >= 1)),
    ])
    gate_rows = []
    gate_users = {}
    for name, mask in gate_specs.items():
        subset = users[mask]
        gate_users[name] = set(subset['user_id'].astype(str).tolist())
        gate_rows.append(OrderedDict([
            ('gate', name),
            ('users', int(len(subset))),
            ('mean_pre_all', float(subset['pre_all'].mean()) if len(subset) else 0.0),
            ('mean_gap_all', float(subset['gap_all'].mean()) if len(subset) else 0.0),
            ('mean_val_proxy', float(subset['val_proxy'].mean()) if len(subset) else 0.0),
            ('mean_test_purchase', float(subset['test_purchase'].mean()) if len(subset) else 0.0),
        ]))

    needed_users = set().union(*gate_users.values()) if gate_users else set()
    per_gate_targets = defaultdict(set)
    per_gate_history_targets = defaultdict(set)
    if needed_users:
        for chunk in pd.read_csv(user_log, usecols=usecols, chunksize=args.chunksize):
            chunk = chunk[chunk['user_id'].astype(str).isin(needed_users)].copy()
            if chunk.empty:
                continue
            chunk['window'] = assign_windows(chunk['time_stamp'], windows)
            chunk = chunk[chunk['window'].notna()].copy()
            if chunk.empty:
                continue
            chunk['action_type'] = pd.to_numeric(chunk['action_type'], errors='coerce').fillna(-1).astype(int)
            for gate, users_for_gate in gate_users.items():
                gdf = chunk[chunk['user_id'].astype(str).isin(users_for_gate)]
                if gdf.empty:
                    continue
                history = gdf[gdf['window'].isin(['pre', 'val'])]
                test_purchase = gdf[(gdf['window'] == 'test') & (gdf['action_type'] == PURCHASE_ACTION)]
                for level in TARGET_LEVELS:
                    col = target_col(level)
                    per_gate_history_targets[(gate, level)].update(safe_set(history[col]))
                    per_gate_targets[(gate, level)].update(safe_set(test_purchase[col]))

    target_gate_rows = []
    for gate, users_for_gate in gate_users.items():
        for level in TARGET_LEVELS:
            targets = per_gate_targets[(gate, level)]
            history = per_gate_history_targets[(gate, level)]
            target_gate_rows.append(OrderedDict([
                ('gate', gate),
                ('level', level),
                ('users', len(users_for_gate)),
                ('test_purchase_targets', len(targets)),
                ('seen_pre_or_val_targets', len(history)),
                ('test_target_oov_vs_user_history', len(targets - history)),
                ('test_target_oov_rate_vs_user_history', float(len(targets - history) / max(len(targets), 1))),
            ]))

    turnover_rows = []
    for left, right in [('pre', 'val'), ('pre', 'test'), ('val', 'test')]:
        for level in TARGET_LEVELS:
            left_targets = window_purchase_target_sets[(left, level)] or window_target_sets[(left, level)]
            right_targets = window_purchase_target_sets[(right, level)] or window_target_sets[(right, level)]
            overlap = len(left_targets & right_targets)
            union = len(left_targets | right_targets)
            turnover_rows.append(OrderedDict([
                ('pair', f'{left}_to_{right}'),
                ('level', level),
                ('left_targets', len(left_targets)),
                ('right_targets', len(right_targets)),
                ('overlap', overlap),
                ('jaccard', float(overlap / max(union, 1))),
                ('right_new_vs_left', len(right_targets - left_targets)),
                ('right_new_rate', float(len(right_targets - left_targets) / max(len(right_targets), 1))),
            ]))

    label_rows = OrderedDict()
    for name, path in [('train', train_labels), ('test', test_labels)]:
        if path:
            df = pd.read_csv(path)
            label_rows[f'{name}_rows'] = int(len(df))
            if 'label' in df.columns:
                label_rows[f'{name}_positive_labels'] = int((df['label'] == 1).sum())
            if 'merchant_id' in df.columns:
                label_rows[f'{name}_merchants'] = int(df['merchant_id'].nunique())
            if 'user_id' in df.columns:
                label_rows[f'{name}_users'] = int(df['user_id'].nunique())

    payload = OrderedDict([
        ('args', vars(args)),
        ('inputs', OrderedDict([('user_log', user_log), ('train_labels', train_labels), ('test_labels', test_labels)])),
        ('provenance', OrderedDict([('cwd', os.getcwd()), ('argv', sys.argv), ('hostname', platform.node()), ('python', sys.version.split()[0])])),
        ('windows', [{'name': n, 'start': s, 'end': e} for n, s, e in windows]),
        ('action_mapping', ACTION_NAMES),
        ('rows_seen', rows_seen),
        ('rows_used_in_windows', rows_used),
        ('label_rows', label_rows),
        ('action_counts', OrderedDict(sorted(action_counts.items()))),
        ('window_action_counts', OrderedDict((f'{w}:{a}', n) for (w, a), n in sorted(window_action_counts.items()))),
        ('unique_users_in_windows', int(len(users))),
        ('gate_counts', gate_rows),
        ('target_level_gate_counts', target_gate_rows),
        ('target_turnover', turnover_rows),
        ('runtime_seconds', float(time.time() - started)),
    ])
    out_json = os.path.join(args.output_dir, 'tmall_gate0_fast_summary.json')
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(gate_rows).to_csv(os.path.join(args.output_dir, 'tmall_gate0_gate_counts.csv'), index=False)
    pd.DataFrame(target_gate_rows).to_csv(os.path.join(args.output_dir, 'tmall_gate0_target_level_counts.csv'), index=False)
    pd.DataFrame(turnover_rows).to_csv(os.path.join(args.output_dir, 'tmall_gate0_turnover.csv'), index=False)

    lines = [
        '# Tmall Gate-0 Fast Probe',
        '',
        f'- User log: `{user_log}`',
        f'- Train labels: `{train_labels or "MISSING"}`',
        f'- Test labels: `{test_labels or "MISSING"}`',
        f'- Rows seen: `{rows_seen}`',
        f'- Rows used in windows: `{rows_used}`',
        f'- Unique users in windows: `{len(users)}`',
        '',
        '## Gate Counts',
        '',
        '| Gate | Users | Mean pre | Mean gap | Mean val proxy | Mean test purchase |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in gate_rows:
        lines.append(f"| {row['gate']} | {row['users']} | {row['mean_pre_all']:.2f} | {row['mean_gap_all']:.2f} | {row['mean_val_proxy']:.2f} | {row['mean_test_purchase']:.2f} |")
    lines.extend(['', '## Target-Level Gate Counts', '', '| Gate | Level | Users | Test purchase targets | OOV vs user history | OOV rate |', '|---|---|---:|---:|---:|---:|'])
    for row in target_gate_rows:
        lines.append(f"| {row['gate']} | {row['level']} | {row['users']} | {row['test_purchase_targets']} | {row['test_target_oov_vs_user_history']} | {row['test_target_oov_rate_vs_user_history']:.3f} |")
    lines.extend(['', '## Target Turnover', '', '| Pair | Level | Left targets | Right targets | Overlap | Jaccard | Right-new rate |', '|---|---|---:|---:|---:|---:|---:|'])
    for row in turnover_rows:
        lines.append(f"| {row['pair']} | {row['level']} | {row['left_targets']} | {row['right_targets']} | {row['overlap']} | {row['jaccard']:.4f} | {row['right_new_rate']:.3f} |")
    with open(os.path.join(args.output_dir, 'tmall_gate0_fast_summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(out_json)


if __name__ == '__main__':
    main()
