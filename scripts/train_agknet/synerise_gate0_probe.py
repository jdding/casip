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


EVENT_FILES = OrderedDict([
    ('product_buy', 'product_buy.parquet'),
    ('add_to_cart', 'add_to_cart.parquet'),
    ('remove_from_cart', 'remove_from_cart.parquet'),
    ('page_visit', 'page_visit.parquet'),
    ('search_query', 'search_query.parquet'),
])

ITEM_EVENTS = {'product_buy', 'add_to_cart', 'remove_from_cart'}
PROXY_EVENTS = {'add_to_cart', 'remove_from_cart', 'page_visit', 'search_query'}
PURCHASE_EVENT = 'product_buy'
TARGET_LEVELS = ('sku', 'category', 'price')


def find_file(input_dir, file_name):
    if os.path.isfile(input_dir):
        return input_dir if os.path.basename(input_dir) == file_name else None
    candidates = []
    for root, _, _ in os.walk(input_dir):
        candidates.extend(glob.glob(os.path.join(root, file_name)))
    return sorted(candidates)[0] if candidates else None


def parse_windows(text):
    windows = []
    for spec in text.split(','):
        spec = spec.strip()
        if not spec:
            continue
        name, start, end = spec.split(':', 2)
        windows.append((name, pd.Timestamp(start), pd.Timestamp(end)))
    return windows


def assign_windows(values, windows):
    ts = pd.to_datetime(values, errors='coerce')
    out = pd.Series(pd.NA, index=values.index, dtype='object')
    for name, start, end in windows:
        mask = (ts >= start) & (ts < end)
        out.loc[mask] = name
    return out


def iter_parquet_batches(path, columns, batch_size):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        df = pd.read_parquet(path, columns=columns)
        yield df
        return

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def load_product_maps(path):
    if not path:
        return {}, {}
    df = pd.read_parquet(path, columns=['sku', 'category', 'price'])
    df = df.drop_duplicates('sku')
    category = dict(zip(df['sku'].astype(str), df['category'].astype(str)))
    price = dict(zip(df['sku'].astype(str), df['price'].astype(str)))
    return category, price


def safe_unique(series):
    return set(series.dropna().astype(str).unique().tolist())


def enrich_item_chunk(chunk, category_map, price_map):
    chunk['sku'] = chunk['sku'].astype(str)
    chunk['category'] = chunk['sku'].map(category_map).fillna('UNKNOWN')
    chunk['price'] = chunk['sku'].map(price_map).fillna('UNKNOWN')
    return chunk


def update_count_pass(chunk, event_name, windows, window_names, user_count_frames, event_counts, window_event_counts):
    chunk['window'] = assign_windows(chunk['timestamp'], windows)
    chunk = chunk[chunk['window'].notna()].copy()
    if chunk.empty:
        return 0
    event_counts[event_name] += int(len(chunk))
    for window, n in chunk.groupby('window').size().items():
        window_event_counts[(str(window), event_name)] += int(n)
    grouped = chunk.groupby(['client_id', 'window']).size().rename('n').reset_index()
    if grouped.empty:
        return int(len(chunk))
    grouped['client_id'] = grouped['client_id'].astype(str)
    wide = (
        grouped
        .pivot_table(index='client_id', columns='window', values='n', aggfunc='sum', fill_value=0)
        .reindex(columns=window_names, fill_value=0)
    )
    out = pd.DataFrame(index=wide.index)
    for window in window_names:
        values = wide[window].astype('int64')
        out[f'{window}_all'] = values
        out[f'{window}_{event_name}'] = values
        if event_name in PROXY_EVENTS:
            out[f'{window}_proxy'] = values
        if event_name == PURCHASE_EVENT:
            out[f'{window}_purchase'] = values
    user_count_frames.append(out.reset_index())
    return int(len(chunk))


def collect_target_sets(chunk, event_name, windows, gate_users, category_map, price_map, target_sets):
    if event_name not in ITEM_EVENTS:
        return
    chunk['window'] = assign_windows(chunk['timestamp'], windows)
    chunk = chunk[chunk['window'].notna()].copy()
    if chunk.empty:
        return
    chunk['client_id'] = chunk['client_id'].astype(str)
    chunk = chunk[chunk['client_id'].isin(gate_users)]
    if chunk.empty:
        return
    chunk = enrich_item_chunk(chunk, category_map, price_map)
    for (client_id, window), udf in chunk.groupby(['client_id', 'window'], sort=False):
        bucket = 'test_purchase' if event_name == PURCHASE_EVENT and window == 'test' else 'history'
        if bucket == 'history' and window not in {'pre', 'val'}:
            continue
        for level in TARGET_LEVELS:
            values = safe_unique(udf[level])
            if values:
                target_sets[(client_id, bucket, level)].update(values)


def main():
    parser = argparse.ArgumentParser(description='Synerise RecSys 2025 Protocol C Gate-0 probe.')
    parser.add_argument('--input-dir', default='data/synerise')
    parser.add_argument('--output-dir', default='results/20260507_synerise_gate0_probe')
    parser.add_argument('--batch-size', type=int, default=500_000)
    parser.add_argument('--max-rows-per-file', type=int, default=0, help='Optional smoke-test row cap per event file.')
    parser.add_argument(
        '--windows',
        required=True,
        help=(
            'Comma-separated name:start:end windows over timestamps, e.g. '
            'pre:2024-01-01:2024-04-01,gap:2024-04-01:2024-04-15,'
            'val:2024-04-15:2024-05-15,test:2024-05-15:2024-06-30. '
            'End is exclusive. Required because release timestamps must be verified after download.'
        ),
    )
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    windows = parse_windows(args.windows)
    window_names = [w[0] for w in windows]
    required_names = {'pre', 'gap', 'val', 'test'}
    missing_windows = sorted(required_names - set(window_names))
    if missing_windows:
        raise ValueError(f'Windows must include {sorted(required_names)}; missing {missing_windows}')

    event_paths = OrderedDict((name, find_file(args.input_dir, file_name)) for name, file_name in EVENT_FILES.items())
    missing_events = [name for name, path in event_paths.items() if not path]
    if missing_events:
        raise FileNotFoundError(f'Missing Synerise event files under {args.input_dir}: {missing_events}')
    product_properties = find_file(args.input_dir, 'product_properties.parquet')
    if not product_properties:
        raise FileNotFoundError(f'Missing product_properties.parquet under {args.input_dir}')

    category_map, price_map = load_product_maps(product_properties)

    rows_seen = Counter()
    rows_used = Counter()
    event_counts = Counter()
    window_event_counts = Counter()
    user_count_frames = []

    for event_name, path in event_paths.items():
        columns = ['client_id', 'timestamp']
        if event_name in ITEM_EVENTS:
            columns.append('sku')
        seen_for_file = 0
        for chunk in iter_parquet_batches(path, columns, args.batch_size):
            if args.max_rows_per_file and seen_for_file >= args.max_rows_per_file:
                break
            if args.max_rows_per_file:
                chunk = chunk.head(max(args.max_rows_per_file - seen_for_file, 0))
            seen_for_file += len(chunk)
            rows_seen[event_name] += int(len(chunk))
            rows_used[event_name] += update_count_pass(
                chunk,
                event_name,
                windows,
                window_names,
                user_count_frames,
                event_counts,
                window_event_counts,
            )
        print(
            f"[count-pass] {event_name}: rows_seen={rows_seen[event_name]} rows_used={rows_used[event_name]}",
            file=sys.stderr,
            flush=True,
        )

    if not user_count_frames:
        raise RuntimeError('No users found in configured windows')

    users = pd.concat(user_count_frames, ignore_index=True)
    users = users.groupby('client_id', as_index=False).sum(numeric_only=True)
    for window in window_names:
        for key in ['all', 'proxy', 'purchase'] + list(EVENT_FILES.keys()):
            col = f'{window}_{key}'
            if col not in users.columns:
                users[col] = 0

    gate_specs = OrderedDict([
        ('pre_any_gap_silent_val_proxy_test_buy',
         (users['pre_all'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1)),
        ('pre_buy_gap_silent_val_proxy_test_buy',
         (users['pre_purchase'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1)),
        ('pre_any_gap_low5_val_proxy_test_buy',
         (users['pre_all'] >= 1) & (users['gap_all'] <= 5) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1)),
        ('pre_any_gap_silent_val_cart_search_test_buy',
         (users['pre_all'] >= 1) & (users['gap_all'] == 0)
         & ((users['val_add_to_cart'] + users['val_search_query']) >= 1)
         & (users['test_purchase'] >= 1)),
    ])

    gate_rows = []
    gate_users = {}
    for gate_name, mask in gate_specs.items():
        subset = users[mask]
        gate_users[gate_name] = set(subset['client_id'].astype(str).tolist())
        gate_rows.append(OrderedDict([
            ('gate', gate_name),
            ('users', int(len(subset))),
            ('mean_pre_all', float(subset['pre_all'].mean()) if len(subset) else 0.0),
            ('mean_gap_all', float(subset['gap_all'].mean()) if len(subset) else 0.0),
            ('mean_val_proxy', float(subset['val_proxy'].mean()) if len(subset) else 0.0),
            ('mean_val_add_to_cart', float(subset['val_add_to_cart'].mean()) if len(subset) else 0.0),
            ('mean_val_search_query', float(subset['val_search_query'].mean()) if len(subset) else 0.0),
            ('mean_test_purchase', float(subset['test_purchase'].mean()) if len(subset) else 0.0),
        ]))

    all_gate_users = set().union(*gate_users.values()) if gate_users else set()
    target_sets = defaultdict(set)
    if all_gate_users:
        for event_name, path in event_paths.items():
            columns = ['client_id', 'timestamp']
            if event_name in ITEM_EVENTS:
                columns.append('sku')
            seen_for_file = 0
            for chunk in iter_parquet_batches(path, columns, args.batch_size):
                if args.max_rows_per_file and seen_for_file >= args.max_rows_per_file:
                    break
                if args.max_rows_per_file:
                    chunk = chunk.head(max(args.max_rows_per_file - seen_for_file, 0))
                seen_for_file += len(chunk)
                collect_target_sets(
                    chunk,
                    event_name,
                    windows,
                    all_gate_users,
                    category_map,
                    price_map,
                    target_sets,
                )

    target_rows = []
    for gate_name, users_for_gate in gate_users.items():
        for level in TARGET_LEVELS:
            test_targets = set()
            history_targets = set()
            for client_id in users_for_gate:
                test_targets.update(target_sets.get((client_id, 'test_purchase', level), set()))
                history_targets.update(target_sets.get((client_id, 'history', level), set()))
            target_rows.append(OrderedDict([
                ('gate', gate_name),
                ('level', level),
                ('users', int(len(users_for_gate))),
                ('test_purchase_targets', int(len(test_targets))),
                ('seen_pre_or_val_targets', int(len(history_targets))),
                ('test_target_oov_vs_user_history', int(len(test_targets - history_targets))),
                ('test_target_oov_rate_vs_user_history', float(len(test_targets - history_targets) / max(len(test_targets), 1))),
            ]))

    payload = OrderedDict([
        ('args', vars(args)),
        ('inputs', OrderedDict([
            ('events', event_paths),
            ('product_properties', product_properties),
        ])),
        ('provenance', OrderedDict([
            ('cwd', os.getcwd()),
            ('argv', sys.argv),
            ('hostname', platform.node()),
            ('python', sys.version.split()[0]),
        ])),
        ('windows', [{'name': n, 'start': str(s), 'end': str(e)} for n, s, e in windows]),
        ('rows_seen', OrderedDict((k, int(v)) for k, v in rows_seen.items())),
        ('rows_used_in_windows', OrderedDict((k, int(v)) for k, v in rows_used.items())),
        ('event_counts_in_windows', OrderedDict((k, int(v)) for k, v in event_counts.items())),
        ('window_event_counts', OrderedDict((f'{w}:{a}', int(n)) for (w, a), n in sorted(window_event_counts.items()))),
        ('unique_users_in_windows', int(len(users))),
        ('gate_counts', gate_rows),
        ('target_level_gate_counts', target_rows),
        ('runtime_seconds', float(time.time() - started)),
    ])

    out_json = os.path.join(args.output_dir, 'synerise_gate0_summary.json')
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(gate_rows).to_csv(os.path.join(args.output_dir, 'synerise_gate0_gate_counts.csv'), index=False)
    pd.DataFrame(target_rows).to_csv(os.path.join(args.output_dir, 'synerise_gate0_target_level_counts.csv'), index=False)

    lines = [
        '# Synerise Gate-0 Probe',
        '',
        '## Inputs',
        '',
        f"- Input dir: `{args.input_dir}`",
        f"- Product properties: `{product_properties}`",
        f"- Unique users in windows: `{len(users)}`",
        '',
        '## Gate Counts',
        '',
        '| Gate | Users | Mean pre | Mean gap | Mean val proxy | Mean val cart | Mean val search | Mean test buy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in gate_rows:
        lines.append(
            f"| {row['gate']} | {row['users']} | {row['mean_pre_all']:.2f} | "
            f"{row['mean_gap_all']:.2f} | {row['mean_val_proxy']:.2f} | "
            f"{row['mean_val_add_to_cart']:.2f} | {row['mean_val_search_query']:.2f} | "
            f"{row['mean_test_purchase']:.2f} |"
        )
    lines.extend([
        '',
        '## Target-Level Gate Counts',
        '',
        '| Gate | Level | Users | Test purchase targets | OOV vs user history | OOV rate |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for row in target_rows:
        lines.append(
            f"| {row['gate']} | {row['level']} | {row['users']} | {row['test_purchase_targets']} | "
            f"{row['test_target_oov_vs_user_history']} | {row['test_target_oov_rate_vs_user_history']:.3f} |"
        )
    lines.extend([
        '',
        '## Interpretation Stub',
        '',
        'This is metadata-only Gate-0. It does not train a recommender, run CASP, or use test-aware source selection.',
        'If item-level `sku` targets are too sparse, inspect category and price rows before rejecting Synerise.',
    ])
    with open(os.path.join(args.output_dir, 'synerise_gate0_summary.md'), 'w') as f:
        f.write('\n'.join(lines))

    print(out_json)


if __name__ == '__main__':
    main()
