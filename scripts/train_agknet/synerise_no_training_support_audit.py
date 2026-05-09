#!/usr/bin/env python3
import argparse
import glob
import json
import os
import platform
import re
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
KS = (10, 50, 100, 500)


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
        out.loc[(ts >= start) & (ts < end)] = name
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


def event_weight(event_name):
    if event_name == 'product_buy':
        return 4.0
    if event_name == 'add_to_cart':
        return 2.0
    if event_name == 'remove_from_cart':
        return 0.5
    return 0.0


def name_prefix(value, n_tokens):
    nums = re.findall(r'\d+', str(value))
    if not nums:
        return 'UNKNOWN'
    return '_'.join(nums[:n_tokens])


def load_product_meta(path, name_prefix_tokens):
    df = pd.read_parquet(path, columns=['sku', 'category', 'price', 'name'])
    df = df.drop_duplicates('sku')
    df['sku'] = df['sku'].astype(str)
    df['category'] = df['category'].astype(str)
    df['price'] = df['price'].astype(str)
    df['name_prefix'] = df['name'].map(lambda x: name_prefix(x, name_prefix_tokens))
    category = dict(zip(df['sku'], df['category']))
    price = dict(zip(df['sku'], df['price']))
    name_code = dict(zip(df['sku'], df['name_prefix']))
    return category, price, name_code


def update_count_pass(chunk, event_name, windows, window_names, frames, event_counts):
    chunk['window'] = assign_windows(chunk['timestamp'], windows)
    chunk = chunk[chunk['window'].notna()].copy()
    if chunk.empty:
        return 0
    event_counts[event_name] += int(len(chunk))
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
    frames.append(out.reset_index())
    return int(len(chunk))


def build_gate_users(event_paths, windows, window_names, batch_size, max_rows_per_file, gate_name):
    frames = []
    event_counts = Counter()
    rows_seen = Counter()
    rows_used = Counter()
    for event_name, path in event_paths.items():
        columns = ['client_id', 'timestamp']
        if event_name in ITEM_EVENTS:
            columns.append('sku')
        seen_for_file = 0
        for chunk in iter_parquet_batches(path, columns, batch_size):
            if max_rows_per_file and seen_for_file >= max_rows_per_file:
                break
            if max_rows_per_file:
                chunk = chunk.head(max(max_rows_per_file - seen_for_file, 0))
            seen_for_file += len(chunk)
            rows_seen[event_name] += int(len(chunk))
            rows_used[event_name] += update_count_pass(
                chunk,
                event_name,
                windows,
                window_names,
                frames,
                event_counts,
            )
        print(
            f"[gate-count] {event_name}: rows_seen={rows_seen[event_name]} rows_used={rows_used[event_name]}",
            file=sys.stderr,
            flush=True,
        )
    users = pd.concat(frames, ignore_index=True).groupby('client_id', as_index=False).sum(numeric_only=True)
    for window in window_names:
        for key in ['all', 'proxy', 'purchase'] + list(EVENT_FILES.keys()):
            col = f'{window}_{key}'
            if col not in users.columns:
                users[col] = 0

    masks = {
        'pre_any_gap_silent_val_proxy_test_buy':
            (users['pre_all'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1),
        'pre_buy_gap_silent_val_proxy_test_buy':
            (users['pre_purchase'] >= 1) & (users['gap_all'] == 0) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1),
        'pre_any_gap_low5_val_proxy_test_buy':
            (users['pre_all'] >= 1) & (users['gap_all'] <= 5) & (users['val_proxy'] >= 1) & (users['test_purchase'] >= 1),
        'pre_any_gap_silent_val_cart_search_test_buy':
            (users['pre_all'] >= 1) & (users['gap_all'] == 0)
            & ((users['val_add_to_cart'] + users['val_search_query']) >= 1)
            & (users['test_purchase'] >= 1),
    }
    if gate_name not in masks:
        raise ValueError(f'Unknown gate {gate_name}; choose one of {sorted(masks)}')
    gate = users[masks[gate_name]].copy()
    return users, gate['client_id'].astype(str).tolist(), rows_seen, rows_used, event_counts


def enrich_item_chunk(chunk, category_map, price_map, name_map):
    chunk['sku'] = chunk['sku'].astype(str)
    chunk['category'] = chunk['sku'].map(category_map).fillna('UNKNOWN')
    chunk['price'] = chunk['sku'].map(price_map).fillna('UNKNOWN')
    chunk['name_prefix'] = chunk['sku'].map(name_map).fillna('UNKNOWN')
    chunk['category_price'] = chunk['category'] + '|price=' + chunk['price']
    return chunk


def update_counter(counter, grouped, key_cols):
    for row in grouped.itertuples(index=False):
        key = tuple(str(getattr(row, col)) for col in key_cols)
        if len(key) == 1:
            counter[key[0]] += float(row.score)
        else:
            bucket = key[0] if len(key) == 2 else '|'.join(key[:-1])
            item = key[-1]
            counter[bucket][item] += float(row.score)


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
    target_set = set(targets)
    for k in KS:
        out[k] = int(bool(set(candidates[:k]) & target_set))
    return out


def main():
    parser = argparse.ArgumentParser(description='No-training support audit for Synerise Protocol C.')
    parser.add_argument('--input-dir', default='data/synerise_dataset')
    parser.add_argument('--output-dir', default='results/20260507_synerise_support_audit')
    parser.add_argument('--batch-size', type=int, default=1_000_000)
    parser.add_argument('--max-rows-per-file', type=int, default=0)
    parser.add_argument('--windows', required=True)
    parser.add_argument('--gate', default='pre_any_gap_silent_val_proxy_test_buy')
    parser.add_argument('--source-topk-per-bucket', type=int, default=200)
    parser.add_argument('--name-prefix-tokens', type=int, default=4)
    parser.add_argument('--max-users', type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    windows = parse_windows(args.windows)
    window_names = [w[0] for w in windows]
    event_paths = OrderedDict((name, find_file(args.input_dir, file_name)) for name, file_name in EVENT_FILES.items())
    missing_events = [name for name, path in event_paths.items() if not path]
    if missing_events:
        raise FileNotFoundError(f'Missing Synerise event files under {args.input_dir}: {missing_events}')
    product_properties = find_file(args.input_dir, 'product_properties.parquet')
    if not product_properties:
        raise FileNotFoundError(f'Missing product_properties.parquet under {args.input_dir}')

    category_map, price_map, name_map = load_product_meta(product_properties, args.name_prefix_tokens)
    users, gate_users, rows_seen, rows_used, event_counts = build_gate_users(
        event_paths,
        windows,
        window_names,
        args.batch_size,
        args.max_rows_per_file,
        args.gate,
    )
    if args.max_users:
        gate_users = gate_users[:args.max_users]
    gate_user_set = set(gate_users)

    global_scores = Counter()
    category_scores = defaultdict(Counter)
    price_scores = defaultdict(Counter)
    category_price_scores = defaultdict(Counter)
    name_scores = defaultdict(Counter)
    user_existing = defaultdict(Counter)
    user_targets = defaultdict(set)
    user_categories = defaultdict(set)
    user_prices = defaultdict(set)
    user_category_prices = defaultdict(set)
    user_names = defaultdict(set)

    for event_name, path in event_paths.items():
        if event_name not in ITEM_EVENTS:
            continue
        columns = ['client_id', 'timestamp', 'sku']
        seen_for_file = 0
        for chunk in iter_parquet_batches(path, columns, args.batch_size):
            if args.max_rows_per_file and seen_for_file >= args.max_rows_per_file:
                break
            if args.max_rows_per_file:
                chunk = chunk.head(max(args.max_rows_per_file - seen_for_file, 0))
            seen_for_file += len(chunk)
            chunk['window'] = assign_windows(chunk['timestamp'], windows)
            chunk = chunk[chunk['window'].notna()].copy()
            if chunk.empty:
                continue
            chunk['client_id'] = chunk['client_id'].astype(str)
            chunk = enrich_item_chunk(chunk, category_map, price_map, name_map)

            deployable = chunk[chunk['window'].isin(['pre', 'val'])].copy()
            if not deployable.empty:
                deployable['score'] = event_weight(event_name)
                update_counter(global_scores, deployable.groupby('sku', sort=False)['score'].sum().reset_index(), ['sku'])
                update_counter(category_scores, deployable.groupby(['category', 'sku'], sort=False)['score'].sum().reset_index(), ['category', 'sku'])
                update_counter(price_scores, deployable.groupby(['price', 'sku'], sort=False)['score'].sum().reset_index(), ['price', 'sku'])
                update_counter(category_price_scores, deployable.groupby(['category_price', 'sku'], sort=False)['score'].sum().reset_index(), ['category_price', 'sku'])
                update_counter(name_scores, deployable.groupby(['name_prefix', 'sku'], sort=False)['score'].sum().reset_index(), ['name_prefix', 'sku'])

            gated = chunk[chunk['client_id'].isin(gate_user_set)]
            if gated.empty:
                continue
            gated_deployable = gated[gated['window'].isin(['pre', 'val'])]
            for row in gated_deployable.itertuples(index=False):
                user_id = row.client_id
                item = str(row.sku)
                score = event_weight(event_name)
                user_existing[user_id][item] += score
                user_categories[user_id].add(str(row.category))
                user_prices[user_id].add(str(row.price))
                user_category_prices[user_id].add(str(row.category_price))
                user_names[user_id].add(str(row.name_prefix))
            if event_name == PURCHASE_EVENT:
                test_rows = gated[gated['window'] == 'test']
                for row in test_rows.itertuples(index=False):
                    user_targets[row.client_id].add(str(row.sku))
        print(f"[support-pass] {event_name}: rows_seen={seen_for_file}", file=sys.stderr, flush=True)

    global_top = top_items(global_scores, max(KS))
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

    source_names = [
        'existing_user_replay',
        'global_deployable_popularity',
        'bridge_category',
        'bridge_price',
        'bridge_category_price',
        'bridge_name_prefix',
        'bridge_union_oracle',
        'existing_plus_bridge_oracle',
        'existing_plus_bridge_all',
    ]
    source_hits = {name: Counter() for name in source_names}
    existing_miss_recovered = Counter()
    users_eval = 0

    for user_id in gate_users:
        targets = user_targets.get(user_id, set())
        if not targets:
            continue
        users_eval += 1
        existing = rank_user_counter(user_existing[user_id], global_scores, max(KS))
        source_lists = {
            'existing_user_replay': existing,
            'global_deployable_popularity': global_top,
            'bridge_category': bucket_candidates('category', user_categories[user_id], category_scores),
            'bridge_price': bucket_candidates('price', user_prices[user_id], price_scores),
            'bridge_category_price': bucket_candidates('category_price', user_category_prices[user_id], category_price_scores),
            'bridge_name_prefix': bucket_candidates('name_prefix', user_names[user_id], name_scores),
        }
        combined_scores = Counter()
        for source in ('existing_user_replay', 'bridge_category', 'bridge_price', 'bridge_category_price', 'bridge_name_prefix'):
            for rank, item in enumerate(source_lists[source]):
                combined_scores[item] += max(KS) - rank
        source_lists['existing_plus_bridge_all'] = [item for item, _ in combined_scores.most_common(max(KS))]
        existing_hit = hit_vector(existing, targets)
        bridge_sources = ['bridge_category', 'bridge_price', 'bridge_category_price', 'bridge_name_prefix']
        for source, candidates in source_lists.items():
            hits = hit_vector(candidates, targets)
            for k, v in hits.items():
                source_hits[source][k] += v
                if source.startswith('bridge') and source != 'bridge_union_oracle' and v and not existing_hit[k]:
                    existing_miss_recovered[(source, k)] += 1
        for k in KS:
            bridge_oracle_hit = any(set(source_lists[source][:k]) & targets for source in bridge_sources)
            source_hits['bridge_union_oracle'][k] += int(bridge_oracle_hit)
            source_hits['existing_plus_bridge_oracle'][k] += int(existing_hit[k] or bridge_oracle_hit)

    rows = []
    for source, counts in source_hits.items():
        row = OrderedDict([('source', source)])
        for k in KS:
            row[f'hit@{k}'] = int(counts[k])
        rows.append(row)

    existing_row = next(row for row in rows if row['source'] == 'existing_user_replay')
    oracle_row = next(row for row in rows if row['source'] == 'existing_plus_bridge_oracle')
    decision = (
        'PASS: bridge sources expand no-training support beyond existing replay.'
        if oracle_row['hit@100'] > existing_row['hit@100']
        else 'FAIL: bridge sources do not expand Hit@100 support beyond existing replay.'
    )

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
        ('unique_users_in_windows', int(len(users))),
        ('gate_users', int(len(gate_users))),
        ('users_eval', int(users_eval)),
        ('rows_seen', OrderedDict((k, int(v)) for k, v in rows_seen.items())),
        ('rows_used_in_windows', OrderedDict((k, int(v)) for k, v in rows_used.items())),
        ('event_counts_in_windows', OrderedDict((k, int(v)) for k, v in event_counts.items())),
        ('source_hits', rows),
        ('existing_miss_recovered', OrderedDict((f'{s}@{k}', int(v)) for (s, k), v in sorted(existing_miss_recovered.items()))),
        ('decision', decision),
        ('runtime_seconds', float(time.time() - started)),
    ])

    out_json = os.path.join(args.output_dir, 'synerise_support_audit_summary.json')
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'synerise_support_audit_summary.csv'), index=False)

    lines = [
        '# Synerise No-Training Support Audit',
        '',
        f"- Gate: `{args.gate}`",
        f"- Gate users: `{len(gate_users)}`",
        f"- Evaluated users with test purchases: `{users_eval}`",
        f"- Decision: {decision}",
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
    lines.extend([
        '',
        '## Interpretation Stub',
        '',
        'This is a no-training support audit. `bridge_union_oracle` and `existing_plus_bridge_oracle` are audit-only headroom, not deployable methods.',
        'Proceed to naive insertion / CASP only if support expansion is positive and the bridge source is not just price-bucket popularity.',
    ])
    with open(os.path.join(args.output_dir, 'synerise_support_audit_summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(out_json)


if __name__ == '__main__':
    main()
