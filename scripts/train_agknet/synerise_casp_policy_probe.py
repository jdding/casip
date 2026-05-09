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
        yield pd.read_parquet(path, columns=columns)
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
    return 'UNKNOWN' if not nums else '_'.join(nums[:n_tokens])


def load_product_meta(path, name_prefix_tokens):
    df = pd.read_parquet(path, columns=['sku', 'category', 'price', 'name'])
    df = df.drop_duplicates('sku')
    df['sku'] = df['sku'].astype(str)
    df['category'] = df['category'].astype(str)
    df['price'] = df['price'].astype(str)
    df['name_prefix'] = df['name'].map(lambda x: name_prefix(x, name_prefix_tokens))
    return (
        dict(zip(df['sku'], df['category'])),
        dict(zip(df['sku'], df['price'])),
        dict(zip(df['sku'], df['name_prefix'])),
    )


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


def update_count_pass(chunk, event_name, windows, window_names, frames):
    chunk['window'] = assign_windows(chunk['timestamp'], windows)
    chunk = chunk[chunk['window'].notna()].copy()
    if chunk.empty:
        return 0
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
    for event_name, path in event_paths.items():
        columns = ['client_id', 'timestamp']
        if event_name in ITEM_EVENTS:
            columns.append('sku')
        rows_seen = 0
        rows_used = 0
        for chunk in iter_parquet_batches(path, columns, batch_size):
            if max_rows_per_file and rows_seen >= max_rows_per_file:
                break
            if max_rows_per_file:
                chunk = chunk.head(max(max_rows_per_file - rows_seen, 0))
            rows_seen += len(chunk)
            rows_used += update_count_pass(chunk, event_name, windows, window_names, frames)
        print(f"[gate-count] {event_name}: rows_seen={rows_seen} rows_used={rows_used}", file=sys.stderr, flush=True)
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
    return users[masks[gate_name]]['client_id'].astype(str).tolist()


def top_items(counter, n):
    return [str(item) for item, _ in counter.most_common(n)]


def rank_counter(counter, global_scores, n):
    return [
        item for item, _ in sorted(
            counter.items(),
            key=lambda kv: (-kv[1], -global_scores.get(kv[0], 0.0), kv[0]),
        )[:n]
    ]


def hit(candidates, targets, k):
    return int(bool(set(candidates[:k]) & targets))


def make_promoted(existing, bridge, bridge_n, start_slot):
    protected = list(existing[:start_slot])
    blocked = set(protected)
    inserted = []
    for item in bridge:
        if item not in blocked:
            inserted.append(item)
            blocked.add(item)
        if len(inserted) >= bridge_n:
            break
    out = protected + inserted
    for item in existing[start_slot:]:
        if item not in blocked:
            out.append(item)
            blocked.add(item)
        if len(out) >= max(KS):
            break
    return out[:max(KS)]


def build_eval_state(input_dir, windows, batch_size, max_rows_per_file, gate, source_topk, name_prefix_tokens):
    event_paths = OrderedDict((name, find_file(input_dir, file_name)) for name, file_name in EVENT_FILES.items())
    missing_events = [name for name, path in event_paths.items() if not path]
    if missing_events:
        raise FileNotFoundError(f'Missing Synerise event files under {input_dir}: {missing_events}')
    product_properties = find_file(input_dir, 'product_properties.parquet')
    if not product_properties:
        raise FileNotFoundError(f'Missing product_properties.parquet under {input_dir}')
    category_map, price_map, name_map = load_product_meta(product_properties, name_prefix_tokens)
    window_names = [w[0] for w in windows]
    gate_users = build_gate_users(event_paths, windows, window_names, batch_size, max_rows_per_file, gate)
    gate_user_set = set(gate_users)

    global_scores = Counter()
    bucket_scores = {
        'category': defaultdict(Counter),
        'price': defaultdict(Counter),
        'category_price': defaultdict(Counter),
        'name_prefix': defaultdict(Counter),
    }
    user_existing = defaultdict(Counter)
    user_targets = defaultdict(set)
    user_keys = {source: defaultdict(set) for source in bucket_scores}

    for event_name, path in event_paths.items():
        if event_name not in ITEM_EVENTS:
            continue
        columns = ['client_id', 'timestamp', 'sku']
        rows_seen = 0
        for chunk in iter_parquet_batches(path, columns, batch_size):
            if max_rows_per_file and rows_seen >= max_rows_per_file:
                break
            if max_rows_per_file:
                chunk = chunk.head(max(max_rows_per_file - rows_seen, 0))
            rows_seen += len(chunk)
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
                for source in bucket_scores:
                    update_counter(
                        bucket_scores[source],
                        deployable.groupby([source, 'sku'], sort=False)['score'].sum().reset_index(),
                        [source, 'sku'],
                    )

            gated = chunk[chunk['client_id'].isin(gate_user_set)]
            if gated.empty:
                continue
            gated_deployable = gated[gated['window'].isin(['pre', 'val'])]
            for row in gated_deployable.itertuples(index=False):
                user_id = row.client_id
                item = str(row.sku)
                user_existing[user_id][item] += event_weight(event_name)
                for source in bucket_scores:
                    user_keys[source][user_id].add(str(getattr(row, source)))
            if event_name == PURCHASE_EVENT:
                for row in gated[gated['window'] == 'test'].itertuples(index=False):
                    user_targets[row.client_id].add(str(row.sku))
        print(f"[state-pass] {event_name}: rows_seen={rows_seen}", file=sys.stderr, flush=True)

    bucket_cache = {}

    def bucket_candidates(source, keys):
        scores = Counter()
        for key in sorted(keys):
            cache_key = (source, key)
            if cache_key not in bucket_cache:
                bucket_cache[cache_key] = top_items(bucket_scores[source].get(key, Counter()), source_topk)
            for rank, item in enumerate(bucket_cache[cache_key]):
                scores[item] += source_topk - rank
        return [item for item, _ in scores.most_common(max(KS))]

    rows = []
    for user_id in gate_users:
        targets = user_targets.get(user_id, set())
        if not targets:
            continue
        existing = rank_counter(user_existing[user_id], global_scores, max(KS))
        bridge = {source: bucket_candidates(source, user_keys[source][user_id]) for source in bucket_scores}
        rows.append({
            'user_id': user_id,
            'targets': targets,
            'existing': existing,
            'bridge': bridge,
            'features': {
                f'n_{source}': len(user_keys[source][user_id])
                for source in bucket_scores
            },
        })
    return rows


def evaluate_policies(rows, policy_specs):
    metrics = []
    for source, bridge_n, start_slot, gate_name, gate_fn in policy_specs:
        counts = Counter()
        for row in rows:
            existing = row['existing']
            targets = row['targets']
            if gate_fn(row['features']):
                promoted = make_promoted(existing, row['bridge'][source], bridge_n, start_slot)
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
        out = OrderedDict([
            ('policy', f'{source}__bridge{bridge_n}__slot{start_slot}__{gate_name}'),
            ('source', source),
            ('bridge_n', bridge_n),
            ('start_slot', start_slot),
            ('gate', gate_name),
            ('open_rate', counts['open'] / n),
        ])
        for k in KS:
            gross = counts[f'gross@{k}']
            cannibal = counts[f'cannibal@{k}']
            out[f'base@{k}'] = counts[f'base@{k}']
            out[f'hit@{k}'] = counts[f'hit@{k}']
            out[f'gross@{k}'] = gross
            out[f'cannibal@{k}'] = cannibal
            out[f'net@{k}'] = gross - cannibal
            out[f'ratio@{k}'] = cannibal / gross if gross else None
        metrics.append(out)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Validation-selected protected insertion probe for Synerise Protocol C.')
    parser.add_argument('--input-dir', default='data/synerise_dataset')
    parser.add_argument('--output-dir', default='results/20260507_synerise_casp_policy_probe')
    parser.add_argument('--batch-size', type=int, default=1_000_000)
    parser.add_argument('--max-rows-per-file', type=int, default=0, help='Optional smoke-test row cap per event file.')
    parser.add_argument(
        '--validation-windows',
        default='pre:2022-06-23:2022-10-10,gap:2022-10-10:2022-10-23,val:2022-10-23:2022-11-02,test:2022-11-02:2022-11-12',
    )
    parser.add_argument(
        '--test-windows',
        default='pre:2022-06-23:2022-10-10,gap:2022-10-10:2022-10-23,val:2022-10-23:2022-11-12,test:2022-11-12:2022-12-09',
    )
    parser.add_argument('--gate', default='pre_any_gap_silent_val_proxy_test_buy')
    parser.add_argument('--source-topk-per-bucket', type=int, default=200)
    parser.add_argument('--name-prefix-tokens', type=int, default=4)
    parser.add_argument('--min-validation-net100', type=int, default=5)
    parser.add_argument('--min-validation-net50', type=int, default=0)
    parser.add_argument('--max-validation-ratio100', type=float, default=0.5)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    gate_fns = [('all', lambda f: True)]
    for key in ('n_category', 'n_price', 'n_category_price', 'n_name_prefix'):
        for threshold in (1, 2, 3, 5, 10, 20, 50, 100):
            gate_fns.append((f'{key}_le_{threshold}', lambda f, key=key, threshold=threshold: f[key] <= threshold))
    policy_specs = [
        (source, bridge_n, start_slot, gate_name, gate_fn)
        for source in ('category', 'category_price', 'name_prefix')
        for bridge_n in (10, 25, 50, 100)
        for start_slot in (10, 20)
        for gate_name, gate_fn in gate_fns
    ]

    validation_rows = build_eval_state(
        args.input_dir,
        parse_windows(args.validation_windows),
        args.batch_size,
        args.max_rows_per_file,
        args.gate,
        args.source_topk_per_bucket,
        args.name_prefix_tokens,
    )
    val_metrics = evaluate_policies(validation_rows, policy_specs)
    feasible = [
        row for row in val_metrics
        if row['hit@10'] >= row['base@10']
        and row['net@50'] >= args.min_validation_net50
        and row['net@100'] >= args.min_validation_net100
        and (row['ratio@100'] is None or row['ratio@100'] <= args.max_validation_ratio100)
    ]
    if feasible:
        selected = sorted(
            feasible,
            key=lambda r: (r['net@100'], r['net@50'], -float(r['ratio@100'] or 0), -r['open_rate']),
            reverse=True,
        )[0]
    else:
        selected = sorted(val_metrics, key=lambda r: (r['net@100'], r['net@50'], r['hit@100']), reverse=True)[0]
    selected_spec = next(spec for spec in policy_specs if f'{spec[0]}__bridge{spec[1]}__slot{spec[2]}__{spec[3]}' == selected['policy'])

    test_rows = build_eval_state(
        args.input_dir,
        parse_windows(args.test_windows),
        args.batch_size,
        args.max_rows_per_file,
        args.gate,
        args.source_topk_per_bucket,
        args.name_prefix_tokens,
    )
    selected_test = evaluate_policies(test_rows, [selected_spec])[0]

    pd.DataFrame(val_metrics).to_csv(os.path.join(args.output_dir, 'synerise_casp_validation_grid.csv'), index=False)
    pd.DataFrame([selected]).to_csv(os.path.join(args.output_dir, 'synerise_casp_selected_validation.csv'), index=False)
    pd.DataFrame([selected_test]).to_csv(os.path.join(args.output_dir, 'synerise_casp_selected_test.csv'), index=False)

    passed_test = (
        selected_test['hit@10'] >= selected_test['base@10']
        and selected_test['net@50'] >= 0
        and selected_test['net@100'] > 0
        and (selected_test['ratio@100'] is None or selected_test['ratio@100'] <= args.max_validation_ratio100)
    )
    decision = (
        'PASS: validation-selected protected insertion keeps Top-10 and produces positive constrained test net utility.'
        if passed_test
        else 'FAIL: validation-selected protected insertion does not reproduce constrained positive test net utility.'
    )
    payload = OrderedDict([
        ('args', vars(args)),
        ('provenance', OrderedDict([
            ('cwd', os.getcwd()),
            ('argv', sys.argv),
            ('hostname', platform.node()),
            ('python', sys.version.split()[0]),
        ])),
        ('n_validation_users', len(validation_rows)),
        ('n_test_users', len(test_rows)),
        ('feasible_validation_policies', len(feasible)),
        ('selected_validation', selected),
        ('selected_test', selected_test),
        ('decision', decision),
        ('runtime_seconds', time.time() - started),
    ])
    out_json = os.path.join(args.output_dir, 'synerise_casp_policy_summary.json')
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    lines = [
        '# Synerise CASP Policy Probe',
        '',
        f"- Validation users: `{len(validation_rows)}`",
        f"- Test users: `{len(test_rows)}`",
        f"- Feasible validation policies: `{len(feasible)}`",
        f"- Selected policy: `{selected['policy']}`",
        f"- Decision: {decision}",
        '',
        '## Selected Validation',
        '',
        '| Base@10 | Hit@10 | Base@50 | Hit@50 | Net@50 | Base@100 | Hit@100 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 | Open |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        f"| {selected['base@10']} | {selected['hit@10']} | {selected['base@50']} | {selected['hit@50']} | {selected['net@50']} | "
        f"{selected['base@100']} | {selected['hit@100']} | {selected['gross@100']} | {selected['cannibal@100']} | {selected['net@100']} | {selected['ratio@100']} | {selected['open_rate']:.3f} |",
        '',
        '## Selected Test',
        '',
        '| Base@10 | Hit@10 | Base@50 | Hit@50 | Net@50 | Base@100 | Hit@100 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 | Open |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        f"| {selected_test['base@10']} | {selected_test['hit@10']} | {selected_test['base@50']} | {selected_test['hit@50']} | {selected_test['net@50']} | "
        f"{selected_test['base@100']} | {selected_test['hit@100']} | {selected_test['gross@100']} | {selected_test['cannibal@100']} | {selected_test['net@100']} | {selected_test['ratio@100']} | {selected_test['open_rate']:.3f} |",
    ]
    with open(os.path.join(args.output_dir, 'synerise_casp_policy_summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(out_json)


if __name__ == '__main__':
    main()
