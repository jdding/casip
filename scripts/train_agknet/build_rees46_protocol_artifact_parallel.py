#!/usr/bin/env python3
import argparse
import glob
import json
import os
import pickle
import platform
import sys
import time
import zipfile
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


USECOLS = [
    'event_time',
    'event_type',
    'product_id',
    'category_id',
    'category_code',
    'brand',
    'user_id',
]
IMPLICIT_EVENTS = {'view', 'cart'}
DEFAULT_WINDOWS = (
    'pre:2019-10-01:2019-12-01,'
    'gap:2019-12-01:2020-02-01,'
    'val:2020-02-01:2020-03-15,'
    'test:2020-03-15:2020-05-01'
)


def parse_windows(text):
    windows = []
    for spec in text.split(','):
        if not spec.strip():
            continue
        name, start, end = spec.strip().split(':')
        windows.append((name, start, end))
    names = [x[0] for x in windows]
    if set(names) != {'pre', 'gap', 'val', 'test'}:
        raise ValueError(f'windows must contain pre/gap/val/test, got {names}')
    return windows


def discover_sources(path):
    sources = []
    if os.path.isdir(path):
        for pattern in ('*.zip', '*.csv.gz', '*.csv'):
            for child in sorted(glob.glob(os.path.join(path, pattern))):
                if os.path.basename(child).startswith('sha256_'):
                    continue
                sources.extend(discover_sources(child))
    elif path.endswith('.zip'):
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith('.csv'):
                    sources.append({'kind': 'zip_member', 'path': path, 'member': name, 'label': name})
    else:
        sources.append({'kind': 'file', 'path': path, 'member': '', 'label': os.path.basename(path)})
    return sources


def assign_windows_vectorized(event_times, windows):
    day = event_times.astype(str).str.slice(0, 10)
    out = pd.Series(pd.NA, index=event_times.index, dtype='object')
    for name, start, end in windows:
        mask = (day >= start) & (day < end)
        out.loc[mask] = name
    return out


def normalize_str(value):
    if pd.isna(value):
        return ''
    return str(value)


def iter_source_chunks(source, chunksize, max_rows=0):
    rows_yielded = 0
    if source['kind'] == 'zip_member':
        with zipfile.ZipFile(source['path']) as zf:
            with zf.open(source['member']) as f:
                for chunk in pd.read_csv(f, usecols=USECOLS, chunksize=chunksize):
                    if max_rows and rows_yielded + len(chunk) > max_rows:
                        chunk = chunk.iloc[: max_rows - rows_yielded]
                    rows_yielded += len(chunk)
                    yield chunk
                    if max_rows and rows_yielded >= max_rows:
                        return
    else:
        for chunk in pd.read_csv(source['path'], usecols=USECOLS, chunksize=chunksize):
            if max_rows and rows_yielded + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - rows_yielded]
            rows_yielded += len(chunk)
            yield chunk
            if max_rows and rows_yielded >= max_rows:
                return


def add_product_meta(product_meta, frame):
    cols = ['product_id', 'category_id', 'category_code', 'brand']
    for row in frame[cols].dropna(subset=['product_id']).drop_duplicates('product_id').itertuples(index=False):
        product_id = normalize_str(row.product_id)
        if product_id not in product_meta:
            product_meta[product_id] = (
                normalize_str(row.category_id),
                normalize_str(row.category_code),
                normalize_str(row.brand),
            )


def scan_source(args_tuple):
    source, windows, category_prefixes, chunksize, max_rows, shard_dir = args_tuple
    started = time.time()
    user_counts = defaultdict(Counter)
    product_sets = defaultdict(set)
    category_sets = defaultdict(set)
    brand_sets = defaultdict(set)
    product_meta = OrderedDict()
    popularity = defaultdict(Counter)
    purchase_popularity = defaultdict(Counter)
    event_counts = Counter()
    window_event_counts = Counter()
    rows_seen = 0
    rows_kept = 0

    for chunk in iter_source_chunks(source, chunksize, max_rows=max_rows):
        rows_seen += len(chunk)
        chunk['window'] = assign_windows_vectorized(chunk['event_time'], windows)
        chunk = chunk[chunk['window'].notna()]
        if category_prefixes:
            category_code = chunk['category_code'].fillna('').astype(str)
            keep = pd.Series(False, index=chunk.index)
            for prefix in category_prefixes:
                keep = keep | category_code.str.startswith(prefix)
            chunk = chunk[keep]
        if chunk.empty:
            continue
        rows_kept += len(chunk)
        event_counts.update(chunk['event_type'].astype(str).tolist())
        add_product_meta(product_meta, chunk)
        for (window, event_type), n in chunk.groupby(['window', 'event_type']).size().items():
            window_event_counts[f'{window}:{event_type}'] += int(n)

        for window, wdf in chunk.groupby('window', sort=False):
            products = wdf['product_id'].dropna().astype(str)
            product_sets[window].update(products.unique().tolist())
            category_sets[window].update(wdf['category_id'].dropna().astype(str).unique().tolist())
            brand_sets[window].update(wdf['brand'].dropna().astype(str).unique().tolist())
            popularity[f'{window}_all'].update(products.tolist())
            popularity[f'{window}_implicit'].update(
                wdf[wdf['event_type'].isin(IMPLICIT_EVENTS)]['product_id'].dropna().astype(str).tolist()
            )
            purchase_popularity[f'{window}_purchase'].update(
                wdf[wdf['event_type'] == 'purchase']['product_id'].dropna().astype(str).tolist()
            )
            tmp = wdf[['user_id', 'event_type']].copy()
            tmp['event_group'] = 'other'
            tmp.loc[tmp['event_type'].isin(IMPLICIT_EVENTS), 'event_group'] = 'implicit'
            tmp.loc[tmp['event_type'] == 'purchase', 'event_group'] = 'purchase'
            grouped = tmp.groupby(['user_id', 'event_group']).size()
            for (user_id, group), n in grouped.items():
                user_key = normalize_str(user_id)
                user_counts[user_key][f'{window}_{group}'] += int(n)
                user_counts[user_key][f'{window}_all'] += int(n)

    shard = OrderedDict([
        ('source', source),
        ('rows_seen', rows_seen),
        ('rows_kept', rows_kept),
        ('user_counts', dict(user_counts)),
        ('product_sets', dict(product_sets)),
        ('category_sets', dict(category_sets)),
        ('brand_sets', dict(brand_sets)),
        ('product_meta', product_meta),
        ('popularity', dict(popularity)),
        ('purchase_popularity', dict(purchase_popularity)),
        ('event_counts', event_counts),
        ('window_event_counts', window_event_counts),
        ('runtime_seconds', time.time() - started),
    ])
    safe = source['label'].replace('/', '_').replace('.', '_')
    path = os.path.join(shard_dir, f'scan_{safe}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {'label': source['label'], 'path': path, 'rows_seen': rows_seen, 'rows_kept': rows_kept, 'runtime_seconds': shard['runtime_seconds']}


def merge_scan_shards(paths):
    merged = {
        'user_counts': defaultdict(Counter),
        'product_sets': defaultdict(set),
        'category_sets': defaultdict(set),
        'brand_sets': defaultdict(set),
        'product_meta': OrderedDict(),
        'popularity': defaultdict(Counter),
        'purchase_popularity': defaultdict(Counter),
        'event_counts': Counter(),
        'window_event_counts': Counter(),
        'rows_seen': 0,
        'rows_kept': 0,
        'sources': [],
    }
    for path in paths:
        with open(path, 'rb') as f:
            shard = pickle.load(f)
        merged['sources'].append(shard['source'])
        merged['rows_seen'] += shard['rows_seen']
        merged['rows_kept'] += shard['rows_kept']
        merged['event_counts'].update(shard['event_counts'])
        merged['window_event_counts'].update(shard['window_event_counts'])
        for user_id, counts in shard['user_counts'].items():
            merged['user_counts'][user_id].update(counts)
        for key, values in shard['product_sets'].items():
            merged['product_sets'][key].update(values)
        for key, values in shard['category_sets'].items():
            merged['category_sets'][key].update(values)
        for key, values in shard['brand_sets'].items():
            merged['brand_sets'][key].update(values)
        for product_id, meta in shard['product_meta'].items():
            if product_id not in merged['product_meta']:
                merged['product_meta'][product_id] = meta
        for key, counter in shard['popularity'].items():
            merged['popularity'][key].update(counter)
        for key, counter in shard['purchase_popularity'].items():
            merged['purchase_popularity'][key].update(counter)
    return merged


def collect_source_records(args_tuple):
    source, windows, category_prefixes, chunksize, max_rows, strict_users, out_dir = args_tuple
    strict_set = set(strict_users)
    records = defaultdict(lambda: {'history': [], 'val_feedback': [], 'test_purchases': []})
    for chunk in iter_source_chunks(source, chunksize, max_rows=max_rows):
        chunk['window'] = assign_windows_vectorized(chunk['event_time'], windows)
        chunk = chunk[chunk['window'].notna()]
        if category_prefixes:
            category_code = chunk['category_code'].fillna('').astype(str)
            keep = pd.Series(False, index=chunk.index)
            for prefix in category_prefixes:
                keep = keep | category_code.str.startswith(prefix)
            chunk = chunk[keep]
        if chunk.empty:
            continue
        chunk['user_key'] = chunk['user_id'].astype(str)
        chunk = chunk[chunk['user_key'].isin(strict_set)]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            product_id = normalize_str(row.product_id)
            event = OrderedDict([
                ('time', normalize_str(row.event_time)),
                ('type', normalize_str(row.event_type)),
                ('product_id', product_id),
                ('category_id', normalize_str(row.category_id)),
                ('category_code', normalize_str(row.category_code)),
                ('brand', normalize_str(row.brand)),
            ])
            rec = records[row.user_key]
            if row.window == 'pre':
                rec['history'].append(event)
            elif row.window == 'val' and row.event_type in IMPLICIT_EVENTS:
                rec['val_feedback'].append(event)
            elif row.window == 'test' and row.event_type == 'purchase':
                rec['test_purchases'].append(event)
    safe = source['label'].replace('/', '_').replace('.', '_')
    path = os.path.join(out_dir, f'records_{safe}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(dict(records), f, protocol=pickle.HIGHEST_PROTOCOL)
    return {'label': source['label'], 'path': path, 'users': len(records)}


def top_counter(counter, n):
    return [[str(k), int(v)] for k, v in counter.most_common(n)]


def attach_row_budgets(sources, max_rows):
    if max_rows <= 0:
        return [(source, 0) for source in sources]
    out = []
    remaining = max_rows
    for source in sources:
        if remaining <= 0:
            break
        out.append((source, remaining))
        remaining = 0
    return out


def turnover(product_sets):
    out = OrderedDict()
    for left, right in [('pre', 'val'), ('pre', 'test'), ('val', 'test')]:
        lset, rset = product_sets[left], product_sets[right]
        overlap = len(lset & rset)
        union = len(lset | rset)
        out[f'{left}_to_{right}'] = OrderedDict([
            ('left_products', len(lset)),
            ('right_products', len(rset)),
            ('overlap_products', overlap),
            ('jaccard', float(overlap / max(union, 1))),
            ('right_new_vs_left', len(rset - lset)),
            ('right_new_rate', float(len(rset - lset) / max(len(rset), 1))),
        ])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--chunksize', type=int, default=1_000_000)
    parser.add_argument('--windows', default=DEFAULT_WINDOWS)
    parser.add_argument('--category-prefixes', default='')
    parser.add_argument('--min-pre-events', type=int, default=1)
    parser.add_argument('--min-val-implicit', type=int, default=1)
    parser.add_argument('--min-test-purchases', type=int, default=1)
    parser.add_argument('--top-pop-items', type=int, default=5000)
    parser.add_argument('--max-rows', type=int, default=0, help='Debug only; 0 means full scan.')
    parser.add_argument('--max-strict-users', type=int, default=0, help='Debug only; 0 keeps all strict users.')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    shard_dir = os.path.join(args.output_dir, '_parallel_shards')
    os.makedirs(shard_dir, exist_ok=True)
    windows = parse_windows(args.windows)
    category_prefixes = tuple(x.strip() for x in args.category_prefixes.split(',') if x.strip())
    sources = discover_sources(args.input)
    if not sources:
        raise RuntimeError(f'No input CSV sources found under {args.input}')
    source_budgets = attach_row_budgets(sources, args.max_rows)
    print(json.dumps({
        'stage': 'discover',
        'sources': [s['label'] for s, _ in source_budgets],
        'workers': args.workers,
        'max_rows': args.max_rows,
    }), flush=True)

    scan_jobs = [(s, windows, category_prefixes, args.chunksize, max_rows, shard_dir) for s, max_rows in source_budgets]
    scan_paths = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(scan_jobs))) as ex:
        futures = [ex.submit(scan_source, job) for job in scan_jobs]
        for fut in as_completed(futures):
            res = fut.result()
            scan_paths.append(res['path'])
            print(json.dumps({'stage': 'scan_done', **res}), flush=True)
    merged = merge_scan_shards(scan_paths)

    strict_users = []
    gate_rows = []
    for user_id, counts in merged['user_counts'].items():
        pre_all = counts.get('pre_all', 0)
        gap_all = counts.get('gap_all', 0)
        val_implicit = counts.get('val_implicit', 0)
        test_purchase = counts.get('test_purchase', 0)
        if pre_all >= args.min_pre_events and gap_all == 0 and val_implicit >= args.min_val_implicit and test_purchase >= args.min_test_purchases:
            strict_users.append(user_id)
            gate_rows.append(OrderedDict([
                ('user_id', user_id),
                ('pre_all', pre_all),
                ('pre_purchase', counts.get('pre_purchase', 0)),
                ('gap_all', gap_all),
                ('val_all', counts.get('val_all', 0)),
                ('val_implicit', val_implicit),
                ('val_purchase', counts.get('val_purchase', 0)),
                ('test_all', counts.get('test_all', 0)),
                ('test_purchase', test_purchase),
            ]))
    strict_users.sort()
    if args.max_strict_users and len(strict_users) > args.max_strict_users:
        strict_user_set = set(strict_users[: args.max_strict_users])
        strict_users = strict_users[: args.max_strict_users]
        gate_rows = [row for row in gate_rows if row['user_id'] in strict_user_set]
    print(json.dumps({'stage': 'strict_users', 'n': len(strict_users)}), flush=True)
    if not strict_users:
        raise RuntimeError('No strict users found')

    record_jobs = [
        (s, windows, category_prefixes, args.chunksize, max_rows, strict_users, shard_dir)
        for s, max_rows in source_budgets
    ]
    record_paths = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(record_jobs))) as ex:
        futures = [ex.submit(collect_source_records, job) for job in record_jobs]
        for fut in as_completed(futures):
            res = fut.result()
            record_paths.append(res['path'])
            print(json.dumps({'stage': 'records_done', **res}), flush=True)

    records_map = OrderedDict((u, OrderedDict([
        ('user_id', u),
        ('history', []),
        ('val_feedback', []),
        ('test_purchases', []),
    ])) for u in strict_users)
    for path in record_paths:
        with open(path, 'rb') as f:
            shard_records = pickle.load(f)
        for user_id, rec in shard_records.items():
            out = records_map[user_id]
            out['history'].extend(rec['history'])
            out['val_feedback'].extend(rec['val_feedback'])
            out['test_purchases'].extend(rec['test_purchases'])
    records = list(records_map.values())
    for rec in records:
        if not rec['history'] or not rec['val_feedback'] or not rec['test_purchases']:
            raise RuntimeError(f'Incomplete strict-user record: {rec["user_id"]}')

    product_meta = OrderedDict()
    for product_id, meta in merged['product_meta'].items():
        product_meta[product_id] = OrderedDict([
            ('category_id', meta[0]),
            ('category_code', meta[1]),
            ('brand', meta[2]),
        ])
    artifact = OrderedDict([
        ('artifact_version', 'rees46_protocol_v1'),
        ('args', vars(args)),
        ('provenance', OrderedDict([
            ('cwd', os.getcwd()),
            ('argv', sys.argv),
            ('hostname', platform.node()),
            ('python', sys.version.split()[0]),
            ('created_at_unix', time.time()),
        ])),
        ('windows', [{'name': n, 'start': s, 'end': e} for n, s, e in windows]),
        ('strict_user_counts', gate_rows),
        ('users', records),
        ('catalogs', OrderedDict([
            ('products', OrderedDict((w, sorted(merged['product_sets'][w])) for w, _, _ in windows)),
            ('categories', OrderedDict((w, sorted(merged['category_sets'][w])) for w, _, _ in windows)),
            ('brands', OrderedDict((w, sorted(merged['brand_sets'][w])) for w, _, _ in windows)),
        ])),
        ('product_meta', product_meta),
        ('popularity_top', OrderedDict((name, top_counter(counter, args.top_pop_items)) for name, counter in sorted(merged['popularity'].items()))),
        ('purchase_popularity_top', OrderedDict((name, top_counter(counter, args.top_pop_items)) for name, counter in sorted(merged['purchase_popularity'].items()))),
        ('scan_summary', OrderedDict([
            ('rows_seen', merged['rows_seen']),
            ('rows_kept', merged['rows_kept']),
            ('event_counts', OrderedDict(sorted(merged['event_counts'].items()))),
            ('window_event_counts', OrderedDict(sorted(merged['window_event_counts'].items()))),
            ('unique_users_in_windows', len(merged['user_counts'])),
            ('strict_users', len(records)),
        ])),
        ('catalog_turnover', turnover(merged['product_sets'])),
    ])
    artifact_path = os.path.join(args.output_dir, 'rees46_protocol_artifact.pkl')
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary = OrderedDict([
        ('artifact_path', artifact_path),
        ('args', vars(args)),
        ('scan_summary', artifact['scan_summary']),
        ('catalog_turnover', artifact['catalog_turnover']),
        ('strict_users', len(records)),
        ('mean_history_events', float(sum(len(r['history']) for r in records) / max(len(records), 1))),
        ('mean_val_feedback_events', float(sum(len(r['val_feedback']) for r in records) / max(len(records), 1))),
        ('mean_test_purchase_events', float(sum(len(r['test_purchases']) for r in records) / max(len(records), 1))),
        ('runtime_seconds', float(time.time() - started)),
    ])
    with open(os.path.join(args.output_dir, 'rees46_protocol_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, 'rees46_protocol_summary.md'), 'w') as f:
        f.write('# REES46 Compact Protocol Artifact\n\n')
        f.write(f"- Strict users: `{len(records)}`\n")
        f.write(f"- Mean history events: `{summary['mean_history_events']:.2f}`\n")
        f.write(f"- Mean validation feedback events: `{summary['mean_val_feedback_events']:.2f}`\n")
        f.write(f"- Mean test purchase events: `{summary['mean_test_purchase_events']:.2f}`\n")
        f.write(f"- Artifact: `{artifact_path}`\n\n")
        f.write('## Catalog Turnover\n\n')
        f.write('| Pair | Left products | Right products | Overlap | Jaccard | Right-new rate |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        for pair, row in artifact['catalog_turnover'].items():
            f.write(f"| {pair} | {row['left_products']} | {row['right_products']} | {row['overlap_products']} | {row['jaccard']:.4f} | {row['right_new_rate']:.3f} |\n")
    print(json.dumps({'stage': 'done', 'artifact': artifact_path, 'summary': os.path.join(args.output_dir, 'rees46_protocol_summary.json')}), flush=True)


if __name__ == '__main__':
    main()
