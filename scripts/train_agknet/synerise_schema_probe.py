#!/usr/bin/env python3
import argparse
import glob
import json
import os
import platform
import sys
import time
from collections import OrderedDict

import pandas as pd


EVENT_FILES = OrderedDict([
    ('product_buy', 'product_buy.parquet'),
    ('add_to_cart', 'add_to_cart.parquet'),
    ('remove_from_cart', 'remove_from_cart.parquet'),
    ('page_visit', 'page_visit.parquet'),
    ('search_query', 'search_query.parquet'),
])


def find_file(input_dir, file_name):
    if os.path.isfile(input_dir):
        return input_dir if os.path.basename(input_dir) == file_name else None
    candidates = []
    for root, _, _ in os.walk(input_dir):
        candidates.extend(glob.glob(os.path.join(root, file_name)))
    return sorted(candidates)[0] if candidates else None


def read_timestamp_range(path, batch_size, max_batches):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        df = pd.read_parquet(path, columns=['timestamp'])
        ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
        return len(df), ts.min(), ts.max(), False

    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows
    min_ts = None
    max_ts = None
    batches = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['timestamp']):
        df = batch.to_pandas()
        ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
        if not ts.empty:
            cur_min = ts.min()
            cur_max = ts.max()
            min_ts = cur_min if min_ts is None else min(min_ts, cur_min)
            max_ts = cur_max if max_ts is None else max(max_ts, cur_max)
        batches += 1
        if max_batches and batches >= max_batches:
            return total_rows, min_ts, max_ts, True
    return total_rows, min_ts, max_ts, False


def read_schema(path):
    try:
        import pyarrow.parquet as pq
        schema = pq.ParquetFile(path).schema_arrow
        return OrderedDict((field.name, str(field.type)) for field in schema)
    except Exception:
        df = pd.read_parquet(path)
        return OrderedDict((col, str(dtype)) for col, dtype in df.dtypes.items())


def suggest_windows(min_ts, max_ts):
    if min_ts is None or max_ts is None:
        return ''
    start = pd.Timestamp(min_ts).normalize()
    end = pd.Timestamp(max_ts).normalize() + pd.Timedelta(days=1)
    total_days = max((end - start).days, 1)
    pre_days = max(int(total_days * 0.65), 1)
    gap_days = max(int(total_days * 0.08), 1)
    val_days = max(int(total_days * 0.12), 1)
    pre_end = start + pd.Timedelta(days=pre_days)
    gap_end = min(pre_end + pd.Timedelta(days=gap_days), end)
    val_end = min(gap_end + pd.Timedelta(days=val_days), end)
    if val_end >= end:
        val_end = start + pd.Timedelta(days=max(total_days - 14, 1))
        gap_end = start + pd.Timedelta(days=max(total_days - 28, 1))
        pre_end = start + pd.Timedelta(days=max(total_days - 42, 1))
    return ','.join([
        f'pre:{start.date()}:{pre_end.date()}',
        f'gap:{pre_end.date()}:{gap_end.date()}',
        f'val:{gap_end.date()}:{val_end.date()}',
        f'test:{val_end.date()}:{end.date()}',
    ])


def main():
    parser = argparse.ArgumentParser(description='Inspect Synerise parquet files and suggest Gate-0 windows.')
    parser.add_argument('--input-dir', default='data/synerise')
    parser.add_argument('--output-dir', default='results/20260507_synerise_schema_probe')
    parser.add_argument('--batch-size', type=int, default=500_000)
    parser.add_argument(
        '--max-batches',
        type=int,
        default=0,
        help='Optional row-batch cap per file for fast smoke. Default scans all timestamps.',
    )
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    event_paths = OrderedDict((name, find_file(args.input_dir, file_name)) for name, file_name in EVENT_FILES.items())
    product_properties = find_file(args.input_dir, 'product_properties.parquet')
    all_paths = OrderedDict(event_paths)
    all_paths['product_properties'] = product_properties

    file_rows = []
    global_min = None
    global_max = None
    for name, path in all_paths.items():
        if not path:
            file_rows.append(OrderedDict([
                ('name', name),
                ('path', ''),
                ('exists', False),
            ]))
            continue
        schema = read_schema(path)
        row = OrderedDict([
            ('name', name),
            ('path', path),
            ('exists', True),
            ('schema', schema),
        ])
        if name in EVENT_FILES:
            n_rows, min_ts, max_ts, truncated = read_timestamp_range(path, args.batch_size, args.max_batches)
            row.update(OrderedDict([
                ('rows', int(n_rows)),
                ('min_timestamp', '' if min_ts is None else str(min_ts)),
                ('max_timestamp', '' if max_ts is None else str(max_ts)),
                ('timestamp_scan_truncated', bool(truncated)),
            ]))
            if min_ts is not None:
                global_min = min_ts if global_min is None else min(global_min, min_ts)
            if max_ts is not None:
                global_max = max_ts if global_max is None else max(global_max, max_ts)
        file_rows.append(row)

    windows = suggest_windows(global_min, global_max)
    payload = OrderedDict([
        ('args', vars(args)),
        ('provenance', OrderedDict([
            ('cwd', os.getcwd()),
            ('argv', sys.argv),
            ('hostname', platform.node()),
            ('python', sys.version.split()[0]),
        ])),
        ('files', file_rows),
        ('global_min_timestamp', '' if global_min is None else str(global_min)),
        ('global_max_timestamp', '' if global_max is None else str(global_max)),
        ('suggested_windows', windows),
        ('runtime_seconds', float(time.time() - started)),
    ])

    out_json = os.path.join(args.output_dir, 'synerise_schema_probe_summary.json')
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    lines = [
        '# Synerise Schema Probe',
        '',
        f"- Input dir: `{args.input_dir}`",
        f"- Global timestamp range: `{payload['global_min_timestamp']}` to `{payload['global_max_timestamp']}`",
        f"- Suggested windows: `{windows}`",
        '',
        '## Files',
        '',
        '| Name | Exists | Rows | Min timestamp | Max timestamp | Truncated | Path |',
        '|---|---:|---:|---|---|---:|---|',
    ]
    for row in file_rows:
        lines.append(
            f"| {row['name']} | {row.get('exists', False)} | {row.get('rows', '')} | "
            f"{row.get('min_timestamp', '')} | {row.get('max_timestamp', '')} | "
            f"{row.get('timestamp_scan_truncated', '')} | `{row.get('path', '')}` |"
        )
    lines.extend([
        '',
        '## Next Command',
        '',
        '```bash',
        'python3 scripts/train_agknet/synerise_gate0_probe.py \\',
        f'  --input-dir {args.input_dir} \\',
        '  --output-dir results/20260507_synerise_gate0_probe \\',
        f'  --windows {windows}',
        '```',
    ])
    with open(os.path.join(args.output_dir, 'synerise_schema_probe_summary.md'), 'w') as f:
        f.write('\n'.join(lines))

    print(out_json)


if __name__ == '__main__':
    main()
