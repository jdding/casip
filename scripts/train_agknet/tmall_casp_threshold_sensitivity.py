#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import OrderedDict

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tmall_casp_policy_probe import build_eval_state, evaluate_policies, parse_windows, find_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/tmall')
    parser.add_argument('--output-dir', default='results/20260506_tmall_casp_threshold_sensitivity')
    parser.add_argument('--chunksize', type=int, default=1_000_000)
    parser.add_argument('--validation-grid', default='results/20260506_tmall_casp_policy_probe_net50_10/tmall_casp_validation_grid.csv')
    parser.add_argument('--test-windows', default='pre:501:1001,gap:1001:1101,val:1101:1111,test:1111:1112')
    parser.add_argument('--gate', default='pre_any_gap_silent_val_proxy_test_purchase')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    user_log = find_file(args.input_dir, 'user_log_format1.csv')
    if not user_log:
        raise FileNotFoundError('Missing user_log_format1.csv')

    val_grid = pd.read_csv(args.validation_grid)
    family = val_grid[
        (val_grid['source'] == 'cat_brand') &
        (val_grid['sem_n'] == 50) &
        (val_grid['gate'].isin(['all', 'n_cat_brand_le_1', 'n_cat_brand_le_2', 'n_cat_brand_le_3', 'n_cat_brand_le_5', 'n_cat_brand_le_10', 'n_cat_brand_le_20']))
    ].copy()

    gate_fns = [('all', lambda f: True)]
    for threshold in (1, 2, 3, 5, 10, 20):
        gate_fns.append((f'n_cat_brand_le_{threshold}', lambda f, threshold=threshold: f['n_cat_brand'] <= threshold))
    policy_specs = [('cat_brand', 50, gate_name, gate_fn) for gate_name, gate_fn in gate_fns]

    test_rows = build_eval_state(user_log, parse_windows(args.test_windows), args.chunksize, args.gate, 200)
    test_metrics = pd.DataFrame(evaluate_policies(test_rows, policy_specs))
    out = family.merge(
        test_metrics,
        on=['policy', 'source', 'sem_n', 'gate'],
        suffixes=('_validation', '_test'),
    )
    out = out.sort_values('open_rate_test')
    out_csv = os.path.join(args.output_dir, 'tmall_casp_catbrand_sem50_threshold_sensitivity.csv')
    out.to_csv(out_csv, index=False)

    payload = OrderedDict([
        ('input_validation_grid', args.validation_grid),
        ('test_windows', args.test_windows),
        ('n_test_users', len(test_rows)),
        ('rows', out.to_dict(orient='records')),
    ])
    with open(os.path.join(args.output_dir, 'tmall_casp_catbrand_sem50_threshold_sensitivity.json'), 'w') as f:
        json.dump(payload, f, indent=2)

    lines = [
        '# Tmall CASP Threshold Sensitivity',
        '',
        '- Family: `cat_brand`, `sem_n=50`',
        f"- Test users: `{len(test_rows)}`",
        '',
        '| Gate | Val open | Val net@50 | Val net@100 | Test open | Test net@10 | Test net@50 | Test net@100 | Test ratio@100 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    def fmt_ratio(value):
        return '--' if pd.isna(value) else f'{float(value):.3f}'

    for row in out.to_dict(orient='records'):
        lines.append(
            f"| {row['gate']} | {row['open_rate_validation']:.3f} | {int(row['net@50_validation'])} | "
            f"{int(row['net@100_validation'])} | {row['open_rate_test']:.3f} | {int(row['net@10_test'])} | "
            f"{int(row['net@50_test'])} | {int(row['net@100_test'])} | {fmt_ratio(row['ratio@100_test'])} |"
        )
    with open(os.path.join(args.output_dir, 'tmall_casp_catbrand_sem50_threshold_sensitivity.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(out_csv)


if __name__ == '__main__':
    main()
