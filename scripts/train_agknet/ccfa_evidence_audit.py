#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
import platform
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

import tmall_casp_policy_probe as tmall


KS = (10, 50, 100, 500)


def parse_windows(text):
    return tmall.parse_windows(text)


def first_hit(rank, k):
    return int(rank >= 0 and rank < k)


def bootstrap_ci(values, n_boot=2000, seed=13):
    arr = np.asarray(values, dtype=np.int16)
    n = len(arr)
    observed = int(arr.sum())
    if n == 0:
        return OrderedDict([('observed', 0), ('ci_low', 0.0), ('ci_high', 0.0), ('p_boot_le_zero', None)])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, n), replace=True).sum(axis=1)
    return OrderedDict([
        ('observed', observed),
        ('ci_low', float(np.percentile(samples, 2.5))),
        ('ci_high', float(np.percentile(samples, 97.5))),
        ('p_boot_le_zero', float(np.mean(samples <= 0))),
    ])


def summarize_per_user(df, group_col, ks=KS):
    rows = []
    groups = [('all', df)]
    if group_col and group_col in df.columns:
        for value, gdf in df.groupby(group_col, sort=False):
            groups.append((str(value), gdf))
    for name, gdf in groups:
        row = OrderedDict([('slice', name), ('users', int(len(gdf)))])
        for k in ks:
            base = int(gdf[f'base_hit@{k}'].sum())
            casp = int(gdf[f'casp_hit@{k}'].sum())
            gross = int(((gdf[f'base_hit@{k}'] == 0) & (gdf[f'casp_hit@{k}'] == 1)).sum())
            cann = int(((gdf[f'base_hit@{k}'] == 1) & (gdf[f'casp_hit@{k}'] == 0)).sum())
            row[f'base@{k}'] = base
            row[f'casp@{k}'] = casp
            row[f'net@{k}'] = casp - base
            row[f'gross@{k}'] = gross
            row[f'cannibal@{k}'] = cann
            row[f'ratio@{k}'] = float(cann / gross) if gross else None
        rows.append(row)
    return rows


def write_table_md(rows, columns):
    lines = ['|' + '|'.join(columns) + '|', '|' + '|'.join(['---'] + ['---:'] * (len(columns) - 1)) + '|']
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col)
            if value is None:
                vals.append('')
            elif isinstance(value, float):
                vals.append(f'{value:.3f}')
            else:
                vals.append(str(value))
        lines.append('|' + '|'.join(vals) + '|')
    return lines


def make_tmall_policy_specs():
    gate_fns = [('all', lambda f: True)]
    for key in ('n_cat_brand', 'n_brand', 'n_cat'):
        for threshold in (1, 2, 3, 5, 10, 20):
            gate_fns.append((f'{key}_le_{threshold}', lambda f, key=key, threshold=threshold: f[key] <= threshold))
    return [
        (source, sem_n, gate_name, gate_fn)
        for source in ('cat_brand', 'brand', 'cat')
        for sem_n in (10, 25, 50, 100)
        for gate_name, gate_fn in gate_fns
    ]


def select_tmall_policy(val_metrics, min_net100, min_net50, max_ratio):
    feasible = [
        row for row in val_metrics
        if row['hit@10'] >= row['base@10']
        and row['net@50'] >= min_net50
        and row['net@100'] >= min_net100
        and (row['ratio@100'] is None or row['ratio@100'] <= max_ratio)
    ]
    pool = feasible if feasible else val_metrics
    selected = sorted(
        pool,
        key=lambda r: (r['net@100'], -r['ratio@100'] if r['ratio@100'] is not None else 0, -r['open_rate']),
        reverse=True,
    )[0]
    return selected, len(feasible)


def tmall_spec_by_policy(policy_specs, policy_name):
    for spec in policy_specs:
        if f'{spec[0]}__sem{spec[1]}__{spec[2]}' == policy_name:
            return spec
    raise KeyError(policy_name)


def tmall_per_user_rows(rows, selected_spec):
    source, sem_n, _, gate_fn = selected_spec
    out = []
    for row in rows:
        opened = int(gate_fn(row['features']))
        existing = row['existing']
        targets = row['targets']
        promoted = tmall.make_promoted(existing, row['semantic'][source], sem_n) if opened else existing
        history_items = row.get('history_items', set())
        pre_items = row.get('pre_items', set())
        target_oov = any(item not in history_items for item in targets)
        target_pre_oov = any(item not in pre_items for item in targets)
        record = OrderedDict([
            ('user_id', row['user_id']),
            ('opened', opened),
            ('n_targets', len(targets)),
            ('target_oov_user_history', bool(target_oov)),
            ('target_oov_pre_history', bool(target_pre_oov)),
            ('n_cat_brand', row['features']['n_cat_brand']),
            ('n_brand', row['features']['n_brand']),
            ('n_cat', row['features']['n_cat']),
        ])
        for k in KS:
            base = tmall.hit(existing, targets, k)
            casp = tmall.hit(promoted, targets, k)
            record[f'base_hit@{k}'] = base
            record[f'casp_hit@{k}'] = casp
            record[f'delta@{k}'] = casp - base
        out.append(record)
    return pd.DataFrame(out)


def run_tmall(args, output_dir):
    user_log = tmall.find_file(args.tmall_input_dir, 'user_log_format1.csv')
    if not user_log:
        raise FileNotFoundError('Missing user_log_format1.csv')
    policy_specs = make_tmall_policy_specs()
    split_rows = []
    selected_policies = []
    for idx, windows_text in enumerate(args.tmall_validation_windows):
        val_rows = tmall.build_eval_state(user_log, parse_windows(windows_text), args.chunksize, args.gate, args.source_topk_per_bucket)
        val_metrics = tmall.evaluate_policies(val_rows, policy_specs)
        selected, feasible_count = select_tmall_policy(
            val_metrics,
            args.min_validation_net100,
            args.min_validation_net50,
            args.max_validation_ratio,
        )
        split_name = f'validation_{idx + 1}'
        selected_policies.append(selected['policy'])
        split_row = OrderedDict([
            ('split', split_name),
            ('windows', windows_text),
            ('validation_users', len(val_rows)),
            ('feasible_policies', feasible_count),
            ('selected_policy', selected['policy']),
            ('open_val', selected['open_rate']),
            ('net@50_val', selected['net@50']),
            ('net@100_val', selected['net@100']),
            ('ratio@100_val', selected['ratio@100']),
        ])
        selected_spec = tmall_spec_by_policy(policy_specs, selected['policy'])
        split_rows.append((split_row, selected_spec))

    test_rows = tmall.build_eval_state(
        user_log,
        parse_windows(args.tmall_test_windows),
        args.chunksize,
        args.gate,
        args.source_topk_per_bucket,
    )
    robustness_rows = []
    per_user_frames = {}
    for split_row, selected_spec in split_rows:
        test_metrics = tmall.evaluate_policies(test_rows, [selected_spec])[0]
        for key, value in test_metrics.items():
            if key in ('policy', 'source', 'sem_n', 'gate', 'open_rate') or key.startswith(('hit@', 'base@', 'net@', 'gross@', 'cannibal@', 'ratio@')):
                split_row[f'{key}_test'] = value
        robustness_rows.append(split_row)
        if split_row['split'] == 'validation_1':
            per_user_frames['main'] = tmall_per_user_rows(test_rows, selected_spec)

    main_df = per_user_frames['main']
    main_per_user_path = os.path.join(output_dir, 'tmall_casp_per_user.csv')
    main_df.to_csv(main_per_user_path, index=False)
    slice_rows = []
    for group_col in ('target_oov_user_history', 'target_oov_pre_history', 'opened'):
        for row in summarize_per_user(main_df, group_col):
            row['group'] = group_col
            slice_rows.append(row)
    slice_path = os.path.join(output_dir, 'tmall_casp_slices.csv')
    pd.DataFrame(slice_rows).to_csv(slice_path, index=False)
    bootstrap_rows = []
    for k in KS:
        ci = bootstrap_ci(main_df[f'delta@{k}'].to_numpy(), args.bootstrap_samples, args.seed + k)
        ci['metric'] = f'net@{k}'
        bootstrap_rows.append(ci)
    bootstrap_path = os.path.join(output_dir, 'tmall_casp_bootstrap.csv')
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)
    robustness_path = os.path.join(output_dir, 'tmall_casp_alternate_validation.csv')
    pd.DataFrame(robustness_rows).to_csv(robustness_path, index=False)
    return OrderedDict([
        ('per_user', main_per_user_path),
        ('slices', slice_path),
        ('bootstrap', bootstrap_path),
        ('alternate_validation', robustness_path),
        ('selected_policies', selected_policies),
        ('robustness_rows', robustness_rows),
        ('slice_rows', slice_rows),
        ('bootstrap_rows', bootstrap_rows),
    ])


def run_rees46(args, output_dir):
    with open(args.rees46_artifact, 'rb') as f:
        artifact = pickle.load(f)
    user_meta = {}
    pre_catalog = set(str(x) for x in artifact['catalogs']['products']['pre'])
    for record in artifact['users']:
        history_items = {str(e['product_id']) for e in record['history']}
        deployable_items = set(history_items)
        deployable_items.update(str(e['product_id']) for e in record['val_feedback'])
        targets = {str(e['product_id']) for e in record['test_purchases']}
        user_meta[str(record['user_id'])] = OrderedDict([
            ('target_oov_user_history', any(t not in deployable_items for t in targets)),
            ('target_oov_pre_history', any(t not in history_items for t in targets)),
            ('target_right_new_global_pre', any(t not in pre_catalog for t in targets)),
            ('n_targets', len(targets)),
        ])
    df = pd.read_csv(args.rees46_per_user)
    test = df[df['phase'] == 'test'].copy()
    threshold = float(args.rees46_gate_threshold)
    mask = test['top_semid_share'] >= threshold
    test['opened'] = mask.astype(int)
    for k in KS:
        test[f'base_hit@{k}'] = ((test['existing_rank'] >= 0) & (test['existing_rank'] < k)).astype(int)
        effective_rank = test['existing_rank'].where(~mask, test['fused_rank'])
        test[f'casp_hit@{k}'] = ((effective_rank >= 0) & (effective_rank < k)).astype(int)
        test[f'delta@{k}'] = test[f'casp_hit@{k}'] - test[f'base_hit@{k}']
    for col in ('target_oov_user_history', 'target_oov_pre_history', 'target_right_new_global_pre', 'n_targets'):
        test[col] = test['user_id'].astype(str).map(lambda u, col=col: user_meta.get(u, {}).get(col, False if col != 'n_targets' else 0))
    per_user_path = os.path.join(output_dir, 'rees46_casp_per_user.csv')
    test.to_csv(per_user_path, index=False)
    slice_rows = []
    for group_col in ('target_oov_user_history', 'target_oov_pre_history', 'target_right_new_global_pre', 'opened'):
        for row in summarize_per_user(test, group_col):
            row['group'] = group_col
            slice_rows.append(row)
    slice_path = os.path.join(output_dir, 'rees46_casp_slices.csv')
    pd.DataFrame(slice_rows).to_csv(slice_path, index=False)
    bootstrap_rows = []
    for k in KS:
        ci = bootstrap_ci(test[f'delta@{k}'].to_numpy(), args.bootstrap_samples, args.seed + 1000 + k)
        ci['metric'] = f'net@{k}'
        bootstrap_rows.append(ci)
    bootstrap_path = os.path.join(output_dir, 'rees46_casp_bootstrap.csv')
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)
    return OrderedDict([
        ('per_user', per_user_path),
        ('slices', slice_path),
        ('bootstrap', bootstrap_path),
        ('slice_rows', slice_rows),
        ('bootstrap_rows', bootstrap_rows),
    ])


def write_markdown(path, summary):
    lines = [
        '# CCF-A Evidence Audit',
        '',
        '## Tmall Alternate Validation',
        '',
    ]
    tmall_alt = summary['tmall']['robustness_rows']
    lines.extend(write_table_md(tmall_alt, [
        'split', 'validation_users', 'feasible_policies', 'selected_policy',
        'net@50_val', 'net@100_val', 'hit@50_test', 'hit@100_test', 'net@50_test', 'net@100_test', 'ratio@100_test',
    ]))
    lines.extend(['', '## Tmall Main Slices', ''])
    lines.extend(write_table_md(summary['tmall']['slice_rows'], ['group', 'slice', 'users', 'base@100', 'casp@100', 'net@100', 'gross@100', 'cannibal@100', 'ratio@100']))
    lines.extend(['', '## Tmall Bootstrap', ''])
    lines.extend(write_table_md(summary['tmall']['bootstrap_rows'], ['metric', 'observed', 'ci_low', 'ci_high', 'p_boot_le_zero']))
    lines.extend(['', '## REES46 Slices', ''])
    lines.extend(write_table_md(summary['rees46']['slice_rows'], ['group', 'slice', 'users', 'base@100', 'casp@100', 'net@100', 'gross@100', 'cannibal@100', 'ratio@100']))
    lines.extend(['', '## REES46 Bootstrap', ''])
    lines.extend(write_table_md(summary['rees46']['bootstrap_rows'], ['metric', 'observed', 'ci_low', 'ci_high', 'p_boot_le_zero']))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Build CCF-A evidence slices, robustness, and bootstrap CIs for CASP.')
    parser.add_argument('--output-dir', default='results/20260507_ccfa_evidence_audit')
    parser.add_argument('--tmall-input-dir', default='data/tmall')
    parser.add_argument('--chunksize', type=int, default=1_000_000)
    parser.add_argument('--gate', default='pre_any_gap_silent_val_proxy_test_purchase')
    parser.add_argument('--tmall-validation-windows', nargs='+', default=[
        'pre:501:1001,gap:1001:1101,val:1101:1106,test:1106:1111',
        'pre:501:1001,gap:1001:1101,val:1101:1105,test:1105:1111',
        'pre:501:1001,gap:1001:1101,val:1101:1107,test:1107:1111',
    ])
    parser.add_argument('--tmall-test-windows', default='pre:501:1001,gap:1001:1101,val:1101:1111,test:1111:1112')
    parser.add_argument('--min-validation-net100', type=int, default=5)
    parser.add_argument('--min-validation-net50', type=int, default=10)
    parser.add_argument('--max-validation-ratio', type=float, default=0.5)
    parser.add_argument('--source-topk-per-bucket', type=int, default=200)
    parser.add_argument('--rees46-artifact', default='results/20260505_rees46_protocol_parallel_all_protocol/rees46_protocol_artifact.pkl')
    parser.add_argument('--rees46-per-user', default='results/20260506_rees46_semantic_confidence_full/rees46_semantic_confidence_audit_per_user.csv')
    parser.add_argument('--rees46-gate-threshold', type=float, default=0.5454545454545454)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    summary = OrderedDict([
        ('args', vars(args)),
        ('provenance', OrderedDict([
            ('cwd', os.getcwd()),
            ('argv', sys.argv),
            ('hostname', platform.node()),
            ('python', sys.version.split()[0]),
        ])),
    ])
    summary['tmall'] = run_tmall(args, args.output_dir)
    summary['rees46'] = run_rees46(args, args.output_dir)
    summary['runtime_seconds'] = time.time() - started
    summary_path = os.path.join(args.output_dir, 'ccfa_evidence_audit_summary.json')
    md_path = os.path.join(args.output_dir, 'ccfa_evidence_audit_summary.md')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    write_markdown(md_path, summary)
    print(json.dumps({'summary': summary_path, 'markdown': md_path}, indent=2))


if __name__ == '__main__':
    main()
