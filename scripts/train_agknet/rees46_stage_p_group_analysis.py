#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import pandas as pd

import rees46_confidence_calibrated_promotion as calibrator
import rees46_exact_source_protected_list_fusion as exact
import rees46_no_training_baselines as base


def parse_time(value):
    text = str(value or '').replace(' UTC', '+00:00')
    return datetime.fromisoformat(text)


def days_between(left, right):
    try:
        return (parse_time(right) - parse_time(left)).total_seconds() / 86400.0
    except Exception:
        return 0.0


def event_times(events):
    return [str(e.get('time') or '') for e in events if e.get('time')]


def build_user_features(artifact, phase, val_context_frac, val_target_types):
    records = exact.make_records(
        artifact['users'],
        phase,
        val_context_frac,
        [x.strip() for x in val_target_types.split(',') if x.strip()],
    )
    rows = []
    for record in records:
        semantic_record = record['semantic_record']
        history_times = event_times(semantic_record.get('history', []))
        context_times = event_times(semantic_record.get('val_feedback', []))
        last_history = max(history_times) if history_times else ''
        first_context = min(context_times) if context_times else ''
        rows.append(OrderedDict([
            ('user_id', str(record['user_id'])),
            ('phase', phase),
            ('n_history', int(record['n_history'])),
            ('n_context', int(record['n_context'])),
            ('n_targets', int(len(record['targets']))),
            ('gap_days', float(days_between(last_history, first_context))),
        ]))
    return pd.DataFrame(rows)


def summarize(part, mask, k):
    open_mask = mask.loc[part.index]
    gross = int(part.loc[open_mask, f'gross@{k}'].sum())
    cann = int(part.loc[open_mask, f'cannibal@{k}'].sum())
    existing_hits = int(((part['existing_rank'] >= 0) & (part['existing_rank'] < k)).sum())
    fused_hits = int(existing_hits + gross - cann)
    return OrderedDict([
        ('n_users', int(len(part))),
        ('open_users', int(open_mask.sum())),
        ('open_rate', float(open_mask.mean()) if len(part) else 0.0),
        (f'existing_hit@{k}', existing_hits),
        (f'effective_hit@{k}', fused_hits),
        (f'gross@{k}', gross),
        (f'cannibal@{k}', cann),
        (f'net@{k}', gross - cann),
        (f'ratio@{k}', float(cann / gross) if gross else None),
    ])


def bin_feature(df, feature, bins):
    if df[feature].nunique(dropna=True) <= 1:
        return pd.Series(['all'] * len(df), index=df.index)
    try:
        return pd.qcut(df[feature], q=bins, duplicates='drop').astype(str)
    except ValueError:
        return pd.Series(['all'] * len(df), index=df.index)


def group_table(df, selected_config, features, bins, k):
    mask = calibrator.gate_mask(df, selected_config)
    rows = []
    for phase, phase_df in df.groupby('phase', sort=False):
        phase_mask = mask.loc[phase_df.index]
        overall = OrderedDict([
            ('phase', phase),
            ('feature', 'overall'),
            ('bin', 'all'),
        ])
        overall.update(summarize(phase_df, phase_mask, k))
        rows.append(overall)
        for feature in features:
            labels = bin_feature(phase_df, feature, bins)
            for label, idx in labels.groupby(labels).groups.items():
                part = phase_df.loc[idx]
                row = OrderedDict([
                    ('phase', phase),
                    ('feature', feature),
                    ('bin', str(label)),
                    ('mean_value', float(part[feature].mean())),
                ])
                row.update(summarize(part, phase_mask, k))
                rows.append(row)
    return rows


def write_markdown(path, args, summary, rows):
    lines = [
        '# REES46 Stage P Group Analysis',
        '',
        f"- Selected gate: `{summary['selected_config']['name']}`",
        f"- Evaluation K: `{args.k}`",
        '- Accounting: overlap remains credited to the existing source; group tables report only gated gross/cannibal/net deltas.',
        '',
        '## Overall',
        '',
        '| Phase | Users | Open | Open rate | Existing Hit@100 | Effective Hit@100 | Gross | Cannibal | Net | Ratio |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        if row['feature'] != 'overall':
            continue
        ratio = row.get(f'ratio@{args.k}')
        ratio_text = '' if ratio is None else f'{ratio:.3f}'
        lines.append(
            f"| {row['phase']} | {row['n_users']} | {row['open_users']} | {row['open_rate']:.3f} | "
            f"{row[f'existing_hit@{args.k}']} | {row[f'effective_hit@{args.k}']} | "
            f"{row[f'gross@{args.k}']} | {row[f'cannibal@{args.k}']} | "
            f"{row[f'net@{args.k}']} | {ratio_text} |"
        )
    lines.extend([
        '',
        '## Test Group Slices',
        '',
        '| Feature | Bin | Users | Open rate | Gross | Cannibal | Net | Ratio |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in rows:
        if row['phase'] != 'test' or row['feature'] == 'overall':
            continue
        ratio = row.get(f'ratio@{args.k}')
        ratio_text = '' if ratio is None else f'{ratio:.3f}'
        lines.append(
            f"| `{row['feature']}` | {row['bin']} | {row['n_users']} | {row['open_rate']:.3f} | "
            f"{row[f'gross@{args.k}']} | {row[f'cannibal@{args.k}']} | "
            f"{row[f'net@{args.k}']} | {ratio_text} |"
        )
    lines.extend(['', '## Interpretation', '', summary['interpretation']])
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Group-level loss/gain analysis for REES46 Stage P confidence gate.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--per-user', required=True)
    parser.add_argument('--stage-p-summary', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--bins', type=int, default=4)
    parser.add_argument('--val-context-frac', type=float, default=0.5)
    parser.add_argument('--val-target-types', default='cart')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    artifact = base.load_artifact(args.artifact)
    per_user = pd.read_csv(args.per_user)
    per_user['user_id'] = per_user['user_id'].astype(str)
    features = pd.concat([
        build_user_features(artifact, 'val', args.val_context_frac, args.val_target_types),
        build_user_features(artifact, 'test', args.val_context_frac, args.val_target_types),
    ], ignore_index=True)
    with open(args.stage_p_summary) as f:
        stage_p = json.load(f)
    selected_config = stage_p['selected_config']
    df = per_user.merge(features, on=['user_id', 'phase'], how='left', validate='one_to_one')
    missing = int(df['n_history'].isna().sum())
    if missing:
        raise RuntimeError(f'Missing group features for {missing} rows')
    group_features = [
        'gap_days',
        'n_history',
        'n_context',
        'n_targets',
        'top_semid_share',
        'top_semid_margin',
        'semid_entropy',
        'semantic_candidate_count',
    ]
    rows = group_table(df, selected_config, group_features, args.bins, args.k)
    test_rows = [r for r in rows if r['phase'] == 'test' and r['feature'] != 'overall']
    best = sorted(test_rows, key=lambda r: (r[f'net@{args.k}'], -(r.get(f'ratio@{args.k}') or 0.0)), reverse=True)[:8]
    interpretation = (
        "Stage P-A gains are not credited through overlap. The strongest positive slices should be used "
        "to decide whether Stage P-B needs interaction terms such as semantic confidence by history/context density."
    )
    summary = OrderedDict([
        ('args', vars(args)),
        ('selected_config', selected_config),
        ('overall', [r for r in rows if r['feature'] == 'overall']),
        ('top_test_slices', best),
        ('interpretation', interpretation),
    ])
    summary_json = os.path.join(args.output_dir, 'rees46_stage_p_group_analysis_summary.json')
    summary_md = os.path.join(args.output_dir, 'rees46_stage_p_group_analysis_summary.md')
    group_csv = os.path.join(args.output_dir, 'rees46_stage_p_group_analysis_slices.csv')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows).to_csv(group_csv, index=False)
    write_markdown(summary_md, args, summary, rows)
    print(json.dumps({'summary': summary_json, 'markdown': summary_md, 'slices': group_csv}, indent=2))


if __name__ == '__main__':
    main()
