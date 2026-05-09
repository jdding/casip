#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict

import pandas as pd

import rees46_confidence_calibrated_promotion as gate_eval
import rees46_no_training_baselines as base
import rees46_stage_p_group_analysis as group_analysis
import rees46_stage_p_transparent_calibrator as transparent


def quantile_thresholds(df, feature, qs):
    vals = df[feature]
    if vals.nunique(dropna=True) <= 1:
        return []
    return sorted(set(float(vals.quantile(q)) for q in qs))


def config_name(conditions):
    if not conditions:
        return 'existing_only'
    return '__'.join(f"{c['feature']}_{c['direction']}_{c['threshold']:.6g}" for c in conditions)


def condition(feature, direction, threshold):
    return OrderedDict([
        ('feature', feature),
        ('direction', direction),
        ('threshold', float(threshold)),
    ])


def build_configs(val_df):
    configs = [OrderedDict([('name', 'existing_only'), ('conditions', [])])]
    qs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    semantic_conditions = []
    for feature, direction in [
        ('top_semid_share', 'ge'),
        ('top_semid_margin', 'ge'),
        ('semid_entropy', 'le'),
        ('confidence_entropy', 'ge'),
    ]:
        for threshold in quantile_thresholds(val_df, feature, qs):
            semantic_conditions.append(condition(feature, direction, threshold))
    user_conditions = []
    for feature, direction in [
        ('n_history', 'ge'),
        ('n_history', 'le'),
        ('n_context', 'ge'),
        ('n_context', 'le'),
        ('gap_days', 'ge'),
        ('gap_days', 'le'),
        ('semantic_candidate_count', 'le'),
        ('semantic_candidate_count', 'ge'),
    ]:
        for threshold in quantile_thresholds(val_df, feature, [0.25, 0.5, 0.75]):
            user_conditions.append(condition(feature, direction, threshold))
    seen = set()
    for sem in semantic_conditions:
        for conds in [[sem]] + [[sem, user] for user in user_conditions]:
            name = config_name(conds)
            if name in seen:
                continue
            seen.add(name)
            configs.append(OrderedDict([('name', name), ('conditions', conds)]))
    return configs


def select(rows, selection_k, max_ratio, min_net, max_open_rate):
    key = f'@{selection_k}'
    base = next(r for r in rows if r['config'] == 'existing_only')
    candidates = []
    for row in rows:
        ratio = row['cannibalization_ratio'].get(key)
        ratio_ok = ratio is None or ratio <= max_ratio
        net_ok = row['net_gain'].get(key, 0) >= min_net
        top10_ok = row['hit_counts']['hit@10'] >= base['hit_counts']['hit@10']
        open_ok = max_open_rate < 0 or row['gate_open_rate'] <= max_open_rate
        if ratio_ok and net_ok and top10_ok and open_ok:
            candidates.append(row)
    if not candidates:
        candidates = rows
    return max(candidates, key=lambda r: (
        r['net_gain'].get(key, 0),
        -(r['cannibalization_ratio'].get(key) if r['cannibalization_ratio'].get(key) is not None else 0.0),
        r['gross_recovery'].get(key, 0),
        -r['gate_open_rate'],
    ))


def write_markdown(path, args, summary):
    lines = [
        '# REES46 Stage P Interaction Rule Grid',
        '',
        f"- Selected rule: `{summary['selected_config']['name']}`",
        f"- Selection constraints: validation ratio <= `{args.max_validation_ratio}`, net@{args.selection_k} >= `{args.min_validation_net}`, open rate <= `{args.max_open_rate}`",
        '- Rule class: semantic-confidence condition plus optional one user-state/candidate condition.',
        '',
        '## Selected Result',
        '',
        '| Split | Open rate | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for split in ['validation', 'test']:
        row = summary['selected'][split]
        ratio = row['cannibalization_ratio'].get('@100')
        ratio_text = '' if ratio is None else f'{ratio:.3f}'
        lines.append(
            f"| {split} | {row['gate_open_rate']:.3f} | {row['hit_counts'].get('hit@10', 0)} | "
            f"{row['hit_counts'].get('hit@50', 0)} | {row['hit_counts'].get('hit@100', 0)} | "
            f"{row['hit_counts'].get('hit@500', 0)} | {row['gross_recovery'].get('@100', 0)} | "
            f"{row['cannibalized_hit'].get('@100', 0)} | {row['net_gain'].get('@100', 0)} | {ratio_text} |"
        )
    lines.extend([
        '',
        '## Top Validation Rules',
        '',
        '| Rule | Open rate | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for row in summary['top_validation_rules']:
        ratio = row['cannibalization_ratio'].get('@100')
        ratio_text = '' if ratio is None else f'{ratio:.3f}'
        lines.append(
            f"| `{row['config']}` | {row['gate_open_rate']:.3f} | "
            f"{row['gross_recovery'].get('@100', 0)} | {row['cannibalized_hit'].get('@100', 0)} | "
            f"{row['net_gain'].get('@100', 0)} | {ratio_text} |"
        )
    lines.extend(['', '## Interpretation', '', summary['interpretation']])
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Transparent Stage P interaction rule grid.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--per-user', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--selection-k', type=int, default=100)
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--max-validation-ratio', type=float, default=0.0)
    parser.add_argument('--min-validation-net', type=int, default=20)
    parser.add_argument('--max-open-rate', type=float, default=0.7)
    parser.add_argument('--val-context-frac', type=float, default=0.5)
    parser.add_argument('--val-target-types', default='cart')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    artifact = base.load_artifact(args.artifact)
    df = transparent.build_feature_frame(pd.read_csv(args.per_user), artifact, args)
    val_df = df[df['phase'] == 'val'].copy()
    test_df = df[df['phase'] == 'test'].copy()
    configs = build_configs(val_df)
    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    val_rows = [gate_eval.evaluate_config(val_df, config, ks) for config in configs]
    selected_val = select(
        val_rows,
        args.selection_k,
        args.max_validation_ratio,
        args.min_validation_net,
        args.max_open_rate,
    )
    selected_config = next(c for c in configs if c['name'] == selected_val['config'])
    test_row = gate_eval.evaluate_config(test_df, selected_config, ks)
    key = f'@{args.selection_k}'
    test_ratio = test_row['cannibalization_ratio'].get(key)
    gate_pass = test_row['net_gain'].get(key, 0) > 0 and (test_ratio is None or test_ratio < 0.5)
    top_val = sorted(
        val_rows,
        key=lambda r: (
            r['net_gain'].get(key, 0),
            -(r['cannibalization_ratio'].get(key) if r['cannibalization_ratio'].get(key) is not None else 0.0),
            r['gross_recovery'].get(key, 0),
        ),
        reverse=True,
    )[:20]
    interpretation = (
        f"Selected `{selected_config['name']}`. Test net@{args.selection_k}="
        f"{test_row['net_gain'].get(key, 0)}, ratio={test_ratio}, gate_pass={gate_pass}. "
        "This is the transparent interaction-rule P-B variant; compare against P-A and learned LR/tree."
    )
    summary = OrderedDict([
        ('args', vars(args)),
        ('selected_config', selected_config),
        ('selected', OrderedDict([
            ('validation', selected_val),
            ('test', test_row),
        ])),
        ('top_validation_rules', top_val),
        ('gate_pass', bool(gate_pass)),
        ('interpretation', interpretation),
    ])
    summary_json = os.path.join(args.output_dir, 'rees46_stage_p_interaction_rule_grid_summary.json')
    summary_md = os.path.join(args.output_dir, 'rees46_stage_p_interaction_rule_grid_summary.md')
    grid_csv = os.path.join(args.output_dir, 'rees46_stage_p_interaction_rule_grid_val.csv')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    pd.json_normalize(val_rows).to_csv(grid_csv, index=False)
    write_markdown(summary_md, args, summary)
    print(json.dumps({'summary': summary_json, 'markdown': summary_md, 'grid': grid_csv}, indent=2))


if __name__ == '__main__':
    main()
