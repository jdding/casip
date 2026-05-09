#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict

import pandas as pd


def parse_ints(text):
    return [int(x) for x in text.split(',') if x.strip()]


def feature_thresholds(df, feature, quantiles):
    vals = df[feature]
    if vals.nunique(dropna=True) <= 1:
        return []
    return sorted(set(float(vals.quantile(q)) for q in quantiles))


def build_gate_configs(val_df, quantiles):
    configs = [OrderedDict([
        ('name', 'existing_only'),
        ('conditions', []),
    ]), OrderedDict([
        ('name', 'always_open'),
        ('conditions', [{'feature': 'always', 'direction': 'always', 'threshold': None}]),
    ])]
    specs = [
        ('top_semid_count', 'ge'),
        ('top_semid_share', 'ge'),
        ('top_semid_margin', 'ge'),
        ('semid_entropy', 'le'),
        ('n_semids', 'le'),
        ('n_semids', 'ge'),
        ('top_bucket_size', 'le'),
        ('top_bucket_size', 'ge'),
        ('top_bucket_specificity', 'ge'),
        ('top_bucket_specificity', 'le'),
        ('semantic_candidate_count', 'le'),
        ('semantic_candidate_count', 'ge'),
    ]
    one_way = []
    for feature, direction in specs:
        for threshold in feature_thresholds(val_df, feature, quantiles):
            config = OrderedDict([
                ('name', f'{feature}_{direction}_{threshold:.6g}'),
                ('conditions', [OrderedDict([
                    ('feature', feature),
                    ('direction', direction),
                    ('threshold', threshold),
                ])]),
            ])
            configs.append(config)
            one_way.append(config)
    # A small, interpretable two-condition grid around the strongest audit signal:
    # vote concentration high AND entropy low. Keep this intentionally narrow to
    # avoid a feature-search paper.
    share_thresholds = feature_thresholds(val_df, 'top_semid_share', [0.4, 0.5, 0.6, 0.7])
    margin_thresholds = feature_thresholds(val_df, 'top_semid_margin', [0.4, 0.5, 0.6, 0.7])
    entropy_thresholds = feature_thresholds(val_df, 'semid_entropy', [0.3, 0.4, 0.5, 0.6])
    for share in share_thresholds:
        for entropy in entropy_thresholds:
            configs.append(OrderedDict([
                ('name', f'top_semid_share_ge_{share:.6g}__semid_entropy_le_{entropy:.6g}'),
                ('conditions', [
                    OrderedDict([('feature', 'top_semid_share'), ('direction', 'ge'), ('threshold', share)]),
                    OrderedDict([('feature', 'semid_entropy'), ('direction', 'le'), ('threshold', entropy)]),
                ]),
            ]))
    for margin in margin_thresholds:
        for entropy in entropy_thresholds:
            configs.append(OrderedDict([
                ('name', f'top_semid_margin_ge_{margin:.6g}__semid_entropy_le_{entropy:.6g}'),
                ('conditions', [
                    OrderedDict([('feature', 'top_semid_margin'), ('direction', 'ge'), ('threshold', margin)]),
                    OrderedDict([('feature', 'semid_entropy'), ('direction', 'le'), ('threshold', entropy)]),
                ]),
            ]))
    return configs


def gate_mask(df, config):
    if not config['conditions']:
        return pd.Series(False, index=df.index)
    mask = pd.Series(True, index=df.index)
    for condition in config['conditions']:
        feature = condition['feature']
        if feature == 'always':
            continue
        threshold = float(condition['threshold'])
        if condition['direction'] == 'ge':
            mask &= df[feature] >= threshold
        elif condition['direction'] == 'le':
            mask &= df[feature] <= threshold
        else:
            raise ValueError(condition['direction'])
    return mask


def first_hit_series(df, k, mask):
    existing_hit = (df['existing_rank'] >= 0) & (df['existing_rank'] < k)
    fused_hit = (df['fused_rank'] >= 0) & (df['fused_rank'] < k)
    return existing_hit.where(~mask, fused_hit)


def evaluate_config(df, config, ks):
    mask = gate_mask(df, config)
    out = OrderedDict([
        ('config', config['name']),
        ('conditions', config['conditions']),
        ('n_users', int(len(df))),
        ('gate_open_users', int(mask.sum())),
        ('gate_open_rate', float(mask.mean()) if len(df) else 0.0),
        ('hit_counts', OrderedDict()),
        ('gross_recovery', OrderedDict()),
        ('cannibalized_hit', OrderedDict()),
        ('net_gain', OrderedDict()),
        ('cannibalization_ratio', OrderedDict()),
    ])
    for k in ks:
        existing_hit = (df['existing_rank'] >= 0) & (df['existing_rank'] < k)
        effective_hit = first_hit_series(df, k, mask)
        gross = int((mask & (~existing_hit) & effective_hit).sum())
        cann = int((mask & existing_hit & (~effective_hit)).sum())
        out['hit_counts'][f'hit@{k}'] = int(effective_hit.sum())
        out['gross_recovery'][f'@{k}'] = gross
        out['cannibalized_hit'][f'@{k}'] = cann
        out['net_gain'][f'@{k}'] = gross - cann
        out['cannibalization_ratio'][f'@{k}'] = float(cann / gross) if gross else None
    return out


def selection_key(row, selection_k, base_hit10, min_net):
    key = f'@{selection_k}'
    top10_ok = row['hit_counts']['hit@10'] >= base_hit10
    net = row['net_gain'].get(key, 0)
    ratio = row['cannibalization_ratio'].get(key)
    ratio_val = ratio if ratio is not None else 0.0
    return (
        int(top10_ok),
        int(net >= min_net),
        -ratio_val,
        net,
        row['gross_recovery'].get(key, 0),
        -row['gate_open_rate'],
    )


def select_config(rows, selection_k, base_hit10, max_ratio, min_net, max_open_rate):
    candidates = []
    key = f'@{selection_k}'
    for row in rows:
        ratio = row['cannibalization_ratio'].get(key)
        ratio_ok = ratio is None or ratio <= max_ratio
        top10_ok = row['hit_counts']['hit@10'] >= base_hit10
        net_ok = row['net_gain'].get(key, 0) >= min_net
        open_ok = max_open_rate < 0 or row['gate_open_rate'] <= max_open_rate
        if ratio_ok and top10_ok and net_ok and open_ok:
            candidates.append(row)
    if not candidates:
        candidates = rows
    return max(candidates, key=lambda r: selection_key(r, selection_k, base_hit10, min_net))


def write_markdown(path, args, summary):
    selected = summary['selected_config']
    lines = [
        '# REES46 Confidence-Calibrated Promotion',
        '',
        f"- Input per-user audit: `{args.per_user}`",
        f"- Base promotion policy: `{summary['base_promotion_policy']}`",
        f"- Selected gate: `{selected['name']}`",
        f"- Selection: validation `net_gain@{args.selection_k}` with Top-10 no-regression and ratio <= `{args.max_validation_ratio}`",
        '- Accounting rule: overlap is not gross recovery; it may only help as a confidence feature in future list-level calibrators.',
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
        '## Top Validation Gates',
        '',
        '| Gate | Open rate | Net@100 | Gross@100 | Cannibal@100 | Ratio@100 | Hit@10 | Hit@100 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in summary['top_validation_gates']:
        ratio = row['cannibalization_ratio'].get('@100')
        ratio_text = '' if ratio is None else f'{ratio:.3f}'
        lines.append(
            f"| `{row['config']}` | {row['gate_open_rate']:.3f} | {row['net_gain'].get('@100', 0)} | "
            f"{row['gross_recovery'].get('@100', 0)} | {row['cannibalized_hit'].get('@100', 0)} | "
            f"{ratio_text} | {row['hit_counts'].get('hit@10', 0)} | {row['hit_counts'].get('hit@100', 0)} |"
        )
    lines.extend(['', '## Interpretation', '', summary['interpretation']])
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Select a confidence-gated semantic promotion policy from per-user audit outcomes.')
    parser.add_argument('--per-user', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--base-promotion-policy', default='q5__sem50__slot10__all')
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--selection-k', type=int, default=100)
    parser.add_argument('--quantiles', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('--max-validation-ratio', type=float, default=0.0)
    parser.add_argument('--min-validation-net', type=int, default=20)
    parser.add_argument('--max-open-rate', type=float, default=-1.0, help='If >=0, only select gates opening at most this validation user fraction.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ks = parse_ints(args.ks)
    quantiles = [float(x) for x in args.quantiles.split(',') if x.strip()]
    df = pd.read_csv(args.per_user)
    val_df = df[df['phase'] == 'val'].copy()
    test_df = df[df['phase'] == 'test'].copy()
    if val_df.empty or test_df.empty:
        raise ValueError('per-user audit must contain val and test phases')

    configs = build_gate_configs(val_df, quantiles)
    val_rows = [evaluate_config(val_df, config, ks) for config in configs]
    base_val = next(row for row in val_rows if row['config'] == 'existing_only')
    selected_val = select_config(
        val_rows,
        args.selection_k,
        base_val['hit_counts']['hit@10'],
        args.max_validation_ratio,
        args.min_validation_net,
        args.max_open_rate,
    )
    selected_config = next(config for config in configs if config['name'] == selected_val['config'])
    test_rows = [evaluate_config(test_df, config, ks) for config in [configs[0], selected_config]]
    base_test = next(row for row in test_rows if row['config'] == 'existing_only')
    selected_test = next(row for row in test_rows if row['config'] == selected_config['name'])
    top_val = sorted(
        val_rows,
        key=lambda r: selection_key(r, args.selection_k, base_val['hit_counts']['hit@10'], args.min_validation_net),
        reverse=True,
    )[:20]
    test_ratio = selected_test['cannibalization_ratio'].get(f'@{args.selection_k}')
    test_net = selected_test['net_gain'].get(f'@{args.selection_k}', 0)
    gate_pass = (
        test_net > 0
        and (test_ratio is None or test_ratio < 0.5)
        and selected_test['hit_counts']['hit@10'] >= base_test['hit_counts']['hit@10']
    )
    interpretation = (
        f"Selected `{selected_config['name']}`. Test net@{args.selection_k}={test_net}, "
        f"cannibal/gross={test_ratio}, gate_pass={gate_pass}. "
        "This is a Stage P-A policy over a fixed promotion candidate list; it validates whether "
        "semantic confidence proxies can decide when to open promotion, but it is not yet a "
        "candidate-level learned scorer."
    )
    summary = OrderedDict([
        ('args', vars(args)),
        ('base_promotion_policy', args.base_promotion_policy),
        ('selected_config', selected_config),
        ('base_validation', base_val),
        ('base_test', base_test),
        ('selected', OrderedDict([
            ('validation', selected_val),
            ('test', selected_test),
        ])),
        ('top_validation_gates', top_val),
        ('gate_pass', bool(gate_pass)),
        ('interpretation', interpretation),
    ])
    summary_json = os.path.join(args.output_dir, 'rees46_confidence_calibrated_promotion_summary.json')
    summary_md = os.path.join(args.output_dir, 'rees46_confidence_calibrated_promotion_summary.md')
    grid_csv = os.path.join(args.output_dir, 'rees46_confidence_calibrated_promotion_val_grid.csv')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    pd.json_normalize(val_rows).to_csv(grid_csv, index=False)
    write_markdown(summary_md, args, summary)
    print(json.dumps({'summary': summary_json, 'markdown': summary_md, 'grid': grid_csv}, indent=2))


if __name__ == '__main__':
    main()
