#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import rees46_confidence_calibrated_promotion as gate_eval
import rees46_no_training_baselines as base
import rees46_stage_p_group_analysis as group_analysis


def safe_log1p(x):
    return np.log1p(np.maximum(np.asarray(x, dtype=float), 0.0))


def build_feature_frame(per_user, artifact, args):
    features = pd.concat([
        group_analysis.build_user_features(artifact, 'val', args.val_context_frac, args.val_target_types),
        group_analysis.build_user_features(artifact, 'test', args.val_context_frac, args.val_target_types),
    ], ignore_index=True)
    per_user = per_user.copy()
    per_user['user_id'] = per_user['user_id'].astype(str)
    df = per_user.merge(features, on=['user_id', 'phase'], how='left', validate='one_to_one')
    if int(df['n_history'].isna().sum()):
        raise RuntimeError('Missing artifact-derived user features')
    return add_features(df)


def add_features(df):
    out = df.copy()
    out['inv_n_history'] = 1.0 / (1.0 + out['n_history'].astype(float))
    out['inv_n_context'] = 1.0 / (1.0 + out['n_context'].astype(float))
    out['log_n_history'] = safe_log1p(out['n_history'])
    out['log_n_context'] = safe_log1p(out['n_context'])
    out['log_gap_days'] = safe_log1p(out['gap_days'])
    out['log_semantic_candidate_count'] = safe_log1p(out['semantic_candidate_count'])
    out['inv_entropy'] = 1.0 - out['semid_entropy'].astype(float)
    out['confidence_entropy'] = out['top_semid_share'] * out['inv_entropy']
    out['confidence_inv_history'] = out['top_semid_share'] * out['inv_n_history']
    out['confidence_gap'] = out['top_semid_share'] * out['log_gap_days']
    out['confidence_context'] = out['top_semid_share'] * out['log_n_context']
    out['margin_inv_history'] = out['top_semid_margin'] * out['inv_n_history']
    out['margin_context'] = out['top_semid_margin'] * out['log_n_context']
    return out


FEATURES = [
    'top_semid_share',
    'top_semid_margin',
    'semid_entropy',
    'top_semid_count',
    'n_semids',
    'top_bucket_size',
    'top_bucket_specificity',
    'semantic_candidate_count',
    'n_history',
    'n_context',
    'n_targets',
    'gap_days',
    'inv_n_history',
    'inv_n_context',
    'log_n_history',
    'log_n_context',
    'log_gap_days',
    'log_semantic_candidate_count',
    'inv_entropy',
    'confidence_entropy',
    'confidence_inv_history',
    'confidence_gap',
    'confidence_context',
    'margin_inv_history',
    'margin_context',
]


def training_subset(df, k, neutral_weight):
    gross = df[f'gross@{k}'].astype(int)
    cann = df[f'cannibal@{k}'].astype(int)
    label = pd.Series(np.nan, index=df.index, dtype=float)
    label[gross > 0] = 1.0
    label[cann > 0] = 0.0
    if neutral_weight > 0:
        label[label.isna()] = 0.0
        weights = pd.Series(neutral_weight, index=df.index, dtype=float)
        weights[gross > 0] = 1.0
        weights[cann > 0] = 1.0
        return df, label.astype(int), weights
    mask = label.notna()
    return df.loc[mask], label.loc[mask].astype(int), pd.Series(1.0, index=df.loc[mask].index)


def make_model(kind, penalty_weight, max_depth):
    if kind == 'logistic':
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight={0: float(penalty_weight), 1: 1.0},
                max_iter=2000,
                solver='liblinear',
            ),
        )
    if kind == 'tree':
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=20,
            class_weight={0: float(penalty_weight), 1: 1.0},
            random_state=42,
        )
    if kind == 'hgb':
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=8,
            l2_regularization=1.0,
            random_state=42,
        )
    raise ValueError(kind)


def score_model(model, df):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(df[FEATURES])[:, 1]
    return model.decision_function(df[FEATURES])


def eval_threshold(df, score, threshold, ks):
    tmp = df.copy()
    tmp['_score'] = score
    config = OrderedDict([
        ('name', f'score_ge_{threshold:.6g}'),
        ('conditions', [OrderedDict([('feature', '_score'), ('direction', 'ge'), ('threshold', threshold)])]),
    ])
    return gate_eval.evaluate_config(tmp, config, ks)


def select_threshold(df, score, ks, selection_k, max_ratio, min_net, max_open_rate):
    thresholds = sorted(set(float(x) for x in np.quantile(score, np.linspace(0.05, 0.95, 19))))
    thresholds.extend([float(np.max(score) + 1e-9), float(np.min(score) - 1e-9)])
    rows = [eval_threshold(df, score, t, ks) for t in thresholds]
    base_hit10 = int(((df['existing_rank'] >= 0) & (df['existing_rank'] < 10)).sum())
    candidates = []
    key = f'@{selection_k}'
    for row in rows:
        ratio = row['cannibalization_ratio'].get(key)
        ratio_ok = ratio is None or ratio <= max_ratio
        net_ok = row['net_gain'].get(key, 0) >= min_net
        top10_ok = row['hit_counts']['hit@10'] >= base_hit10
        open_ok = max_open_rate < 0 or row['gate_open_rate'] <= max_open_rate
        if ratio_ok and net_ok and top10_ok and open_ok:
            candidates.append(row)
    if not candidates:
        candidates = rows
    selected = max(
        candidates,
        key=lambda r: (
            r['net_gain'].get(key, 0),
            -(r['cannibalization_ratio'].get(key) if r['cannibalization_ratio'].get(key) is not None else 0.0),
            r['gross_recovery'].get(key, 0),
            -r['gate_open_rate'],
        ),
    )
    return selected, rows


def coefficient_table(model):
    if not hasattr(model, 'named_steps') or 'logisticregression' not in model.named_steps:
        return []
    lr = model.named_steps['logisticregression']
    rows = []
    for name, coef in zip(FEATURES, lr.coef_[0]):
        rows.append(OrderedDict([('feature', name), ('coef', float(coef))]))
    return sorted(rows, key=lambda r: abs(r['coef']), reverse=True)


def tree_text(model):
    if isinstance(model, DecisionTreeClassifier):
        return export_text(model, feature_names=FEATURES, max_depth=4)
    return ''


def write_markdown(path, args, summary):
    lines = [
        '# REES46 Stage P Transparent Calibrator',
        '',
        f"- Model: `{summary['model_kind']}`",
        f"- Selected threshold: `{summary['selected_threshold']}`",
        f"- Training labels: gross@{args.selection_k}=positive, cannibal@{args.selection_k}=negative; neutral weight `{args.neutral_weight}`",
        f"- Negative/cannibal penalty weight: `{args.penalty_weight}`",
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
    if summary['coefficients']:
        lines.extend([
            '',
            '## Logistic Coefficients',
            '',
            '| Feature | Coef |',
            '|---|---:|',
        ])
        for row in summary['coefficients'][:15]:
            lines.append(f"| `{row['feature']}` | {row['coef']:.4f} |")
    if summary.get('tree_text'):
        lines.extend(['', '## Tree', '', '```text', summary['tree_text'], '```'])
    lines.extend(['', '## Interpretation', '', summary['interpretation']])
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Transparent Stage P-B calibrator over semantic confidence and user-state features.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--per-user', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', choices=['logistic', 'tree', 'hgb'], default='logistic')
    parser.add_argument('--selection-k', type=int, default=100)
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--penalty-weight', type=float, default=3.0)
    parser.add_argument('--neutral-weight', type=float, default=0.05)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--max-validation-ratio', type=float, default=0.5)
    parser.add_argument('--min-validation-net', type=int, default=20)
    parser.add_argument('--max-open-rate', type=float, default=0.7)
    parser.add_argument('--val-context-frac', type=float, default=0.5)
    parser.add_argument('--val-target-types', default='cart')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    artifact = base.load_artifact(args.artifact)
    df = build_feature_frame(pd.read_csv(args.per_user), artifact, args)
    val_df = df[df['phase'] == 'val'].copy()
    test_df = df[df['phase'] == 'test'].copy()
    train_x, train_y, sample_weight = training_subset(val_df, args.selection_k, args.neutral_weight)
    model = make_model(args.model, args.penalty_weight, args.max_depth)
    model.fit(train_x[FEATURES], train_y, **({'sample_weight': sample_weight} if args.model in ['tree', 'hgb'] else {'logisticregression__sample_weight': sample_weight}))
    val_score = score_model(model, val_df)
    selected_val, val_threshold_rows = select_threshold(
        val_df,
        val_score,
        ks,
        args.selection_k,
        args.max_validation_ratio,
        args.min_validation_net,
        args.max_open_rate,
    )
    threshold = float(selected_val['config'].replace('score_ge_', ''))
    test_score = score_model(model, test_df)
    selected_test = eval_threshold(test_df, test_score, threshold, ks)
    test_ratio = selected_test['cannibalization_ratio'].get(f'@{args.selection_k}')
    test_net = selected_test['net_gain'].get(f'@{args.selection_k}', 0)
    base_test_hit10 = int(((test_df['existing_rank'] >= 0) & (test_df['existing_rank'] < 10)).sum())
    gate_pass = (
        test_net > 0
        and (test_ratio is None or test_ratio < 0.5)
        and selected_test['hit_counts']['hit@10'] >= base_test_hit10
    )
    interpretation = (
        f"Selected threshold {threshold}. Test net@{args.selection_k}={test_net}, "
        f"cannibal/gross={test_ratio}, gate_pass={gate_pass}. "
        "This is the transparent P-B calibrator; compare it against P-A before escalating to a richer model."
    )
    summary = OrderedDict([
        ('args', vars(args)),
        ('model_kind', args.model),
        ('train_counts', OrderedDict([
            ('n_train_rows', int(len(train_x))),
            ('positive_gross', int((train_y == 1).sum())),
            ('negative_cannibal_or_neutral', int((train_y == 0).sum())),
        ])),
        ('selected_threshold', threshold),
        ('selected', OrderedDict([
            ('validation', selected_val),
            ('test', selected_test),
        ])),
        ('top_validation_thresholds', sorted(
            val_threshold_rows,
            key=lambda r: (
                r['net_gain'].get(f'@{args.selection_k}', 0),
                -(r['cannibalization_ratio'].get(f'@{args.selection_k}') if r['cannibalization_ratio'].get(f'@{args.selection_k}') is not None else 0.0),
            ),
            reverse=True,
        )[:15]),
        ('coefficients', coefficient_table(model)),
        ('tree_text', tree_text(model)),
        ('gate_pass', bool(gate_pass)),
        ('interpretation', interpretation),
    ])
    summary_json = os.path.join(args.output_dir, f'rees46_stage_p_{args.model}_calibrator_summary.json')
    summary_md = os.path.join(args.output_dir, f'rees46_stage_p_{args.model}_calibrator_summary.md')
    thresholds_csv = os.path.join(args.output_dir, f'rees46_stage_p_{args.model}_threshold_grid.csv')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    pd.json_normalize(val_threshold_rows).to_csv(thresholds_csv, index=False)
    write_markdown(summary_md, args, summary)
    print(json.dumps({'summary': summary_json, 'markdown': summary_md, 'thresholds': thresholds_csv}, indent=2))


if __name__ == '__main__':
    main()
