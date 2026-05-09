#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, OrderedDict

import pandas as pd

import rees46_confidence_calibrated_promotion as gate_eval
import rees46_exact_source_protected_list_fusion as exact
import rees46_no_training_baselines as base
import rees46_semantic_bridge_audit as semantic
import rees46_semantic_confidence_audit as conf
import rees46_source_aware_promotion_gate as promo
import rees46_validation_safe_gate as gate


def parse_ints(text):
    return [int(x) for x in text.split(',') if x.strip()]


def rank_stats(values, missing_value):
    vals = [missing_value if v is None else int(v) for v in values]
    if not vals:
        return 0.0, 0.0, float(missing_value)
    vals = sorted(vals)
    return float(sum(vals) / len(vals)), float(vals[len(vals) // 2]), float(min(vals))


def mean_or_zero(values):
    return float(sum(values) / len(values)) if values else 0.0


def semantic_ids_for_items(items, meta, scheme):
    return [semantic.semantic_id(item, meta, scheme) for item in items if semantic.semantic_id(item, meta, scheme)]


def selected_for_promotion(existing_list, semantic_list, config, promotion_k, max_eval_k):
    existing_rank = {item: idx for idx, item in enumerate(existing_list[:max_eval_k])}
    selected = []
    selected_semantic_ranks = []
    for sr, item in enumerate(semantic_list[:int(config['semantic_budget'])]):
        item = str(item)
        if not promo.candidate_allowed(item, config['filter'], existing_rank, promotion_k):
            continue
        if item in selected:
            continue
        selected.append(item)
        selected_semantic_ranks.append(sr)
        if len(selected) >= int(config['quota']):
            break
    return selected, selected_semantic_ranks


def action_features(record, state, scheme, scope, bucket_ranks, existing_weights, args, promotion_config, phase):
    max_k = max(parse_ints(args.ks))
    max_list = max(max_k, args.source_topk, args.existing_pool_k)
    source_lists = gate.make_source_lists(
        record['profile'],
        state['indexed'],
        state['global_counter'],
        state['fallback_counter'],
        event_weights={'view': args.event_view_weight, 'cart': args.event_cart_weight},
    )
    existing_list = gate.blend_ranked_lists(
        source_lists,
        existing_weights,
        state['indexed']['rank_order'],
        args.source_topk,
        max_list,
    )
    semantic_list = semantic.candidate_list_for_user(
        record['semantic_record'],
        scheme,
        scope,
        state['sem_meta'],
        bucket_ranks,
        args.semantic_source_topk,
    )
    fused = promo.promote(existing_list, semantic_list, promotion_config, args.selection_k, max_k)
    existing_rank_map = {item: idx for idx, item in enumerate(existing_list[:max_list])}
    selected, selected_semantic_ranks = selected_for_promotion(
        existing_list,
        semantic_list,
        promotion_config,
        args.selection_k,
        max_k,
    )
    existing_top = set(existing_list[:args.selection_k])
    fused_top = set(fused[:args.selection_k])
    inserted = [x for x in fused[:args.selection_k] if x not in existing_top]
    displaced = [x for x in existing_list[:args.selection_k] if x not in fused_top]
    selected_existing_ranks = [existing_rank_map.get(x) for x in selected]
    selected_rank_gaps = [
        (args.existing_pool_k if r is None else r) - int(promotion_config['insert_start'])
        for r in selected_existing_ranks
    ]
    displaced_ranks = [existing_rank_map.get(x, args.selection_k) for x in displaced]
    selected_sids = set(semantic_ids_for_items(selected, state['sem_meta'], scheme))
    existing_top10_sids = set(semantic_ids_for_items(existing_list[:10], state['sem_meta'], scheme))
    existing_top100_sids = set(semantic_ids_for_items(existing_list[:100], state['sem_meta'], scheme))
    semantic_top50_sids = semantic_ids_for_items(semantic_list[:int(promotion_config['semantic_budget'])], state['sem_meta'], scheme)
    semid_counts = Counter(semantic_top50_sids)
    features = conf.semantic_confidence_features(record['semantic_record'], scheme, scope, state['sem_meta'], bucket_ranks)
    existing_rank = promo.first_hit_rank(existing_list[:max_k], record['targets'])
    fused_rank = promo.first_hit_rank(fused, record['targets'])
    out = OrderedDict([
        ('user_id', record['user_id']),
        ('phase', phase),
        ('existing_rank', int(existing_rank)),
        ('fused_rank', int(fused_rank)),
        ('selected_count', int(len(selected))),
        ('inserted_count@100', int(len(inserted))),
        ('displaced_count@100', int(len(displaced))),
        ('selected_semantic_only_count', int(sum(1 for r in selected_existing_ranks if r is None))),
        ('selected_tail_overlap_count', int(sum(1 for r in selected_existing_ranks if r is not None and r >= args.selection_k))),
        ('selected_overlap_top500_count', int(sum(1 for r in selected_existing_ranks if r is not None and r < max_k))),
        ('selected_semantic_rank_mean', mean_or_zero(selected_semantic_ranks)),
        ('selected_semantic_rank_min', float(min(selected_semantic_ranks)) if selected_semantic_ranks else 0.0),
    ])
    mean_rank, median_rank, min_rank = rank_stats(selected_existing_ranks, args.existing_pool_k)
    out['selected_existing_rank_mean_fill'] = mean_rank
    out['selected_existing_rank_median_fill'] = median_rank
    out['selected_existing_rank_min_fill'] = min_rank
    out['selected_rank_gap_mean'] = mean_or_zero(selected_rank_gaps)
    out['selected_rank_gap_min'] = float(min(selected_rank_gaps)) if selected_rank_gaps else 0.0
    out['displaced_rank_mean'] = mean_or_zero(displaced_ranks)
    out['displaced_rank_min'] = float(min(displaced_ranks)) if displaced_ranks else 0.0
    out['displaced_rank_max'] = float(max(displaced_ranks)) if displaced_ranks else 0.0
    out['collision_top10_semid'] = int(bool(selected_sids & existing_top10_sids))
    out['collision_top100_semid'] = int(bool(selected_sids & existing_top100_sids))
    out['selected_semid_count'] = int(len(selected_sids))
    out['semantic_top50_semid_entropy'] = conf.entropy_from_counts(list(semid_counts.values()))
    out['semantic_top50_n_semids'] = int(len(semid_counts))
    out.update(features)
    for k in parse_ints(args.ks):
        base_hit = 0 <= existing_rank < k
        fused_hit = 0 <= fused_rank < k
        out[f'gross@{k}'] = int((not base_hit) and fused_hit)
        out[f'cannibal@{k}'] = int(base_hit and (not fused_hit))
        out[f'net@{k}'] = int(fused_hit) - int(base_hit)
    return out


def build_phase_rows(artifact, args, phase, scheme, scope, promotion_config, existing_weights):
    state = exact.build_phase_state(artifact, phase, 'val' if phase == 'val' else 'test')
    bucket_ranks, _ = semantic.build_bucket_ranks(
        state['active_products'],
        state['sem_meta'],
        [scheme],
        state['sem_pop_purchase'],
        state['sem_pop_implicit'],
        state['sem_pop_pre'],
        state['sem_pop_all'],
        args.bucket_topk,
    )
    records = exact.make_records(
        artifact['users'][:args.max_users] if args.max_users else artifact['users'],
        phase,
        args.val_context_frac,
        [x.strip() for x in args.val_target_types.split(',') if x.strip()],
    )
    rows = []
    for idx, record in enumerate(records):
        if idx and idx % 1000 == 0:
            print(f'[{phase}] residual rows {idx}/{len(records)}', flush=True)
        rows.append(action_features(record, state, scheme, scope, bucket_ranks, existing_weights, args, promotion_config, phase))
    return rows


def feature_thresholds(df, feature, qs):
    if df[feature].nunique(dropna=True) <= 1:
        return []
    return sorted(set(float(df[feature].quantile(q)) for q in qs))


def build_configs(val_df):
    configs = [OrderedDict([('name', 'existing_only'), ('conditions', [])])]
    specs = [
        ('top_semid_share', 'ge'),
        ('top_semid_margin', 'ge'),
        ('semid_entropy', 'le'),
        ('selected_existing_rank_mean_fill', 'ge'),
        ('selected_existing_rank_min_fill', 'ge'),
        ('selected_rank_gap_mean', 'ge'),
        ('selected_semantic_rank_mean', 'le'),
        ('selected_semantic_only_count', 'ge'),
        ('selected_tail_overlap_count', 'ge'),
        ('collision_top10_semid', 'le'),
        ('collision_top100_semid', 'le'),
        ('semantic_top50_n_semids', 'le'),
        ('semantic_top50_semid_entropy', 'le'),
    ]
    qmap = {feature: [0.25, 0.5, 0.75] for feature, _ in specs}
    qmap.update({
        'top_semid_share': [0.4, 0.5, 0.6, 0.7],
        'top_semid_margin': [0.4, 0.5, 0.6, 0.7],
        'semid_entropy': [0.3, 0.4, 0.5, 0.6],
    })
    single_conditions = []
    for feature, direction in specs:
        for threshold in feature_thresholds(val_df, feature, qmap[feature]):
            cond = OrderedDict([('feature', feature), ('direction', direction), ('threshold', threshold)])
            single_conditions.append(cond)
            configs.append(OrderedDict([
                ('name', f'{feature}_{direction}_{threshold:.6g}'),
                ('conditions', [cond]),
            ]))
    semantic_conds = [c for c in single_conditions if c['feature'] in {'top_semid_share', 'top_semid_margin', 'semid_entropy'}]
    residual_conds = [c for c in single_conditions if c['feature'] not in {'top_semid_share', 'top_semid_margin', 'semid_entropy'}]
    seen = set(c['name'] for c in configs)
    for sem in semantic_conds:
        for res in residual_conds:
            conditions = [sem, res]
            name = '__'.join(f"{c['feature']}_{c['direction']}_{c['threshold']:.6g}" for c in conditions)
            if name in seen:
                continue
            seen.add(name)
            configs.append(OrderedDict([('name', name), ('conditions', conditions)]))
    return configs


def select_config(rows, selection_k, max_ratio, min_net, max_open_rate):
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
        '# REES46 Stage P List-Residual Audit',
        '',
        f"- Selected rule: `{summary['selected_config']['name']}`",
        '- Target: user-action delta Hit@100, not direct estimation of displaced-item purchase probability.',
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
    parser = argparse.ArgumentParser(description='List/candidate residual audit for REES46 Stage P.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--semantic-source', default='semid_category_brand_context')
    parser.add_argument('--promotion-name', default='q5__sem50__slot10__all')
    parser.add_argument('--promotion-quota', type=int, default=5)
    parser.add_argument('--promotion-semantic-budget', type=int, default=50)
    parser.add_argument('--promotion-insert-start', type=int, default=10)
    parser.add_argument('--promotion-filter', default='all')
    parser.add_argument('--existing-weights', default='')
    parser.add_argument('--selection-k', type=int, default=100)
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--max-validation-ratio', type=float, default=0.0)
    parser.add_argument('--min-validation-net', type=int, default=20)
    parser.add_argument('--max-open-rate', type=float, default=0.7)
    parser.add_argument('--source-topk', type=int, default=500)
    parser.add_argument('--existing-pool-k', type=int, default=1000)
    parser.add_argument('--semantic-source-topk', type=int, default=500)
    parser.add_argument('--bucket-topk', type=int, default=2000)
    parser.add_argument('--val-context-frac', type=float, default=0.5)
    parser.add_argument('--val-target-types', default='cart')
    parser.add_argument('--event-view-weight', type=float, default=1.0)
    parser.add_argument('--event-cart-weight', type=float, default=3.0)
    parser.add_argument('--max-users', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    artifact = base.load_artifact(args.artifact)
    scheme, scope, semantic_source_name = exact.split_semantic_source(args.semantic_source)
    existing_weights = exact.parse_weights(args.existing_weights)
    promotion_config = OrderedDict([
        ('name', args.promotion_name),
        ('quota', args.promotion_quota),
        ('semantic_budget', args.promotion_semantic_budget),
        ('insert_start', args.promotion_insert_start),
        ('filter', args.promotion_filter),
    ])
    val_rows = build_phase_rows(artifact, args, 'val', scheme, scope, promotion_config, existing_weights)
    test_rows = build_phase_rows(artifact, args, 'test', scheme, scope, promotion_config, existing_weights)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    configs = build_configs(val_df)
    ks = parse_ints(args.ks)
    val_eval = [gate_eval.evaluate_config(val_df, config, ks) for config in configs]
    selected_val = select_config(
        val_eval,
        args.selection_k,
        args.max_validation_ratio,
        args.min_validation_net,
        args.max_open_rate,
    )
    selected_config = next(c for c in configs if c['name'] == selected_val['config'])
    selected_test = gate_eval.evaluate_config(test_df, selected_config, ks)
    key = f'@{args.selection_k}'
    test_ratio = selected_test['cannibalization_ratio'].get(key)
    gate_pass = selected_test['net_gain'].get(key, 0) > 0 and (test_ratio is None or test_ratio < 0.5)
    top_val = sorted(
        val_eval,
        key=lambda r: (
            r['net_gain'].get(key, 0),
            -(r['cannibalization_ratio'].get(key) if r['cannibalization_ratio'].get(key) is not None else 0.0),
            r['gross_recovery'].get(key, 0),
        ),
        reverse=True,
    )[:20]
    interpretation = (
        f"Selected `{selected_config['name']}` over action/list residual features. "
        f"Test net@{args.selection_k}={selected_test['net_gain'].get(key, 0)}, ratio={test_ratio}, gate_pass={gate_pass}. "
        "The supervised target is the realized promotion-action delta; displaced-item probability remains unobserved."
    )
    summary = OrderedDict([
        ('args', vars(args)),
        ('semantic_source', semantic_source_name),
        ('promotion_config', promotion_config),
        ('selected_config', selected_config),
        ('selected', OrderedDict([
            ('validation', selected_val),
            ('test', selected_test),
        ])),
        ('top_validation_rules', top_val),
        ('gate_pass', bool(gate_pass)),
        ('interpretation', interpretation),
    ])
    summary_json = os.path.join(args.output_dir, 'rees46_stage_p_list_residual_summary.json')
    summary_md = os.path.join(args.output_dir, 'rees46_stage_p_list_residual_summary.md')
    val_csv = os.path.join(args.output_dir, 'rees46_stage_p_list_residual_val_rows.csv')
    test_csv = os.path.join(args.output_dir, 'rees46_stage_p_list_residual_test_rows.csv')
    grid_csv = os.path.join(args.output_dir, 'rees46_stage_p_list_residual_val_grid.csv')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    pd.json_normalize(val_eval).to_csv(grid_csv, index=False)
    write_markdown(summary_md, args, summary)
    print(json.dumps({'summary': summary_json, 'markdown': summary_md, 'val_rows': val_csv, 'test_rows': test_csv, 'grid': grid_csv}, indent=2))


if __name__ == '__main__':
    main()
