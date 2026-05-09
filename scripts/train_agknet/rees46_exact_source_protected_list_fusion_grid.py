#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict
from types import SimpleNamespace

import pandas as pd

import rees46_exact_source_protected_list_fusion as exact
import rees46_no_training_baselines as base
import rees46_semantic_bridge_audit as semantic
import rees46_validation_safe_gate as gate


def parse_ints(text):
    return [int(x) for x in text.split(',') if x.strip()]


def parse_floats(text):
    return [float(x) for x in text.split(',') if x.strip()]


def init_stats(config, ks):
    return OrderedDict([
        ('config', config['name']),
        ('alpha', config['alpha']),
        ('beta', config['beta']),
        ('budget', config['budget']),
        ('n_users', 0),
        ('rank_sum', 0.0),
        ('rank_found_count', 0),
        ('hit_counts', OrderedDict((f'hit@{k}', 0) for k in ks)),
        ('gross_recovery', OrderedDict((f'@{k}', 0) for k in ks)),
        ('cannibalized_hit', OrderedDict((f'@{k}', 0) for k in ks)),
        ('inserted_items_total', OrderedDict((f'@{k}', 0) for k in ks)),
        ('target_inserted', OrderedDict((f'@{k}', 0) for k in ks)),
        ('target_displaced', OrderedDict((f'@{k}', 0) for k in ks)),
    ])


def update_stats(stats, existing_list, semantic_list, targets, config, ks, max_k):
    existing_rank = exact.first_hit_rank(existing_list, targets)
    if config['budget'] == 0:
        fused_list = existing_list[:max_k]
    else:
        fused_list, _ = exact.fuse_lists(
            existing_list,
            semantic_list,
            config['alpha'],
            config['beta'],
            config['budget'],
            max_k,
        )
    fused_rank = exact.first_hit_rank(fused_list, targets)
    stats['n_users'] += 1
    if fused_rank >= 0:
        stats['rank_sum'] += fused_rank
        stats['rank_found_count'] += 1
    for k in ks:
        existing_top = set(existing_list[:k])
        fused_top = set(fused_list[:k])
        base_hit = 0 <= existing_rank < k
        fused_hit = 0 <= fused_rank < k
        inserted = fused_top - existing_top
        displaced = existing_top - fused_top
        stats['hit_counts'][f'hit@{k}'] += int(fused_hit)
        stats['gross_recovery'][f'@{k}'] += int((not base_hit) and fused_hit)
        stats['cannibalized_hit'][f'@{k}'] += int(base_hit and (not fused_hit))
        stats['inserted_items_total'][f'@{k}'] += len(inserted)
        stats['target_inserted'][f'@{k}'] += int(any(t in inserted for t in targets))
        stats['target_displaced'][f'@{k}'] += int(any(t in displaced for t in targets))


def finalize_stats(stats, ks):
    n = max(int(stats['n_users']), 1)
    out = OrderedDict(stats)
    out['hit_rates'] = OrderedDict((f'hit@{k}', out['hit_counts'][f'hit@{k}'] / n) for k in ks)
    out['mean_first_hit_rank'] = (
        float(out['rank_sum'] / out['rank_found_count'])
        if out['rank_found_count'] else None
    )
    out['net_gain'] = OrderedDict()
    out['net_gain_per_100_inserted_items'] = OrderedDict()
    for k in ks:
        key = f'@{k}'
        net = out['gross_recovery'][key] - out['cannibalized_hit'][key]
        inserted = out['inserted_items_total'][key]
        out['net_gain'][key] = int(net)
        out['net_gain_per_100_inserted_items'][key] = float(net * 100.0 / inserted) if inserted else None
    del out['rank_sum']
    del out['rank_found_count']
    return out


def config_name(alpha, beta, budget):
    if budget == 0:
        return 'existing_only'
    return f'a{alpha:g}__b{beta:g}__budget{budget}'


def build_configs(alphas, betas, budgets, include_existing_only):
    configs = []
    if include_existing_only:
        configs.append(OrderedDict([('name', 'existing_only'), ('alpha', 0.0), ('beta', 0.0), ('budget', 0)]))
    for alpha in alphas:
        for beta in betas:
            for budget in budgets:
                configs.append(OrderedDict([
                    ('name', config_name(alpha, beta, budget)),
                    ('alpha', alpha),
                    ('beta', beta),
                    ('budget', budget),
                ]))
    return configs


def validation_grid(artifact, args, scheme, scope, existing_weights, configs, ks):
    state = exact.build_phase_state(artifact, 'val', 'val')
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
    users = artifact['users'][:args.max_users] if args.max_users else artifact['users']
    records = exact.make_records(
        users,
        'val',
        args.val_context_frac,
        [x.strip() for x in args.val_target_types.split(',') if x.strip()],
    )
    max_k = max(ks)
    max_list = max(max_k, args.source_topk)
    stats = OrderedDict((c['name'], init_stats(c, ks)) for c in configs)
    for idx, record in enumerate(records):
        if idx and idx % 1000 == 0:
            print(f'[val-grid] processed {idx}/{len(records)} users', flush=True)
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
        for config in configs:
            update_stats(stats[config['name']], existing_list, semantic_list, record['targets'], config, ks, max_list)
    return [finalize_stats(v, ks) for v in stats.values()]


def selection_key(row, selection_k):
    key = f'@{selection_k}'
    return (
        row['hit_counts'].get(f'hit@{selection_k}', 0),
        row['hit_counts'].get('hit@50', 0),
        row['hit_counts'].get('hit@100', 0),
        -row['cannibalized_hit'].get(key, 0),
        row['net_gain'].get(key, 0),
        -(row['mean_first_hit_rank'] if row['mean_first_hit_rank'] is not None else 10**9),
    )


def write_markdown(path, args, summary):
    selected = summary['selected_config']
    lines = [
        '# REES46 Exact List-Fusion Grid Selection',
        '',
        f"- Semantic source: `{summary['semantic_source']}`",
        f"- Selected on validation Hit@{args.selection_k}: `{selected['config']}`",
        f"- Selected alpha/beta/budget: `{selected['alpha']}` / `{selected['beta']}` / `{selected['budget']}`",
        '',
        '## Selected Exact Result',
        '',
        '| Split | System | Hit@10 | Hit@50 | Hit@100 | Hit@500 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for split in ['validation', 'test']:
        for system in ['existing', 'semantic', 'fused']:
            hits = summary['selected_exact'][split][system]['hit_counts']
            lines.append(
                f"| {split} | {system} | {hits.get('hit@10', 0)} | {hits.get('hit@50', 0)} | "
                f"{hits.get('hit@100', 0)} | {hits.get('hit@500', 0)} |"
            )
    lines.extend([
        '',
        '## Selected Displacement',
        '',
        '| Split | K | Gross recovery | Cannibalized hit | Net gain | Inserted items | Net gain / 100 inserted |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for split in ['validation', 'test']:
        for k, row in summary['selected_exact'][split]['displacement'].items():
            ratio = row['net_gain_per_100_inserted_items']
            ratio_text = '' if ratio is None else f'{ratio:.3f}'
            lines.append(
                f"| {split} | {k[1:]} | {row['gross_recovery']} | {row['cannibalized_hit']} | "
                f"{row['net_gain']} | {row['inserted_items_total']} | {ratio_text} |"
            )
    lines.extend([
        '',
        '## Top Validation Configs',
        '',
        '| Config | Hit@10 | Hit@50 | Hit@100 | Gross@10 | Cannibal@10 | Net@10 | Inserted@10 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in summary['top_validation_configs']:
        lines.append(
            f"| `{row['config']}` | {row['hit_counts'].get('hit@10', 0)} | "
            f"{row['hit_counts'].get('hit@50', 0)} | {row['hit_counts'].get('hit@100', 0)} | "
            f"{row['gross_recovery'].get('@10', 0)} | {row['cannibalized_hit'].get('@10', 0)} | "
            f"{row['net_gain'].get('@10', 0)} | {row['inserted_items_total'].get('@10', 0)} |"
        )
    lines.extend([
        '',
        '## Interpretation',
        '',
        summary['interpretation'],
    ])
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Validation-selected exact list fusion grid for REES46 Stage M.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--semantic-source', default='semid_category_brand_context')
    parser.add_argument('--existing-weights', default='')
    parser.add_argument('--alphas', default='0.25,0.5,0.75,1.0,1.5,2.0')
    parser.add_argument('--betas', default='0,5,10,20,50,100')
    parser.add_argument('--budgets', default='5,10,25,50,100,500')
    parser.add_argument('--include-existing-only', action='store_true')
    parser.add_argument('--selection-k', type=int, default=10)
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--source-topk', type=int, default=500)
    parser.add_argument('--semantic-source-topk', type=int, default=500)
    parser.add_argument('--bucket-topk', type=int, default=2000)
    parser.add_argument('--val-context-frac', type=float, default=0.5)
    parser.add_argument('--val-target-types', default='cart')
    parser.add_argument('--event-view-weight', type=float, default=1.0)
    parser.add_argument('--event-cart-weight', type=float, default=3.0)
    parser.add_argument('--max-users', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ks = parse_ints(args.ks)
    scheme, scope, semantic_source_name = exact.split_semantic_source(args.semantic_source)
    existing_weights = exact.parse_weights(args.existing_weights)
    configs = build_configs(parse_floats(args.alphas), parse_floats(args.betas), parse_ints(args.budgets), args.include_existing_only)
    artifact = base.load_artifact(args.artifact)
    grid_rows = validation_grid(artifact, args, scheme, scope, existing_weights, configs, ks)
    selected_grid = max(grid_rows, key=lambda r: selection_key(r, args.selection_k))
    selected_args = SimpleNamespace(**vars(args))
    selected_args.alpha = float(selected_grid['alpha'])
    selected_args.beta = float(selected_grid['beta'])
    selected_args.budget = int(selected_grid['budget'])
    val_rows, val_meta = exact.evaluate_phase(artifact, selected_args, 'val', scheme, scope, existing_weights)
    test_rows, test_meta = exact.evaluate_phase(artifact, selected_args, 'test', scheme, scope, existing_weights)
    selected_exact = OrderedDict([
        ('validation_meta', val_meta),
        ('test_meta', test_meta),
        ('validation', exact.summarize_rows(val_rows, ks)),
        ('test', exact.summarize_rows(test_rows, ks)),
    ])
    top_grid = sorted(grid_rows, key=lambda r: selection_key(r, args.selection_k), reverse=True)[:20]
    test_delta10 = selected_exact['test']['fused']['hit_counts'].get('hit@10', 0) - selected_exact['test']['existing']['hit_counts'].get('hit@10', 0)
    test_disp10 = selected_exact['test']['displacement']['@10']
    interpretation = (
        f"Exact validation selected `{selected_grid['config']}`. On test, fused Hit@10 changes by {test_delta10:+d}, "
        f"with gross recovery {test_disp10['gross_recovery']} and cannibalized hits {test_disp10['cannibalized_hit']}. "
        "This selection accounts for fixed-length displacement, unlike the earlier rank-only proxy."
    )
    summary = OrderedDict([
        ('args', vars(args)),
        ('semantic_source', semantic_source_name),
        ('existing_weights', existing_weights),
        ('selected_config', OrderedDict([
            ('config', selected_grid['config']),
            ('alpha', float(selected_grid['alpha'])),
            ('beta', float(selected_grid['beta'])),
            ('budget', int(selected_grid['budget'])),
        ])),
        ('top_validation_configs', top_grid),
        ('selected_exact', selected_exact),
        ('interpretation', interpretation),
    ])
    summary_json = os.path.join(args.output_dir, 'rees46_exact_list_fusion_grid_summary.json')
    summary_md = os.path.join(args.output_dir, 'rees46_exact_list_fusion_grid_summary.md')
    grid_csv = os.path.join(args.output_dir, 'rees46_exact_list_fusion_grid.csv')
    val_csv = os.path.join(args.output_dir, 'rees46_exact_list_fusion_selected_val_per_user.csv')
    test_csv = os.path.join(args.output_dir, 'rees46_exact_list_fusion_selected_test_per_user.csv')
    pd.json_normalize(grid_rows).to_csv(grid_csv, index=False)
    pd.DataFrame(val_rows).to_csv(val_csv, index=False)
    pd.DataFrame(test_rows).to_csv(test_csv, index=False)
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    write_markdown(summary_md, args, summary)
    print(json.dumps({'summary': summary_json, 'markdown': summary_md, 'grid': grid_csv}, indent=2))


if __name__ == '__main__':
    main()
