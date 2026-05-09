#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict

import pandas as pd

import rees46_exact_source_protected_list_fusion as exact
import rees46_no_training_baselines as base
import rees46_semantic_bridge_audit as semantic
import rees46_validation_safe_gate as gate


def parse_ints(text):
    return [int(x) for x in text.split(',') if x.strip()]


def build_configs(quotas, semantic_budgets, insert_starts, filters, include_existing_only=True):
    configs = []
    if include_existing_only:
        configs.append(OrderedDict([
            ('name', 'existing_only'),
            ('quota', 0),
            ('semantic_budget', 0),
            ('insert_start', 10**9),
            ('filter', 'none'),
        ]))
    for quota in quotas:
        for semantic_budget in semantic_budgets:
            for insert_start in insert_starts:
                for mode in filters:
                    configs.append(OrderedDict([
                        ('name', f'q{quota}__sem{semantic_budget}__slot{insert_start}__{mode}'),
                        ('quota', quota),
                        ('semantic_budget', semantic_budget),
                        ('insert_start', insert_start),
                        ('filter', mode),
                    ]))
    return configs


def first_hit_rank(items, targets):
    target_set = set(str(t) for t in targets)
    for idx, item in enumerate(items):
        if item in target_set:
            return idx
    return -1


def candidate_allowed(item, mode, existing_rank, k):
    er = existing_rank.get(item)
    in_existing = er is not None
    if item in existing_rank and er < k:
        return False
    if mode == 'all':
        return True
    if mode == 'overlap':
        return in_existing and er >= k
    if mode == 'semantic_only':
        return not in_existing
    if mode == 'tail_overlap':
        return in_existing and er >= k and er < 500
    raise ValueError(f'Unknown filter: {mode}')


def promote(existing_list, semantic_list, config, k, max_eval_k):
    if config['quota'] <= 0:
        return existing_list[:max_eval_k]
    insert_start = min(int(config['insert_start']), k)
    existing_rank = {item: idx for idx, item in enumerate(existing_list[:max_eval_k])}
    selected = []
    for sr, item in enumerate(semantic_list[:int(config['semantic_budget'])]):
        item = str(item)
        if not candidate_allowed(item, config['filter'], existing_rank, k):
            continue
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= int(config['quota']):
            break
    if not selected:
        return existing_list[:max_eval_k]
    selected_set = set(selected)
    prefix = [x for x in existing_list[:insert_start] if x not in selected_set]
    suffix = [x for x in existing_list[insert_start:max_eval_k] if x not in selected_set]
    fused = prefix + selected + suffix
    return fused[:max_eval_k]


def init_stats(config, ks):
    return OrderedDict([
        ('config', config['name']),
        ('quota', int(config['quota'])),
        ('semantic_budget', int(config['semantic_budget'])),
        ('insert_start', int(config['insert_start'])),
        ('filter', config['filter']),
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


def update_stats(stats, existing_list, semantic_list, targets, config, ks, max_eval_k, promotion_k):
    fused_list = promote(existing_list, semantic_list, config, promotion_k, max_eval_k)
    existing_rank = first_hit_rank(existing_list[:max_eval_k], targets)
    fused_rank = first_hit_rank(fused_list, targets)
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
    out = OrderedDict(stats)
    n = max(int(out['n_users']), 1)
    out['hit_rates'] = OrderedDict((f'hit@{k}', out['hit_counts'][f'hit@{k}'] / n) for k in ks)
    out['mean_first_hit_rank'] = (
        float(out['rank_sum'] / out['rank_found_count'])
        if out['rank_found_count'] else None
    )
    out['net_gain'] = OrderedDict()
    out['cannibalization_ratio'] = OrderedDict()
    out['net_gain_per_100_inserted_items'] = OrderedDict()
    for k in ks:
        key = f'@{k}'
        gross = int(out['gross_recovery'][key])
        cannibal = int(out['cannibalized_hit'][key])
        inserted = int(out['inserted_items_total'][key])
        net = gross - cannibal
        out['net_gain'][key] = int(net)
        out['cannibalization_ratio'][key] = float(cannibal / gross) if gross else None
        out['net_gain_per_100_inserted_items'][key] = float(net * 100.0 / inserted) if inserted else None
    del out['rank_sum']
    del out['rank_found_count']
    return out


def build_user_lists(artifact, args, phase, scheme, scope, existing_weights):
    active_window = 'val' if phase == 'val' else 'test'
    state = exact.build_phase_state(artifact, phase, active_window)
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
        phase,
        args.val_context_frac,
        [x.strip() for x in args.val_target_types.split(',') if x.strip()],
    )
    max_eval_k = max(parse_ints(args.ks))
    max_list = max(max_eval_k, args.source_topk, args.existing_pool_k)
    rows = []
    for idx, record in enumerate(records):
        if idx and idx % 1000 == 0:
            print(f'[{phase}] built {idx}/{len(records)} user lists', flush=True)
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
        rows.append((existing_list[:max_list], semantic_list, record['targets']))
    return rows


def evaluate_configs(user_lists, configs, ks, promotion_k):
    max_eval_k = max(ks)
    stats = OrderedDict((config['name'], init_stats(config, ks)) for config in configs)
    for idx, (existing_list, semantic_list, targets) in enumerate(user_lists):
        if idx and idx % 1000 == 0:
            print(f'[eval] processed {idx}/{len(user_lists)} users', flush=True)
        for config in configs:
            update_stats(stats[config['name']], existing_list, semantic_list, targets, config, ks, max_eval_k, promotion_k)
    return [finalize_stats(v, ks) for v in stats.values()]


def selection_key(row, selection_k, base_hit10):
    key = f'@{selection_k}'
    hit10 = row['hit_counts'].get('hit@10', 0)
    violates_top10 = int(hit10 < base_hit10)
    cannibal_ratio = row['cannibalization_ratio'].get(key)
    cannibal_penalty = cannibal_ratio if cannibal_ratio is not None else 0.0
    return (
        -violates_top10,
        row['net_gain'].get(key, 0),
        row['hit_counts'].get(f'hit@{selection_k}', 0),
        -cannibal_penalty,
        -row['cannibalized_hit'].get(key, 0),
        row['gross_recovery'].get(key, 0),
    )


def write_markdown(path, args, summary):
    selected = summary['selected_config']
    lines = [
        '# REES46 Source-Aware Promotion Gate',
        '',
        f"- Semantic source: `{summary['semantic_source']}`",
        f"- Selection target: validation net gain @`{args.selection_k}` with Top-10 no-regression preference",
        f"- Selected: `{selected['name']}`",
        '',
        '## Selected Result',
        '',
        '| Split | Hit@10 | Hit@50 | Hit@100 | Hit@500 |',
        '|---|---:|---:|---:|---:|',
    ]
    for split in ['validation', 'test']:
        hits = summary['selected'][split]['hit_counts']
        lines.append(
            f"| {split} | {hits.get('hit@10', 0)} | {hits.get('hit@50', 0)} | "
            f"{hits.get('hit@100', 0)} | {hits.get('hit@500', 0)} |"
        )
    lines.extend([
        '',
        '## Net Gain Accounting',
        '',
        '| Split | K | Gross | Cannibalized | Net | Cannibal/Gross | Inserted | Net / 100 inserted |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for split in ['validation', 'test']:
        row = summary['selected'][split]
        for k in args.ks.split(','):
            key = f'@{k}'
            ratio = row['cannibalization_ratio'].get(key)
            ratio_text = '' if ratio is None else f'{ratio:.3f}'
            n100 = row['net_gain_per_100_inserted_items'].get(key)
            n100_text = '' if n100 is None else f'{n100:.3f}'
            lines.append(
                f"| {split} | {k} | {row['gross_recovery'].get(key, 0)} | "
                f"{row['cannibalized_hit'].get(key, 0)} | {row['net_gain'].get(key, 0)} | "
                f"{ratio_text} | {row['inserted_items_total'].get(key, 0)} | {n100_text} |"
            )
    lines.extend([
        '',
        '## Top Validation Policies',
        '',
        '| Policy | Hit@10 | Hit@100 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in summary['top_validation_policies']:
        ratio = row['cannibalization_ratio'].get(f"@{args.selection_k}")
        ratio_text = '' if ratio is None else f'{ratio:.3f}'
        lines.append(
            f"| `{row['config']}` | {row['hit_counts'].get('hit@10', 0)} | "
            f"{row['hit_counts'].get(f'hit@{args.selection_k}', 0)} | "
            f"{row['gross_recovery'].get(f'@{args.selection_k}', 0)} | "
            f"{row['cannibalized_hit'].get(f'@{args.selection_k}', 0)} | "
            f"{row['net_gain'].get(f'@{args.selection_k}', 0)} | {ratio_text} |"
        )
    lines.extend(['', '## Interpretation', '', summary['interpretation']])
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Source-aware constrained semantic promotion gate for REES46.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--semantic-source', default='semid_category_brand_context')
    parser.add_argument('--existing-weights', default='')
    parser.add_argument('--quotas', default='1,2,5')
    parser.add_argument('--semantic-budgets', default='5,10,25,50,100')
    parser.add_argument('--insert-starts', default='10,50,75,90')
    parser.add_argument('--filters', default='all,overlap,tail_overlap,semantic_only')
    parser.add_argument('--selection-k', type=int, default=100)
    parser.add_argument('--ks', default='10,50,100,500')
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
    ks = parse_ints(args.ks)
    scheme, scope, semantic_source_name = exact.split_semantic_source(args.semantic_source)
    existing_weights = exact.parse_weights(args.existing_weights)
    configs = build_configs(
        parse_ints(args.quotas),
        parse_ints(args.semantic_budgets),
        parse_ints(args.insert_starts),
        [x.strip() for x in args.filters.split(',') if x.strip()],
    )
    artifact = base.load_artifact(args.artifact)
    val_lists = build_user_lists(artifact, args, 'val', scheme, scope, existing_weights)
    val_rows = evaluate_configs(val_lists, configs, ks, args.selection_k)
    base_val = next(r for r in val_rows if r['config'] == 'existing_only')
    selected_val = max(val_rows, key=lambda r: selection_key(r, args.selection_k, base_val['hit_counts']['hit@10']))
    selected_config = next(c for c in configs if c['name'] == selected_val['config'])
    test_lists = build_user_lists(artifact, args, 'test', scheme, scope, existing_weights)
    existing_config = next(c for c in configs if c['name'] == 'existing_only')
    test_eval_rows = evaluate_configs(test_lists, [existing_config, selected_config], ks, args.selection_k)
    base_test = next(r for r in test_eval_rows if r['config'] == 'existing_only')
    selected_test = next(r for r in test_eval_rows if r['config'] == selected_config['name'])
    top_val = sorted(
        val_rows,
        key=lambda r: selection_key(r, args.selection_k, base_val['hit_counts']['hit@10']),
        reverse=True,
    )[:20]
    test_net = selected_test['net_gain'].get(f'@{args.selection_k}', 0)
    test_ratio = selected_test['cannibalization_ratio'].get(f'@{args.selection_k}')
    gate_pass = test_net > 0 and (test_ratio is None or test_ratio < 0.5) and selected_test['hit_counts']['hit@10'] >= base_test['hit_counts']['hit@10']
    interpretation = (
        f"Validation selected `{selected_config['name']}`. Test net@{args.selection_k} is {test_net} "
        f"with cannibal/gross ratio {test_ratio}. Gate pass={gate_pass}. "
        "This gate evaluates conservative physical promotion from the deep semantic pool, not a unified ranker."
    )
    summary = OrderedDict([
        ('args', vars(args)),
        ('semantic_source', semantic_source_name),
        ('existing_weights', existing_weights),
        ('selected_config', selected_config),
        ('base_validation', base_val),
        ('base_test', base_test),
        ('selected', OrderedDict([
            ('validation', selected_val),
            ('test', selected_test),
        ])),
        ('top_validation_policies', top_val),
        ('gate_pass', bool(gate_pass)),
        ('interpretation', interpretation),
    ])
    summary_json = os.path.join(args.output_dir, 'rees46_source_aware_promotion_gate_summary.json')
    summary_md = os.path.join(args.output_dir, 'rees46_source_aware_promotion_gate_summary.md')
    grid_csv = os.path.join(args.output_dir, 'rees46_source_aware_promotion_gate_val_grid.csv')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    pd.json_normalize(val_rows).to_csv(grid_csv, index=False)
    write_markdown(summary_md, args, summary)
    print(json.dumps({'summary': summary_json, 'markdown': summary_md, 'grid': grid_csv}, indent=2))


if __name__ == '__main__':
    main()
