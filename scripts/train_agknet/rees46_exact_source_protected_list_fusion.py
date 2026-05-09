#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import OrderedDict

import pandas as pd

import rees46_no_training_baselines as base
import rees46_semantic_bridge_audit as semantic
import rees46_validation_safe_gate as gate


DEFAULT_KS = [10, 50, 100, 500]
DEFAULT_EXISTING_WEIGHTS = OrderedDict([
    ('global_purchase_popularity', 0.025),
    ('context_event_weighted_replay', 0.975),
])


def parse_ints(text):
    return [int(x) for x in text.split(',') if x.strip()]


def parse_weights(text):
    if not text.strip():
        return DEFAULT_EXISTING_WEIGHTS.copy()
    weights = OrderedDict()
    for part in text.split(','):
        if not part.strip():
            continue
        name, value = part.split(':', 1)
        weights[name.strip()] = float(value)
    return weights


def split_semantic_source(name):
    name = name.strip()
    if name.endswith('_rank'):
        name = name[:-5]
    if not name.startswith('semid_'):
        raise ValueError(f'Semantic source must start with semid_: {name}')
    body = name[len('semid_'):]
    for scope in ['history_context', 'context', 'history']:
        suffix = f'_{scope}'
        if body.endswith(suffix):
            return body[:-len(suffix)], scope, name
    raise ValueError(f'Cannot parse semantic source scope from {name}')


def make_records(users, phase, val_context_frac, val_target_types):
    records = []
    val_target_types = set(val_target_types or [])
    for record in users:
        if phase == 'val':
            context, targets = gate.split_val_feedback(record['val_feedback'], val_context_frac)
            if val_target_types:
                targets = [x for x in targets if str(x.get('type') or '') in val_target_types]
            if not targets:
                continue
            history = record['history']
        elif phase == 'test':
            history = record['history']
            context = gate.sort_events(record['val_feedback'])
            targets = record['test_purchases']
        else:
            raise ValueError(phase)
        profile = gate.profile_from_events(history, context, targets)
        semantic_record = dict(record)
        semantic_record['history'] = history
        semantic_record['val_feedback'] = context
        records.append(OrderedDict([
            ('user_id', str(record['user_id'])),
            ('profile', profile),
            ('semantic_record', semantic_record),
            ('targets', [str(x['product_id']) for x in targets]),
            ('n_history', len(history)),
            ('n_context', len(context)),
        ]))
    return records


def build_phase_state(artifact, phase, active_catalog_window):
    catalogs = artifact['catalogs']['products']
    active_products = catalogs[active_catalog_window]
    counters = {
        'pre_all': base.counter_from_top(artifact['popularity_top'].get('pre_all', [])),
        'val_implicit': base.counter_from_top(artifact['popularity_top'].get('val_implicit', [])),
        'val_all': base.counter_from_top(artifact['popularity_top'].get('val_all', [])),
        'pre_purchase': base.counter_from_top(artifact['purchase_popularity_top'].get('pre_purchase', [])),
        'val_purchase': base.counter_from_top(artifact['purchase_popularity_top'].get('val_purchase', [])),
    }
    meta_for_existing = base.product_meta_maps(artifact['product_meta'])
    if phase == 'val':
        indexed = base.build_ranked_index(
            active_products,
            meta_for_existing,
            counters['pre_all'],
            counters['pre_all'],
            counters['pre_all'],
            counters['pre_purchase'],
        )
        global_counter = counters['pre_all']
        fallback_counter = counters['pre_all']
        sem_pop_purchase = counters['pre_purchase']
        sem_pop_implicit = counters['pre_all']
        sem_pop_all = counters['pre_all']
    else:
        indexed = base.build_ranked_index(
            active_products,
            meta_for_existing,
            counters['val_implicit'],
            counters['val_all'],
            counters['pre_all'],
            counters['val_purchase'],
        )
        global_counter = counters['val_implicit']
        fallback_counter = counters['pre_all']
        sem_pop_purchase = counters['val_purchase']
        sem_pop_implicit = counters['val_implicit']
        sem_pop_all = counters['val_all']
    sem_meta = semantic.normalize_meta(artifact['product_meta'])
    return OrderedDict([
        ('active_products', active_products),
        ('indexed', indexed),
        ('global_counter', global_counter),
        ('fallback_counter', fallback_counter),
        ('sem_meta', sem_meta),
        ('sem_pop_purchase', sem_pop_purchase),
        ('sem_pop_implicit', sem_pop_implicit),
        ('sem_pop_pre', counters['pre_all']),
        ('sem_pop_all', sem_pop_all),
    ])


def first_hit_rank(candidates, targets):
    target_set = set(str(t) for t in targets)
    for rank, product_id in enumerate(candidates):
        if product_id in target_set:
            return rank
    return -1


def fuse_lists(existing_list, semantic_list, alpha, beta, budget, max_k):
    existing_rank = {str(p): idx for idx, p in enumerate(existing_list[:max_k])}
    semantic_rank = {
        str(p): idx
        for idx, p in enumerate(semantic_list)
        if budget <= 0 or idx < budget
    }
    all_items = set(existing_rank) | set(semantic_rank)
    rows = []
    for item in all_items:
        er = existing_rank.get(item)
        sr = semantic_rank.get(item)
        existing_key = er if er is not None else 10**9
        semantic_key = int(math.floor(beta + alpha * sr)) if sr is not None else 10**9
        fused_key = min(existing_key, semantic_key)
        in_both = er is not None and sr is not None
        source_priority = 0 if er is not None else 1
        rows.append((fused_key, source_priority, existing_key, semantic_key, sr if sr is not None else 10**9, item, in_both))
    rows.sort()
    fused = [row[5] for row in rows[:max_k]]
    provenance = {
        row[5]: OrderedDict([
            ('existing_rank', None if row[2] == 10**9 else int(row[2])),
            ('semantic_rank', None if row[4] == 10**9 else int(row[4])),
            ('fused_key', int(row[0])),
            ('in_both', bool(row[6])),
            ('semantic_improved_key', bool(row[3] < row[2])),
        ])
        for row in rows[:max_k]
    }
    return fused, provenance


def evaluate_record(record, state, scheme, scope, bucket_ranks, existing_weights, args, phase):
    max_k = max(parse_ints(args.ks))
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
        max(max_k, args.source_topk),
    )
    semantic_list = semantic.candidate_list_for_user(
        record['semantic_record'],
        scheme,
        scope,
        state['sem_meta'],
        bucket_ranks,
        args.semantic_source_topk,
    )
    fused_list, provenance = fuse_lists(
        existing_list,
        semantic_list,
        args.alpha,
        args.beta,
        args.budget,
        max(max_k, args.source_topk),
    )
    targets = record['targets']
    existing_rank = first_hit_rank(existing_list, targets)
    semantic_rank = first_hit_rank(semantic_list, targets)
    fused_rank = first_hit_rank(fused_list, targets)
    row = OrderedDict([
        ('user_id', record['user_id']),
        ('phase', phase),
        ('n_history', record['n_history']),
        ('n_context', record['n_context']),
        ('n_targets', len(targets)),
        ('existing_rank', int(existing_rank)),
        ('semantic_rank', int(semantic_rank)),
        ('fused_rank', int(fused_rank)),
        ('existing_candidate_count', int(len(existing_list))),
        ('semantic_candidate_count', int(len(semantic_list))),
    ])
    for k in parse_ints(args.ks):
        existing_top = set(existing_list[:k])
        semantic_top = set(semantic_list[:k])
        fused_top = set(fused_list[:k])
        base_hit = 0 <= existing_rank < k
        semantic_hit = 0 <= semantic_rank < k
        fused_hit = 0 <= fused_rank < k
        inserted = fused_top - existing_top
        displaced = existing_top - fused_top
        collisions = existing_top & semantic_top
        row[f'existing_hit@{k}'] = int(base_hit)
        row[f'semantic_hit@{k}'] = int(semantic_hit)
        row[f'fused_hit@{k}'] = int(fused_hit)
        row[f'gross_recovery@{k}'] = int((not base_hit) and fused_hit)
        row[f'cannibalized_hit@{k}'] = int(base_hit and (not fused_hit))
        row[f'net_gain@{k}'] = int(fused_hit) - int(base_hit)
        row[f'inserted_items@{k}'] = int(len(inserted))
        row[f'displaced_existing_items@{k}'] = int(len(displaced))
        row[f'collision_items@{k}'] = int(len(collisions))
        row[f'target_inserted@{k}'] = int(any(t in inserted for t in targets))
        row[f'target_displaced@{k}'] = int(any(t in displaced for t in targets))
    target_prov = [provenance[t] for t in targets if t in provenance]
    if target_prov:
        row['target_in_both'] = int(any(p['in_both'] for p in target_prov))
        row['target_semantic_improved_key'] = int(any(p['semantic_improved_key'] for p in target_prov))
    else:
        row['target_in_both'] = 0
        row['target_semantic_improved_key'] = 0
    return row


def summarize_rows(rows, ks):
    summary = OrderedDict()
    n = len(rows)
    summary['n_users'] = int(n)
    for system, rank_col in [('existing', 'existing_rank'), ('semantic', 'semantic_rank'), ('fused', 'fused_rank')]:
        ranks = [int(r[rank_col]) for r in rows]
        found = [r for r in ranks if r >= 0]
        system_summary = OrderedDict([
            ('hit_counts', OrderedDict()),
            ('hit_rates', OrderedDict()),
            ('mean_first_hit_rank', float(sum(found) / len(found)) if found else None),
        ])
        for k in ks:
            hits = sum(1 for r in ranks if 0 <= r < k)
            system_summary['hit_counts'][f'hit@{k}'] = int(hits)
            system_summary['hit_rates'][f'hit@{k}'] = float(hits / max(n, 1))
        summary[system] = system_summary
    displacement = OrderedDict()
    for k in ks:
        gross = sum(int(r[f'gross_recovery@{k}']) for r in rows)
        cannibal = sum(int(r[f'cannibalized_hit@{k}']) for r in rows)
        inserted = sum(int(r[f'inserted_items@{k}']) for r in rows)
        displaced = sum(int(r[f'displaced_existing_items@{k}']) for r in rows)
        displacement[f'@{k}'] = OrderedDict([
            ('gross_recovery', int(gross)),
            ('cannibalized_hit', int(cannibal)),
            ('net_gain', int(gross - cannibal)),
            ('inserted_items_total', int(inserted)),
            ('displaced_existing_items_total', int(displaced)),
            ('mean_inserted_items_per_user', float(inserted / max(n, 1))),
            ('mean_displaced_existing_items_per_user', float(displaced / max(n, 1))),
            ('net_gain_per_100_inserted_items', float((gross - cannibal) * 100.0 / inserted) if inserted else None),
            ('target_inserted', int(sum(int(r[f'target_inserted@{k}']) for r in rows))),
            ('target_displaced', int(sum(int(r[f'target_displaced@{k}']) for r in rows))),
            ('collision_items_total', int(sum(int(r[f'collision_items@{k}']) for r in rows))),
        ])
    summary['displacement'] = displacement
    summary['target_collision'] = OrderedDict([
        ('target_in_both_count', int(sum(int(r['target_in_both']) for r in rows))),
        ('target_semantic_improved_key_count', int(sum(int(r['target_semantic_improved_key']) for r in rows))),
    ])
    return summary


def evaluate_phase(artifact, args, phase, scheme, scope, existing_weights):
    active_window = 'val' if phase == 'val' else 'test'
    state = build_phase_state(artifact, phase, active_window)
    bucket_ranks, bucket_sizes = semantic.build_bucket_ranks(
        state['active_products'],
        state['sem_meta'],
        [scheme],
        state['sem_pop_purchase'],
        state['sem_pop_implicit'],
        state['sem_pop_pre'],
        state['sem_pop_all'],
        args.bucket_topk,
    )
    users = artifact['users']
    if args.max_users:
        users = users[:args.max_users]
    records = make_records(
        users,
        phase,
        args.val_context_frac,
        [x.strip() for x in args.val_target_types.split(',') if x.strip()],
    )
    rows = []
    for idx, record in enumerate(records):
        if idx and idx % 1000 == 0:
            print(f'[{phase}] processed {idx}/{len(records)} users', flush=True)
        rows.append(evaluate_record(record, state, scheme, scope, bucket_ranks, existing_weights, args, phase))
    return rows, OrderedDict([
        ('active_catalog_size', len(state['active_products'])),
        ('bucket_stats', OrderedDict([
            ('n_buckets', len(bucket_sizes[scheme])),
            ('mean_bucket_size', float(sum(bucket_sizes[scheme].values()) / max(len(bucket_sizes[scheme]), 1))),
            ('max_bucket_size', int(max(bucket_sizes[scheme].values()) if bucket_sizes[scheme] else 0)),
        ])),
    ])


def write_markdown(path, args, summary):
    selected = summary['config']
    lines = [
        '# REES46 Exact Source-Protected List Fusion',
        '',
        f"- Artifact: `{args.artifact}`",
        f"- Semantic source: `{selected['semantic_source']}`",
        f"- Existing weights: `{selected['existing_weights']}`",
        f"- Alpha/beta/budget: `{selected['alpha']}` / `{selected['beta']}` / `{selected['budget']}`",
        f"- Collision policy: `min-key, no boost`; exact ties keep existing items first.",
        '',
        '## Hit Metrics',
        '',
        '| Split | System | Hit@10 | Hit@50 | Hit@100 | Hit@500 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for split in ['validation', 'test']:
        for system in ['existing', 'semantic', 'fused']:
            hits = summary[split][system]['hit_counts']
            lines.append(
                f"| {split} | {system} | {hits.get('hit@10', 0)} | {hits.get('hit@50', 0)} | "
                f"{hits.get('hit@100', 0)} | {hits.get('hit@500', 0)} |"
            )
    lines.extend([
        '',
        '## Displacement Accounting',
        '',
        '| Split | K | Gross recovery | Cannibalized hit | Net gain | Inserted items | Displaced existing | Net gain / 100 inserted | Target inserted | Target displaced |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for split in ['validation', 'test']:
        for k, row in summary[split]['displacement'].items():
            ratio = row['net_gain_per_100_inserted_items']
            ratio_text = '' if ratio is None else f'{ratio:.3f}'
            lines.append(
                f"| {split} | {k[1:]} | {row['gross_recovery']} | {row['cannibalized_hit']} | "
                f"{row['net_gain']} | {row['inserted_items_total']} | {row['displaced_existing_items_total']} | "
                f"{ratio_text} | {row['target_inserted']} | {row['target_displaced']} |"
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
    parser = argparse.ArgumentParser(description='Exact source-protected candidate-list fusion for REES46 Stage M.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--semantic-source', default='semid_category_brand_context')
    parser.add_argument('--existing-weights', default='')
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--budget', type=int, default=500)
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
    scheme, scope, semantic_source_name = split_semantic_source(args.semantic_source)
    existing_weights = parse_weights(args.existing_weights)
    artifact = base.load_artifact(args.artifact)

    val_rows, val_meta = evaluate_phase(artifact, args, 'val', scheme, scope, existing_weights)
    test_rows, test_meta = evaluate_phase(artifact, args, 'test', scheme, scope, existing_weights)
    val_summary = summarize_rows(val_rows, ks)
    test_summary = summarize_rows(test_rows, ks)
    delta10 = test_summary['fused']['hit_counts'].get('hit@10', 0) - test_summary['existing']['hit_counts'].get('hit@10', 0)
    cannibal10 = test_summary['displacement']['@10']['cannibalized_hit']
    gross10 = test_summary['displacement']['@10']['gross_recovery']
    interpretation = (
        f"Exact list fusion changes test Hit@10 by {delta10:+d}: gross recovery {gross10}, "
        f"cannibalized existing hits {cannibal10}. Because this evaluates actual fixed-length Top-K lists, "
        "the net gain includes displacement/cannibalization and is the relevant replacement for the earlier rank-only proxy."
    )
    summary = OrderedDict([
        ('args', vars(args)),
        ('config', OrderedDict([
            ('semantic_source', semantic_source_name),
            ('scheme', scheme),
            ('scope', scope),
            ('existing_weights', existing_weights),
            ('alpha', args.alpha),
            ('beta', args.beta),
            ('budget', args.budget),
        ])),
        ('validation_meta', val_meta),
        ('test_meta', test_meta),
        ('validation', val_summary),
        ('test', test_summary),
        ('interpretation', interpretation),
    ])

    summary_json = os.path.join(args.output_dir, 'rees46_exact_source_protected_list_fusion_summary.json')
    summary_md = os.path.join(args.output_dir, 'rees46_exact_source_protected_list_fusion_summary.md')
    val_path = os.path.join(args.output_dir, 'rees46_exact_source_protected_list_fusion_val_per_user.csv')
    test_path = os.path.join(args.output_dir, 'rees46_exact_source_protected_list_fusion_test_per_user.csv')
    pd.DataFrame(val_rows).to_csv(val_path, index=False)
    pd.DataFrame(test_rows).to_csv(test_path, index=False)
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    write_markdown(summary_md, args, summary)
    print(json.dumps({'summary': summary_json, 'markdown': summary_md, 'val_per_user': val_path, 'test_per_user': test_path}, indent=2))


if __name__ == '__main__':
    main()
