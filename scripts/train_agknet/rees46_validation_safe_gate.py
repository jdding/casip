#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import pickle
from multiprocessing import Pool
from collections import Counter, OrderedDict, defaultdict

import pandas as pd

import rees46_no_training_baselines as base


DEFAULT_SOURCES = [
    'global_popularity',
    'global_purchase_popularity',
    'context_feedback_replay',
    'context_view_replay',
    'context_cart_replay',
    'context_event_weighted_replay',
    'seen_product_replay',
    'brand_overlap_seen_allowed',
    'brand_replacement_unseen',
    'category_overlap_seen_allowed',
    'category_or_brand_replacement_unseen',
    'category_code_family_replacement_unseen',
]

_WORKER_CONFIGS = None
_WORKER_KS = None
_WORKER_SOURCE_TOPK = None
_WORKER_INDEXED = None
_WORKER_GLOBAL_COUNTER = None
_WORKER_FALLBACK_COUNTER = None
_WORKER_PHASE = None
_WORKER_EVENT_WEIGHTS = None


def load_artifact(path):
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    if artifact.get('artifact_version') != 'rees46_protocol_v1':
        raise RuntimeError(f'Unsupported artifact version: {artifact.get("artifact_version")}')
    return artifact


def sort_events(events):
    return sorted(events, key=lambda x: (str(x.get('time') or ''), str(x.get('product_id') or '')))


def split_val_feedback(events, context_frac):
    ordered = sort_events(events)
    if len(ordered) < 2:
        return ordered, []
    n_context = int(len(ordered) * context_frac)
    n_context = max(1, min(n_context, len(ordered) - 1))
    return ordered[:n_context], ordered[n_context:]


def profile_from_events(history, context, target):
    seen_products = [x['product_id'] for x in history + context]
    history_products = [x['product_id'] for x in history]
    context_products = [x['product_id'] for x in context]
    context_view_products = [x['product_id'] for x in context if x.get('type') == 'view']
    context_cart_products = [x['product_id'] for x in context if x.get('type') == 'cart']
    categories = {str(x.get('category_id') or '') for x in history + context if x.get('category_id')}
    brands = {str(x.get('brand') or '') for x in history + context if x.get('brand')}
    category_codes = {str(x.get('category_code') or '') for x in history + context if x.get('category_code')}
    target_products = [x['product_id'] for x in target]
    return OrderedDict([
        ('seen_products', set(str(x) for x in seen_products)),
        ('history_products', set(str(x) for x in history_products)),
        ('feedback_products', set(str(x) for x in context_products)),
        ('feedback_view_products', set(str(x) for x in context_view_products)),
        ('feedback_cart_products', set(str(x) for x in context_cart_products)),
        ('categories', categories),
        ('brands', brands),
        ('category_codes', category_codes),
        ('test_products', target_products),
    ])


def make_source_lists(profile, indexed, global_counter, fallback_counter, event_weights=None):
    event_weights = event_weights or {'view': 1.0, 'cart': 3.0}
    active_set = indexed['active_set']
    seen_active = profile['seen_products'] & active_set
    history_active = profile['history_products'] & active_set
    feedback_active = profile['feedback_products'] & active_set
    feedback_view_active = profile.get('feedback_view_products', set()) & active_set
    feedback_cart_active = profile.get('feedback_cart_products', set()) & active_set
    event_weighted_counter = Counter()
    for product_id in feedback_view_active:
        event_weighted_counter[product_id] += float(event_weights.get('view', 1.0))
    for product_id in feedback_cart_active:
        event_weighted_counter[product_id] += float(event_weights.get('cart', 3.0))
    category_lists = [indexed['category_products'].get(c, []) for c in profile['categories']]
    brand_lists = [indexed['brand_products'].get(b, []) for b in profile['brands']]
    code_prefixes = {code.split('.')[0] for code in profile['category_codes'] if code}
    code_lists = [indexed['code_family_products'].get(c, []) for c in code_prefixes]
    rank_order = indexed['rank_order']
    seen_products = profile['seen_products']

    return OrderedDict([
        ('global_popularity', indexed['val_popularity']),
        ('global_purchase_popularity', indexed['purchase_oracle']),
        ('history_product_replay', base.rank_products(history_active, global_counter, fallback_counter)),
        ('context_feedback_replay', base.rank_products(feedback_active, global_counter, fallback_counter)),
        ('context_view_replay', base.rank_products(feedback_view_active, global_counter, fallback_counter)),
        ('context_cart_replay', base.rank_products(feedback_cart_active, global_counter, fallback_counter)),
        ('context_event_weighted_replay', base.rank_products(feedback_active, event_weighted_counter, global_counter)),
        ('seen_product_replay', base.rank_products(seen_active, global_counter, fallback_counter)),
        ('category_overlap_seen_allowed', base.merge_ranked_lists(category_lists, rank_order)),
        ('category_replacement_unseen', base.merge_ranked_lists(category_lists, rank_order, seen_products)),
        ('brand_overlap_seen_allowed', base.merge_ranked_lists(brand_lists, rank_order)),
        ('brand_replacement_unseen', base.merge_ranked_lists(brand_lists, rank_order, seen_products)),
        ('category_or_brand_replacement_unseen', base.merge_ranked_lists(category_lists + brand_lists, rank_order, seen_products)),
        ('category_code_family_replacement_unseen', base.merge_ranked_lists(code_lists, rank_order, seen_products)),
    ])


def blend_ranked_lists(source_lists, weights, rank_order, source_topk, max_recs):
    scores = defaultdict(float)
    best_rank = {}
    for name, weight in weights.items():
        if weight <= 0:
            continue
        for rank, product_id in enumerate(source_lists.get(name, [])[:source_topk]):
            if product_id not in best_rank or rank < best_rank[product_id]:
                best_rank[product_id] = rank
            scores[product_id] += float(weight) / float(rank + 1)
    ranked = sorted(
        scores,
        key=lambda p: (-scores[p], best_rank.get(p, source_topk), rank_order.get(p, len(rank_order)), p),
    )
    return ranked[:max_recs]


def first_hit_rank(recommendations, targets):
    target_set = set(str(x) for x in targets)
    for rank, product_id in enumerate(recommendations):
        if product_id in target_set:
            return rank
    return -1


def build_configs(source_names, fine_pop_context_step=0.0):
    configs = []
    for name in source_names:
        configs.append((name, OrderedDict([(name, 1.0)])))
    anchored_pairs = []
    if 'global_popularity' in source_names:
        anchored_pairs.extend(('global_popularity', name) for name in source_names if name != 'global_popularity')
    if 'global_purchase_popularity' in source_names:
        anchored_pairs.extend(
            ('global_purchase_popularity', name)
            for name in source_names
            if name != 'global_purchase_popularity'
        )
    anchored_pairs.extend([
        ('seen_product_replay', 'brand_replacement_unseen'),
        ('seen_product_replay', 'category_or_brand_replacement_unseen'),
        ('brand_overlap_seen_allowed', 'brand_replacement_unseen'),
        ('brand_overlap_seen_allowed', 'category_code_family_replacement_unseen'),
        ('category_overlap_seen_allowed', 'category_or_brand_replacement_unseen'),
    ])
    seen_pairs = set()
    for left, right in anchored_pairs:
        if left not in source_names or right not in source_names:
            continue
        key = tuple(sorted((left, right)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        for lw, rw in ((0.5, 0.5), (0.7, 0.3), (0.3, 0.7)):
            config_name = f'{left}_{lw:.1f}__{right}_{rw:.1f}'
            configs.append((config_name, OrderedDict([(left, lw), (right, rw)])))
    anchored = [
        ('global_popularity', 'brand_overlap_seen_allowed', 'category_code_family_replacement_unseen'),
        ('global_popularity', 'seen_product_replay', 'brand_replacement_unseen'),
        ('global_popularity', 'context_feedback_replay', 'category_or_brand_replacement_unseen'),
        ('seen_product_replay', 'brand_overlap_seen_allowed', 'category_or_brand_replacement_unseen'),
    ]
    for trio in anchored:
        if all(x in source_names for x in trio):
            weights = OrderedDict((name, 1.0 / len(trio)) for name in trio)
            configs.append(('__'.join(trio) + '__equal', weights))
    if fine_pop_context_step and 'global_popularity' in source_names and 'context_feedback_replay' in source_names:
        n_steps = int(round(1.0 / fine_pop_context_step))
        existing = {name for name, _ in configs}
        for i in range(n_steps + 1):
            pop_w = round(i * fine_pop_context_step, 6)
            ctx_w = round(1.0 - pop_w, 6)
            config_name = f'global_popularity_{pop_w:.3f}__context_feedback_replay_{ctx_w:.3f}'
            if config_name in existing:
                continue
            configs.append((config_name, OrderedDict([
                ('global_popularity', pop_w),
                ('context_feedback_replay', ctx_w),
            ])))
            existing.add(config_name)
    if fine_pop_context_step and 'global_purchase_popularity' in source_names and 'context_event_weighted_replay' in source_names:
        n_steps = int(round(1.0 / fine_pop_context_step))
        existing = {name for name, _ in configs}
        for i in range(n_steps + 1):
            pop_w = round(i * fine_pop_context_step, 6)
            ctx_w = round(1.0 - pop_w, 6)
            config_name = f'global_purchase_popularity_{pop_w:.3f}__context_event_weighted_replay_{ctx_w:.3f}'
            if config_name in existing:
                continue
            configs.append((config_name, OrderedDict([
                ('global_purchase_popularity', pop_w),
                ('context_event_weighted_replay', ctx_w),
            ])))
            existing.add(config_name)
    return configs


def summarize_rows(rows, ks):
    n = len(rows)
    out = OrderedDict([
        ('n_users', n),
        ('hit_counts', OrderedDict()),
        ('hit_rates', OrderedDict()),
        ('mean_first_hit_rank', None),
    ])
    ranks = [r['first_hit_rank'] for r in rows if r['first_hit_rank'] >= 0]
    for k in ks:
        hits = sum(1 for r in rows if 0 <= r['first_hit_rank'] < k)
        out['hit_counts'][f'hit@{k}'] = int(hits)
        out['hit_rates'][f'hit@{k}'] = float(hits / max(n, 1))
    if ranks:
        out['mean_first_hit_rank'] = float(sum(ranks) / len(ranks))
    return out


def evaluate_phase(records, configs, ks, source_topk, indexed, global_counter, fallback_counter, phase):
    return evaluate_phase_serial(records, configs, ks, source_topk, indexed, global_counter, fallback_counter, phase)


def evaluate_one_record(row_id, record, configs, ks, source_topk, indexed, global_counter, fallback_counter, phase, event_weights=None):
    max_recs = max(max(ks), source_topk)
    source_lists = make_source_lists(record['profile'], indexed, global_counter, fallback_counter, event_weights=event_weights)
    row = OrderedDict([
        ('row_id', row_id),
        ('user_id', record['user_id']),
        ('n_history', record['n_history']),
        ('n_context', record['n_context']),
        ('n_targets', len(record['targets'])),
        ('phase', phase),
    ])
    for config_name, weights in configs:
        recs = blend_ranked_lists(source_lists, weights, indexed['rank_order'], source_topk, max_recs)
        row[f'{config_name}_rank'] = int(first_hit_rank(recs, record['targets']))
    return row


def summarize_per_user_rows(per_user_rows, configs, ks):
    summaries = OrderedDict()
    for config_name, _ in configs:
        rows = [
            OrderedDict([
                ('user_id', row['user_id']),
                ('first_hit_rank', int(row[f'{config_name}_rank'])),
            ])
            for row in per_user_rows
        ]
        summaries[config_name] = summarize_rows(rows, ks)
    return summaries, per_user_rows


def evaluate_phase_serial(records, configs, ks, source_topk, indexed, global_counter, fallback_counter, phase, workers=1, event_weights=None):
    per_user_rows = []
    print(f'[{phase}] evaluating {len(records)} users, {len(configs)} configs, source_topk={source_topk}, workers=1', flush=True)
    for row_id, record in enumerate(records):
        if row_id and row_id % 1000 == 0:
            print(f'[{phase}] processed {row_id}/{len(records)} users', flush=True)
        per_user_rows.append(
            evaluate_one_record(
                row_id,
                record,
                configs,
                ks,
                source_topk,
                indexed,
                global_counter,
                fallback_counter,
                phase,
                event_weights=event_weights,
            )
        )
    return summarize_per_user_rows(per_user_rows, configs, ks)


def init_gate_worker(configs, ks, source_topk, indexed, global_counter, fallback_counter, phase, event_weights):
    global _WORKER_CONFIGS, _WORKER_KS, _WORKER_SOURCE_TOPK, _WORKER_INDEXED
    global _WORKER_GLOBAL_COUNTER, _WORKER_FALLBACK_COUNTER, _WORKER_PHASE, _WORKER_EVENT_WEIGHTS
    _WORKER_CONFIGS = configs
    _WORKER_KS = ks
    _WORKER_SOURCE_TOPK = source_topk
    _WORKER_INDEXED = indexed
    _WORKER_GLOBAL_COUNTER = global_counter
    _WORKER_FALLBACK_COUNTER = fallback_counter
    _WORKER_PHASE = phase
    _WORKER_EVENT_WEIGHTS = event_weights


def evaluate_worker_task(item):
    row_id, record = item
    return evaluate_one_record(
        row_id,
        record,
        _WORKER_CONFIGS,
        _WORKER_KS,
        _WORKER_SOURCE_TOPK,
        _WORKER_INDEXED,
        _WORKER_GLOBAL_COUNTER,
        _WORKER_FALLBACK_COUNTER,
        _WORKER_PHASE,
        event_weights=_WORKER_EVENT_WEIGHTS,
    )


def evaluate_phase_parallel(records, configs, ks, source_topk, indexed, global_counter, fallback_counter, phase, workers, event_weights=None):
    per_user_rows = []
    print(
        f'[{phase}] evaluating {len(records)} users, {len(configs)} configs, '
        f'source_topk={source_topk}, workers={workers}',
        flush=True,
    )
    with Pool(
        processes=workers,
        initializer=init_gate_worker,
        initargs=(configs, ks, source_topk, indexed, global_counter, fallback_counter, phase, event_weights or {'view': 1.0, 'cart': 3.0}),
    ) as pool:
        for count, row in enumerate(pool.imap(evaluate_worker_task, enumerate(records), chunksize=64), start=1):
            per_user_rows.append(row)
            if count % 1000 == 0:
                print(f'[{phase}] processed {count}/{len(records)} users', flush=True)
    return summarize_per_user_rows(per_user_rows, configs, ks)


def make_phase_records(users, phase, val_context_frac, val_target_types=None):
    records = []
    val_target_types = set(val_target_types or [])
    for record in users:
        if phase == 'val':
            context, targets = split_val_feedback(record['val_feedback'], val_context_frac)
            if val_target_types:
                targets = [x for x in targets if str(x.get('type') or '') in val_target_types]
            if not targets:
                continue
            history = record['history']
        elif phase == 'test':
            history = record['history']
            context = record['val_feedback']
            targets = record['test_purchases']
        else:
            raise ValueError(phase)
        profile = profile_from_events(history, context, targets)
        records.append(OrderedDict([
            ('user_id', record['user_id']),
            ('profile', profile),
            ('targets', [x['product_id'] for x in targets]),
            ('n_history', len(history)),
            ('n_context', len(context)),
        ]))
    return records


def select_config(validation_summaries, metric):
    metric_key = f'hit@{metric}'
    best_name = None
    best_tuple = None
    for name, summary in validation_summaries.items():
        hit10 = summary['hit_counts'].get(metric_key, 0)
        hit50 = summary['hit_counts'].get('hit@50', 0)
        mean_rank = summary['mean_first_hit_rank']
        tie_rank = mean_rank if mean_rank is not None else 10**9
        candidate = (hit10, hit50, -tie_rank)
        if best_tuple is None or candidate > best_tuple:
            best_tuple = candidate
            best_name = name
    return best_name


def main():
    parser = argparse.ArgumentParser(description='Validation-safe REES46 source blend gate.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--sources', default=','.join(DEFAULT_SOURCES))
    parser.add_argument('--source-topk', type=int, default=500)
    parser.add_argument('--val-context-frac', type=float, default=0.5)
    parser.add_argument('--selection-k', type=int, default=10)
    parser.add_argument('--max-users', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fine-pop-context-step', type=float, default=0.0)
    parser.add_argument('--val-target-types', default='', help='Comma-separated validation holdout event types, e.g. cart.')
    parser.add_argument('--event-view-weight', type=float, default=1.0)
    parser.add_argument('--event-cart-weight', type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    source_names = [x.strip() for x in args.sources.split(',') if x.strip()]
    configs = build_configs(source_names, args.fine_pop_context_step)

    artifact = load_artifact(args.artifact)
    users = artifact['users']
    if args.max_users:
        users = users[:args.max_users]

    catalogs = artifact['catalogs']['products']
    counters = {
        'pre_all': base.counter_from_top(artifact['popularity_top'].get('pre_all', [])),
        'val_implicit': base.counter_from_top(artifact['popularity_top'].get('val_implicit', [])),
        'val_all': base.counter_from_top(artifact['popularity_top'].get('val_all', [])),
        'pre_purchase': base.counter_from_top(artifact['purchase_popularity_top'].get('pre_purchase', [])),
        'val_purchase': base.counter_from_top(artifact['purchase_popularity_top'].get('val_purchase', [])),
    }
    meta = base.product_meta_maps(artifact['product_meta'])
    val_indexed = base.build_ranked_index(
        catalogs['val'],
        meta,
        counters['pre_all'],
        counters['pre_all'],
        counters['pre_all'],
        counters['pre_purchase'],
    )
    test_indexed = base.build_ranked_index(
        catalogs['test'],
        meta,
        counters['val_implicit'],
        counters['val_all'],
        counters['pre_all'],
        counters['val_purchase'],
    )

    val_target_types = [x.strip() for x in args.val_target_types.split(',') if x.strip()]
    val_records = make_phase_records(users, 'val', args.val_context_frac, val_target_types=val_target_types)
    test_records = make_phase_records(users, 'test', args.val_context_frac)
    print(
        f'Loaded artifact users={len(users)} val_records={len(val_records)} '
        f'test_records={len(test_records)} configs={len(configs)}',
        flush=True,
    )
    eval_fn = evaluate_phase_parallel if args.workers > 1 else evaluate_phase_serial
    event_weights = {'view': args.event_view_weight, 'cart': args.event_cart_weight}
    val_summaries, val_per_user = eval_fn(
        val_records,
        configs,
        ks,
        args.source_topk,
        val_indexed,
        counters['pre_all'],
        counters['pre_all'],
        'val',
        args.workers,
        event_weights=event_weights,
    )
    selected = select_config(val_summaries, args.selection_k)
    test_summaries, test_per_user = eval_fn(
        test_records,
        configs,
        ks,
        args.source_topk,
        test_indexed,
        counters['val_implicit'],
        counters['pre_all'],
        'test',
        args.workers,
        event_weights=event_weights,
    )

    config_weights = OrderedDict((name, weights) for name, weights in configs)
    top_val = sorted(
        val_summaries.items(),
        key=lambda kv: (
            kv[1]['hit_counts'].get(f'hit@{args.selection_k}', 0),
            kv[1]['hit_counts'].get('hit@50', 0),
        ),
        reverse=True,
    )[:15]
    top_test = sorted(
        test_summaries.items(),
        key=lambda kv: (
            kv[1]['hit_counts'].get(f'hit@{args.selection_k}', 0),
            kv[1]['hit_counts'].get('hit@50', 0),
        ),
        reverse=True,
    )[:15]

    summary = OrderedDict([
        ('args', vars(args)),
        ('artifact_args', artifact.get('args', {})),
        ('n_users_total', len(users)),
        ('n_val_users', len(val_records)),
        ('n_test_users', len(test_records)),
        ('n_configs', len(configs)),
        ('selected_by_val', selected),
        ('selected_weights', config_weights[selected]),
        ('selected_val_summary', val_summaries[selected]),
        ('selected_test_summary', test_summaries[selected]),
        ('top_val_configs', OrderedDict((name, val_summaries[name]) for name, _ in top_val)),
        ('top_test_configs_oracle_diagnostic', OrderedDict((name, test_summaries[name]) for name, _ in top_test)),
    ])

    summary_path = os.path.join(args.output_dir, 'rees46_validation_safe_gate_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(val_per_user).to_csv(os.path.join(args.output_dir, 'rees46_validation_safe_gate_val_per_user.csv'), index=False)
    pd.DataFrame(test_per_user).to_csv(os.path.join(args.output_dir, 'rees46_validation_safe_gate_test_per_user.csv'), index=False)

    md_path = os.path.join(args.output_dir, 'rees46_validation_safe_gate_summary.md')
    lines = [
        '# REES46 Validation-Safe Gate',
        '',
        f"- Artifact: `{args.artifact}`",
        f"- Users: validation `{len(val_records)}`, test `{len(test_records)}`",
        f"- Source top-k per component: `{args.source_topk}`",
        f"- Selected by validation Hit@{args.selection_k}: `{selected}`",
        f"- Selected weights: `{dict(config_weights[selected])}`",
        '',
        '## Selected Validation-To-Test Result',
        '',
        '| Split | Hit@10 | Hit@50 | Hit@100 | Hit@500 |',
        '|---|---:|---:|---:|---:|',
    ]
    for split, row in [('validation', val_summaries[selected]), ('test', test_summaries[selected])]:
        lines.append(
            f"| {split} | {row['hit_counts'].get('hit@10', 0)} | "
            f"{row['hit_counts'].get('hit@50', 0)} | {row['hit_counts'].get('hit@100', 0)} | "
            f"{row['hit_counts'].get('hit@500', 0)} |"
        )
    lines.extend([
        '',
        '## Top Validation Configs',
        '',
        '| Config | Hit@10 | Hit@50 | Hit@100 | Hit@500 |',
        '|---|---:|---:|---:|---:|',
    ])
    for name, _ in top_val:
        row = val_summaries[name]
        lines.append(
            f"| `{name}` | {row['hit_counts'].get('hit@10', 0)} | "
            f"{row['hit_counts'].get('hit@50', 0)} | {row['hit_counts'].get('hit@100', 0)} | "
            f"{row['hit_counts'].get('hit@500', 0)} |"
        )
    lines.extend([
        '',
        '## Top Test Configs (Oracle Diagnostic Only)',
        '',
        'These rows are sorted with test labels and are not deployable selection evidence.',
        '',
        '| Config | Hit@10 | Hit@50 | Hit@100 | Hit@500 |',
        '|---|---:|---:|---:|---:|',
    ])
    for name, _ in top_test:
        row = test_summaries[name]
        lines.append(
            f"| `{name}` | {row['hit_counts'].get('hit@10', 0)} | "
            f"{row['hit_counts'].get('hit@50', 0)} | {row['hit_counts'].get('hit@100', 0)} | "
            f"{row['hit_counts'].get('hit@500', 0)} |"
        )
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(json.dumps({'summary': summary_path, 'markdown': md_path}, indent=2))


if __name__ == '__main__':
    main()
