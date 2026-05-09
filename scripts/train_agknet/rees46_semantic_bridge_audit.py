#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from collections import Counter, OrderedDict, defaultdict

import pandas as pd


DEFAULT_KS = [10, 50, 100, 500]
DEFAULT_SCHEMES = [
    'category',
    'brand',
    'code_family',
    'category_brand',
    'code_brand',
]
DEFAULT_SOURCE_SCOPES = ['history', 'context', 'history_context']
DEFAULT_EXISTING_COLS = [
    'global_popularity_rank',
    'global_purchase_popularity_rank',
    'context_feedback_replay_rank',
    'context_event_weighted_replay_rank',
    'context_cart_replay_rank',
    'context_view_replay_rank',
    'seen_product_replay_rank',
    'global_purchase_popularity_0.025__context_event_weighted_replay_0.975_rank',
]


def sort_events(events):
    return sorted(events, key=lambda x: (str(x.get('time') or ''), str(x.get('product_id') or '')))


def split_val_feedback(events, context_frac):
    ordered = sort_events(events)
    if len(ordered) < 2:
        return ordered, []
    n_context = int(len(ordered) * context_frac)
    n_context = max(1, min(n_context, len(ordered) - 1))
    return ordered[:n_context], ordered[n_context:]


def load_artifact(path):
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    if artifact.get('artifact_version') != 'rees46_protocol_v1':
        raise RuntimeError(f'Unsupported artifact version: {artifact.get("artifact_version")}')
    return artifact


def counter_from_top(rows):
    out = Counter()
    for product_id, count in rows:
        out[str(product_id)] = int(count)
    return out


def code_family(code):
    code = str(code or '')
    return code.split('.')[0] if code else ''


def normalize_meta(meta):
    out = {}
    for product_id, row in meta.items():
        pid = str(product_id)
        category = str(row.get('category_id') or '')
        brand = str(row.get('brand') or '')
        code = str(row.get('category_code') or '')
        out[pid] = {
            'category': category,
            'brand': brand,
            'code_family': code_family(code),
        }
    return out


def semantic_id(product_id, meta, scheme):
    row = meta.get(str(product_id), {})
    category = row.get('category', '')
    brand = row.get('brand', '')
    code = row.get('code_family', '')
    if scheme == 'category':
        return f'cat={category}' if category else ''
    if scheme == 'brand':
        return f'brand={brand}' if brand else ''
    if scheme == 'code_family':
        return f'code={code}' if code else ''
    if scheme == 'category_brand':
        return f'cat={category}|brand={brand}' if category and brand else ''
    if scheme == 'code_brand':
        return f'code={code}|brand={brand}' if code and brand else ''
    raise ValueError(f'Unknown semantic scheme: {scheme}')


def rank_key(product_id, pop_purchase, pop_implicit, pop_pre, pop_all):
    """Leakage-safe intra-semantic-ID micro-ranker.

    Ranking is constrained to items inside the decoded semantic bucket. It uses
    validation purchase popularity first, then validation implicit/all-event
    popularity, and finally pre-period popularity. This is not a learned bridge
    score; it is the no-training audit's bucket-internal truncation rule.
    """
    product_id = str(product_id)
    return (
        -pop_purchase.get(product_id, 0),
        -pop_implicit.get(product_id, 0),
        -pop_all.get(product_id, 0),
        -pop_pre.get(product_id, 0),
        product_id,
    )


def build_bucket_ranks(active_products, meta, schemes, pop_purchase, pop_implicit, pop_pre, pop_all, bucket_topk):
    bucket_ranks = {}
    bucket_sizes = defaultdict(dict)
    active_products = [str(p) for p in active_products]
    for scheme in schemes:
        buckets = defaultdict(list)
        for product_id in active_products:
            sid = semantic_id(product_id, meta, scheme)
            if sid:
                buckets[sid].append(product_id)
        ranked = {}
        for sid, products in buckets.items():
            products = sorted(products, key=lambda p: rank_key(p, pop_purchase, pop_implicit, pop_pre, pop_all))
            bucket_sizes[scheme][sid] = len(products)
            ranked[sid] = products[:bucket_topk] if bucket_topk else products
        bucket_ranks[scheme] = ranked
    return bucket_ranks, bucket_sizes


def event_products(events):
    return [str(e['product_id']) for e in events if e.get('product_id')]


def event_semantic_ids(events, meta, scheme):
    out = []
    for product_id in event_products(events):
        sid = semantic_id(product_id, meta, scheme)
        if sid:
            out.append(sid)
    return out


def first_hit_rank(candidates, targets):
    target_set = set(str(t) for t in targets)
    for idx, product_id in enumerate(candidates):
        if product_id in target_set:
            return idx
    return -1


def dedup_preserve_order(items, excluded):
    seen = set()
    out = []
    for item in items:
        item = str(item)
        if item in excluded or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def candidate_list_for_user(record, scheme, scope, meta, bucket_ranks, source_topk):
    history = record['history']
    context = record['val_feedback']
    if scope == 'history':
        sid_events = history
    elif scope == 'context':
        sid_events = context
    elif scope == 'history_context':
        sid_events = history + context
    else:
        raise ValueError(f'Unknown source scope: {scope}')
    semids = Counter(event_semantic_ids(sid_events, meta, scheme))
    ranked_semids = [sid for sid, _ in semids.most_common()]
    seen_products = set(event_products(history + context))
    raw = []
    for sid in ranked_semids:
        products = bucket_ranks[scheme].get(sid, [])
        raw.extend(products[:source_topk] if source_topk else products)
    return dedup_preserve_order(raw, seen_products)


def summarize_ranks(ranks, ks):
    ranks = [int(r) for r in ranks]
    out = OrderedDict()
    out['n_users'] = len(ranks)
    out['hit_counts'] = OrderedDict()
    out['hit_rates'] = OrderedDict()
    for k in ks:
        hits = sum(1 for r in ranks if 0 <= r < k)
        out['hit_counts'][f'hit@{k}'] = int(hits)
        out['hit_rates'][f'hit@{k}'] = float(hits / max(len(ranks), 1))
    found = [r for r in ranks if r >= 0]
    out['mean_first_hit_rank'] = float(sum(found) / len(found)) if found else None
    return out


def parse_existing_cols(text):
    if text.strip():
        return [c.strip() if c.strip().endswith('_rank') else f'{c.strip()}_rank'
                for c in text.split(',') if c.strip()]
    return list(DEFAULT_EXISTING_COLS)


def load_existing_hits(path, cols, ks):
    if not path:
        return None, []
    df = pd.read_csv(path)
    available = [c for c in cols if c in df.columns]
    if not available:
        raise ValueError(f'No requested existing rank columns found in {path}')
    hits = {}
    for _, row in df.iterrows():
        user_id = str(row['user_id'])
        ranks = [int(row[c]) for c in available]
        hits[user_id] = {k: any(0 <= r < k for r in ranks) for k in ks}
    return hits, available


def write_markdown(path, args, summary):
    lines = [
        '# REES46 Semantic-ID Bridge Audit',
        '',
        f"- Artifact: `{args.artifact}`",
        f"- Phase: `{args.phase}`",
        f"- Existing per-user file: `{args.existing_per_user or ''}`",
        f"- Semantic schemes: `{', '.join(summary['schemes'])}`",
        f"- Source scopes: `{', '.join(summary['source_scopes'])}`",
        f"- Source topK per semantic bucket: `{args.source_topk}`",
        f"- Intra-ID ranking: `{summary['intra_id_ranking']}`",
        '',
        '## Support Gate Summary',
        '',
        '| Group | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Delta@100 vs existing | Delta@500 vs existing |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, group in summary['support_groups'].items():
        hits = group['hit_counts']
        lines.append(
            f"| `{name}` | {hits.get('hit@10', 0)} | {hits.get('hit@50', 0)} | "
            f"{hits.get('hit@100', 0)} | {hits.get('hit@500', 0)} | "
            f"{group.get('delta_vs_existing@100', 0)} | {group.get('delta_vs_existing@500', 0)} |"
        )
    lines.extend([
        '',
        '## Best Semantic Bridge Sources',
        '',
        '| Source | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Existing miss -> hit@100 | Existing miss -> hit@500 | Mean candidates |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in summary['source_table'][:20]:
        lines.append(
            f"| `{row['source']}` | {row['hit@10']} | {row['hit@50']} | "
            f"{row['hit@100']} | {row['hit@500']} | "
            f"{row.get('existing_miss_to_hit@100', 0)} | "
            f"{row.get('existing_miss_to_hit@500', 0)} | "
            f"{row['mean_candidate_count']:.1f} |"
        )
    lines.extend([
        '',
        '## Decision',
        '',
        summary['decision'],
        '',
        '## Interpretation',
        '',
        summary['interpretation'],
    ])
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='No-training Semantic-ID bridge support audit for REES46.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--existing-per-user', default='')
    parser.add_argument('--existing-cols', default='')
    parser.add_argument('--schemes', default=','.join(DEFAULT_SCHEMES))
    parser.add_argument('--source-scopes', default=','.join(DEFAULT_SOURCE_SCOPES))
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--active-catalog-window', default='test', choices=['val', 'test', 'val_test'])
    parser.add_argument('--phase', default='test', choices=['val', 'test'])
    parser.add_argument('--val-context-frac', type=float, default=0.5)
    parser.add_argument('--val-target-types', default='', help='Comma-separated validation holdout event types, e.g. cart.')
    parser.add_argument('--bucket-topk', type=int, default=2000)
    parser.add_argument('--source-topk', type=int, default=500)
    parser.add_argument('--max-users', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    schemes = [x.strip() for x in args.schemes.split(',') if x.strip()]
    scopes = [x.strip() for x in args.source_scopes.split(',') if x.strip()]
    artifact = load_artifact(args.artifact)
    meta = normalize_meta(artifact['product_meta'])

    catalogs = artifact['catalogs']['products']
    if args.active_catalog_window == 'val_test':
        active_products = sorted(set(catalogs['val']) | set(catalogs['test']))
    else:
        active_products = catalogs[args.active_catalog_window]

    if args.phase == 'val':
        pop_purchase = counter_from_top(artifact['purchase_popularity_top'].get('pre_purchase', []))
        pop_implicit = counter_from_top(artifact['popularity_top'].get('pre_all', []))
        pop_pre = counter_from_top(artifact['popularity_top'].get('pre_all', []))
        pop_all = counter_from_top(artifact['popularity_top'].get('pre_all', []))
    else:
        pop_purchase = counter_from_top(artifact['purchase_popularity_top'].get('val_purchase', []))
        pop_implicit = counter_from_top(artifact['popularity_top'].get('val_implicit', []))
        pop_pre = counter_from_top(artifact['popularity_top'].get('pre_all', []))
        pop_all = counter_from_top(artifact['popularity_top'].get('val_all', []))
    bucket_ranks, bucket_sizes = build_bucket_ranks(
        active_products, meta, schemes, pop_purchase, pop_implicit, pop_pre, pop_all, args.bucket_topk
    )

    existing_cols = parse_existing_cols(args.existing_cols)
    existing_hits, existing_available = load_existing_hits(args.existing_per_user, existing_cols, ks)

    users = artifact['users']
    if args.max_users:
        users = users[:args.max_users]

    source_names = [f'semid_{scheme}_{scope}' for scheme in schemes for scope in scopes]
    ranks_by_source = {name: [] for name in source_names}
    counts_by_source = {name: [] for name in source_names}
    per_user_rows = []

    val_target_types = {x.strip() for x in args.val_target_types.split(',') if x.strip()}
    for row_id, record in enumerate(users):
        user_id = str(record['user_id'])
        if args.phase == 'val':
            history = record['history']
            context, target_events = split_val_feedback(record['val_feedback'], args.val_context_frac)
            if val_target_types:
                target_events = [x for x in target_events if str(x.get('type') or '') in val_target_types]
            if not target_events:
                continue
            semantic_record = dict(record)
            semantic_record['history'] = history
            semantic_record['val_feedback'] = context
            targets = [str(x['product_id']) for x in target_events]
        else:
            semantic_record = record
            targets = [str(x['product_id']) for x in record['test_purchases']]
        out = OrderedDict([
            ('row_id', row_id),
            ('user_id', user_id),
            ('n_history', len(semantic_record['history'])),
            ('n_context', len(semantic_record['val_feedback'])),
            ('n_targets', len(targets)),
            ('phase', args.phase),
        ])
        for scheme in schemes:
            for scope in scopes:
                source = f'semid_{scheme}_{scope}'
                candidates = candidate_list_for_user(semantic_record, scheme, scope, meta, bucket_ranks, args.source_topk)
                rank = first_hit_rank(candidates, targets)
                ranks_by_source[source].append(rank)
                counts_by_source[source].append(len(candidates))
                out[f'{source}_rank'] = rank
                out[f'{source}_candidate_count'] = len(candidates)
        per_user_rows.append(out)

    source_table = []
    for source in source_names:
        summary = summarize_ranks(ranks_by_source[source], ks)
        row = OrderedDict([('source', source)])
        for k in ks:
            hit_key = f'hit@{k}'
            row[hit_key] = summary['hit_counts'][hit_key]
            if existing_hits is not None:
                miss_to_hit = 0
                hit_to_miss = 0
                for record, rank in zip(per_user_rows, ranks_by_source[source]):
                    user_id = str(record['user_id'])
                    existing_hit = existing_hits.get(user_id, {}).get(k, False)
                    source_hit = 0 <= rank < k
                    miss_to_hit += int((not existing_hit) and source_hit)
                    hit_to_miss += int(existing_hit and (not source_hit))
                row[f'existing_miss_to_hit@{k}'] = int(miss_to_hit)
                row[f'existing_hit_to_miss@{k}'] = int(hit_to_miss)
        row['mean_candidate_count'] = float(sum(counts_by_source[source]) / max(len(counts_by_source[source]), 1))
        source_table.append(row)
    source_table.sort(
        key=lambda r: (
            r.get('existing_miss_to_hit@100', 0),
            r.get('existing_miss_to_hit@500', 0),
            r.get('hit@100', 0),
            r.get('hit@500', 0),
        ),
        reverse=True,
    )

    n_users = len(per_user_rows)
    support_groups = OrderedDict()
    if existing_hits is not None:
        for group_name in ['existing_union', 'semantic_all_oracle', 'existing_plus_semantic_all']:
            support_groups[group_name] = OrderedDict([
                ('hit_counts', OrderedDict()),
                ('hit_rates', OrderedDict()),
            ])
        for k in ks:
            existing_count = 0
            semantic_count = 0
            union_count = 0
            for idx, record in enumerate(per_user_rows):
                user_id = str(record['user_id'])
                existing_hit = existing_hits.get(user_id, {}).get(k, False)
                semantic_hit = any(0 <= ranks_by_source[source][idx] < k for source in source_names)
                existing_count += int(existing_hit)
                semantic_count += int(semantic_hit)
                union_count += int(existing_hit or semantic_hit)
            support_groups['existing_union']['hit_counts'][f'hit@{k}'] = int(existing_count)
            support_groups['semantic_all_oracle']['hit_counts'][f'hit@{k}'] = int(semantic_count)
            support_groups['existing_plus_semantic_all']['hit_counts'][f'hit@{k}'] = int(union_count)
            for group in support_groups.values():
                group['hit_rates'][f'hit@{k}'] = float(group['hit_counts'][f'hit@{k}'] / max(n_users, 1))
            support_groups['existing_union'][f'delta_vs_existing@{k}'] = 0
            support_groups['semantic_all_oracle'][f'delta_vs_existing@{k}'] = int(semantic_count - existing_count)
            support_groups['existing_plus_semantic_all'][f'delta_vs_existing@{k}'] = int(union_count - existing_count)
    else:
        semantic_oracle_ranks = []
        for idx in range(n_users):
            found = [ranks_by_source[source][idx] for source in source_names if ranks_by_source[source][idx] >= 0]
            semantic_oracle_ranks.append(min(found) if found else -1)
        support_groups['semantic_all_oracle'] = summarize_ranks(semantic_oracle_ranks, ks)

    best = source_table[0] if source_table else {}
    threshold_100 = 0.01 * n_users
    threshold_500 = 0.02 * n_users
    gain100 = best.get('existing_miss_to_hit@100', 0)
    gain500 = best.get('existing_miss_to_hit@500', 0)
    passes = gain100 >= threshold_100 or gain500 >= threshold_500
    decision = (
        f"PASS support gate on `{best.get('source')}`: existing-miss gains "
        f"@100={gain100}, @500={gain500}."
        if passes else
        f"FAIL support gate: best `{best.get('source')}` has existing-miss gains "
        f"@100={gain100}, @500={gain500}, below thresholds "
        f"@100>={threshold_100:.1f} or @500>={threshold_500:.1f}."
    )
    interpretation = (
        'This is a no-training support audit only. A pass means the Semantic-ID bridge changes candidate '
        'reachability enough to justify a calibrated bridge model. A fail means the tested semantic IDs mostly '
        'reshuffle support already covered by popularity/replay/purchase-event sources.'
    )

    summary = OrderedDict([
        ('args', vars(args)),
        ('n_users', n_users),
        ('schemes', schemes),
        ('source_scopes', scopes),
        ('intra_id_ranking', 'val_purchase_popularity -> val_implicit_popularity -> val_all_popularity -> pre_all_popularity within semantic bucket'),
        ('existing_columns_used', existing_available),
        ('support_groups', support_groups),
        ('thresholds', OrderedDict([
            ('existing_miss_to_hit@100', threshold_100),
            ('existing_miss_to_hit@500', threshold_500),
        ])),
        ('decision', decision),
        ('interpretation', interpretation),
        ('source_table', source_table),
        ('bucket_stats', OrderedDict(
            (scheme, OrderedDict([
                ('n_buckets', len(bucket_sizes[scheme])),
                ('mean_bucket_size', float(sum(bucket_sizes[scheme].values()) / max(len(bucket_sizes[scheme]), 1))),
                ('max_bucket_size', int(max(bucket_sizes[scheme].values()) if bucket_sizes[scheme] else 0)),
            ]))
            for scheme in schemes
        )),
    ])

    per_user_path = os.path.join(args.output_dir, 'rees46_semantic_bridge_per_user.csv')
    summary_json = os.path.join(args.output_dir, 'rees46_semantic_bridge_summary.json')
    summary_md = os.path.join(args.output_dir, 'rees46_semantic_bridge_summary.md')
    pd.DataFrame(per_user_rows).to_csv(per_user_path, index=False)
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    write_markdown(summary_md, args, summary)
    print(decision)
    print(f'Wrote {summary_md}')


if __name__ == '__main__':
    main()
