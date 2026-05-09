#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from collections import Counter, OrderedDict, defaultdict

import pandas as pd


DEFAULT_KS = [10, 50, 100, 500]


def load_artifact(path):
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    if artifact.get('artifact_version') != 'rees46_protocol_v1':
        raise RuntimeError(f'Unsupported artifact version: {artifact.get("artifact_version")}')
    return artifact


def counter_from_top(rows):
    counter = Counter()
    for product_id, count in rows:
        counter[str(product_id)] = int(count)
    return counter


def product_meta_maps(product_meta):
    category = {}
    brand = {}
    category_code = {}
    for product_id, meta in product_meta.items():
        pid = str(product_id)
        category[pid] = str(meta.get('category_id') or '')
        brand[pid] = str(meta.get('brand') or '')
        category_code[pid] = str(meta.get('category_code') or '')
    return category, brand, category_code


def rank_products(products, score_counter, fallback_counter=None):
    fallback_counter = fallback_counter or Counter()
    return sorted(
        (str(p) for p in products),
        key=lambda p: (-score_counter.get(p, 0), -fallback_counter.get(p, 0), p),
    )


def build_ranked_index(active_products, meta, pop_val_implicit, pop_val_all, pop_pre_all, pop_test_purchase):
    category_by_product, brand_by_product, code_by_product = meta
    active_products = [str(p) for p in active_products]
    val_popularity = rank_products(active_products, pop_val_implicit, pop_pre_all)
    purchase_oracle = rank_products(active_products, pop_test_purchase, pop_val_all)
    rank_order = {p: i for i, p in enumerate(val_popularity)}

    category_products = defaultdict(list)
    brand_products = defaultdict(list)
    code_family_products = defaultdict(list)
    for product_id in active_products:
        category = category_by_product.get(product_id, '')
        brand = brand_by_product.get(product_id, '')
        code = code_by_product.get(product_id, '')
        if category:
            category_products[category].append(product_id)
        if brand:
            brand_products[brand].append(product_id)
        if code:
            code_family_products[code.split('.')[0]].append(product_id)

    def sort_by_global_rank(mapping):
        return {
            key: sorted(values, key=lambda p: rank_order.get(p, len(rank_order)))
            for key, values in mapping.items()
        }

    return OrderedDict([
        ('active_set', set(active_products)),
        ('val_popularity', val_popularity),
        ('purchase_oracle', purchase_oracle),
        ('rank_order', rank_order),
        ('category_products', sort_by_global_rank(category_products)),
        ('brand_products', sort_by_global_rank(brand_products)),
        ('code_family_products', sort_by_global_rank(code_family_products)),
    ])


def user_profile(record):
    history = record['history']
    val_feedback = record['val_feedback']
    seen_products = [x['product_id'] for x in history + val_feedback]
    history_products = [x['product_id'] for x in history]
    feedback_products = [x['product_id'] for x in val_feedback]
    categories = {str(x.get('category_id') or '') for x in history + val_feedback if x.get('category_id')}
    brands = {str(x.get('brand') or '') for x in history + val_feedback if x.get('brand')}
    category_codes = {str(x.get('category_code') or '') for x in history + val_feedback if x.get('category_code')}
    test_products = [x['product_id'] for x in record['test_purchases']]
    return OrderedDict([
        ('seen_products', set(seen_products)),
        ('history_products', set(history_products)),
        ('feedback_products', set(feedback_products)),
        ('categories', categories),
        ('brands', brands),
        ('category_codes', category_codes),
        ('test_products', test_products),
    ])


def first_hit_rank(recommendations, targets):
    target_set = set(str(x) for x in targets)
    for rank, product_id in enumerate(recommendations):
        if product_id in target_set:
            return rank
    return -1


def merge_ranked_lists(lists, rank_order, seen_products=None):
    seen_products = seen_products or set()
    merged = set()
    for values in lists:
        for product_id in values:
            if product_id not in seen_products:
                merged.add(product_id)
    return sorted(merged, key=lambda p: rank_order.get(p, len(rank_order)))


def evaluate_rows(rows, ks):
    n = len(rows)
    summary = OrderedDict([
        ('n_users', n),
        ('hit_counts', OrderedDict()),
        ('hit_rates', OrderedDict()),
        ('mean_first_hit_rank', None),
    ])
    ranks = [r['first_hit_rank'] for r in rows if r['first_hit_rank'] >= 0]
    for k in ks:
        hits = sum(1 for r in rows if 0 <= r['first_hit_rank'] < k)
        summary['hit_counts'][f'hit@{k}'] = int(hits)
        summary['hit_rates'][f'hit@{k}'] = float(hits / max(n, 1))
    if ranks:
        summary['mean_first_hit_rank'] = float(sum(ranks) / len(ranks))
    return summary


def make_baseline_lists(profile, active_products, indexed, pop_val_implicit, pop_pre_all):
    active_set = indexed['active_set']
    seen_active = profile['seen_products'] & active_set
    history_active = profile['history_products'] & active_set
    feedback_active = profile['feedback_products'] & active_set

    category_lists = [indexed['category_products'].get(c, []) for c in profile['categories']]
    brand_lists = [indexed['brand_products'].get(b, []) for b in profile['brands']]
    code_prefixes = {code.split('.')[0] for code in profile['category_codes'] if code}
    code_lists = [indexed['code_family_products'].get(c, []) for c in code_prefixes]
    seen_products = profile['seen_products']
    rank_order = indexed['rank_order']

    return OrderedDict([
        ('val_popularity', indexed['val_popularity']),
        ('test_purchase_popularity_oracle_upper', indexed['purchase_oracle']),
        ('history_product_replay', rank_products(history_active, pop_val_implicit, pop_pre_all)),
        ('val_feedback_replay', rank_products(feedback_active, pop_val_implicit, pop_pre_all)),
        ('seen_product_replay', rank_products(seen_active, pop_val_implicit, pop_pre_all)),
        ('category_overlap_seen_allowed', merge_ranked_lists(category_lists, rank_order)),
        ('category_replacement_unseen', merge_ranked_lists(category_lists, rank_order, seen_products)),
        ('brand_overlap_seen_allowed', merge_ranked_lists(brand_lists, rank_order)),
        ('brand_replacement_unseen', merge_ranked_lists(brand_lists, rank_order, seen_products)),
        ('category_or_brand_replacement_unseen', merge_ranked_lists(category_lists + brand_lists, rank_order, seen_products)),
        ('category_code_family_replacement_unseen', merge_ranked_lists(code_lists, rank_order, seen_products)),
    ])


def summarize_overlap(per_user_rows, system_names, ks):
    out = OrderedDict()
    for left in system_names:
        for right in system_names:
            if left >= right:
                continue
            left_hit = [row[f'{left}_rank'] >= 0 and row[f'{left}_rank'] < 10 for row in per_user_rows]
            right_hit = [row[f'{right}_rank'] >= 0 and row[f'{right}_rank'] < 10 for row in per_user_rows]
            both = sum(1 for a, b in zip(left_hit, right_hit) if a and b)
            union = sum(1 for a, b in zip(left_hit, right_hit) if a or b)
            out[f'{left}__vs__{right}'] = OrderedDict([
                ('both_hit10', int(both)),
                ('union_hit10', int(union)),
                ('left_only_hit10', int(sum(1 for a, b in zip(left_hit, right_hit) if a and not b))),
                ('right_only_hit10', int(sum(1 for a, b in zip(left_hit, right_hit) if b and not a))),
                ('jaccard_hit10', float(both / max(union, 1))),
            ])
    union_all = {}
    for k in ks:
        hits = 0
        for row in per_user_rows:
            if any(0 <= row[f'{name}_rank'] < k for name in system_names):
                hits += 1
        union_all[f'hit@{k}'] = int(hits)
    out['union_all_systems'] = union_all
    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate no-training REES46 protocol baselines.')
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ks', default='10,50,100,500')
    parser.add_argument('--active-catalog-window', default='test', choices=['val', 'test', 'val_test'])
    parser.add_argument('--max-users', type=int, default=0, help='Debug only; 0 evaluates all users.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    artifact = load_artifact(args.artifact)
    catalogs = artifact['catalogs']['products']
    if args.active_catalog_window == 'val_test':
        active_products = sorted(set(catalogs['val']) | set(catalogs['test']))
    else:
        active_products = catalogs[args.active_catalog_window]

    pop_val_implicit = counter_from_top(artifact['popularity_top'].get('val_implicit', []))
    pop_val_all = counter_from_top(artifact['popularity_top'].get('val_all', []))
    pop_pre_all = counter_from_top(artifact['popularity_top'].get('pre_all', []))
    pop_test_purchase = counter_from_top(artifact['purchase_popularity_top'].get('test_purchase', []))
    meta = product_meta_maps(artifact['product_meta'])
    indexed = build_ranked_index(active_products, meta, pop_val_implicit, pop_val_all, pop_pre_all, pop_test_purchase)

    users = artifact['users']
    if args.max_users:
        users = users[: args.max_users]

    by_system = defaultdict(list)
    per_user_rows = []
    candidate_count_rows = []
    for idx, record in enumerate(users):
        profile = user_profile(record)
        baseline_lists = make_baseline_lists(
            profile,
            active_products,
            indexed,
            pop_val_implicit,
            pop_pre_all,
        )
        row = OrderedDict([
            ('row_id', idx),
            ('user_id', record['user_id']),
            ('n_history', len(record['history'])),
            ('n_val_feedback', len(record['val_feedback'])),
            ('n_test_purchases', len(record['test_purchases'])),
        ])
        for name, recs in baseline_lists.items():
            rank = first_hit_rank(recs, profile['test_products'])
            row[f'{name}_rank'] = int(rank)
            row[f'{name}_candidate_count'] = int(len(recs))
            by_system[name].append(OrderedDict([
                ('user_id', record['user_id']),
                ('first_hit_rank', int(rank)),
                ('candidate_count', int(len(recs))),
            ]))
        per_user_rows.append(row)
        candidate_count_rows.append(OrderedDict(
            [('user_id', record['user_id'])] + [
                (f'{name}_candidate_count', int(len(recs)))
                for name, recs in baseline_lists.items()
            ]
        ))

    system_summaries = OrderedDict()
    for name, rows in sorted(by_system.items()):
        system_summaries[name] = evaluate_rows(rows, ks)
        counts = [r['candidate_count'] for r in rows]
        system_summaries[name]['mean_candidate_count'] = float(sum(counts) / max(len(counts), 1))

    system_names = list(system_summaries.keys())
    overlap = summarize_overlap(per_user_rows, system_names, ks)
    summary = OrderedDict([
        ('args', vars(args)),
        ('artifact_args', artifact.get('args', {})),
        ('n_users', len(users)),
        ('active_catalog_window', args.active_catalog_window),
        ('active_catalog_size', len(active_products)),
        ('systems', system_summaries),
        ('overlap_oracle', overlap),
    ])

    per_user_path = os.path.join(args.output_dir, 'rees46_no_training_per_user.csv')
    pd.DataFrame(per_user_rows).to_csv(per_user_path, index=False)
    summary_path = os.path.join(args.output_dir, 'rees46_no_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(args.output_dir, 'rees46_no_training_summary.md')
    lines = [
        '# REES46 No-Training Baselines',
        '',
        f"- Artifact: `{args.artifact}`",
        f"- Users: `{len(users)}`",
        f"- Active catalog: `{args.active_catalog_window}` / `{len(active_products)}` products",
        '',
        '## Baselines',
        '',
        '| System | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Mean candidates |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, row in system_summaries.items():
        lines.append(
            f"| `{name}` | {row['hit_counts'].get('hit@10', 0)} | "
            f"{row['hit_counts'].get('hit@50', 0)} | {row['hit_counts'].get('hit@100', 0)} | "
            f"{row['hit_counts'].get('hit@500', 0)} | {row['mean_candidate_count']:.1f} |"
        )
    lines.extend([
        '',
        '## Oracle / Complementarity Probe',
        '',
        'The `test_purchase_popularity_oracle_upper` row is intentionally leaky and must not be used as a deployable baseline.',
        'It is included only to expose upper-bound replacement signal under the test purchase distribution.',
        '',
        '| Pair | Union Hit@10 | Left-only | Right-only | Jaccard |',
        '|---|---:|---:|---:|---:|',
    ])
    for pair, row in overlap.items():
        if pair == 'union_all_systems':
            continue
        lines.append(
            f"| `{pair}` | {row['union_hit10']} | {row['left_only_hit10']} | "
            f"{row['right_only_hit10']} | {row['jaccard_hit10']:.4f} |"
        )
    union_row = overlap['union_all_systems']
    lines.extend([
        '',
        f"Union over all no-training systems: "
        f"Hit@10 `{union_row.get('hit@10', 0)}`, Hit@50 `{union_row.get('hit@50', 0)}`, "
        f"Hit@100 `{union_row.get('hit@100', 0)}`, Hit@500 `{union_row.get('hit@500', 0)}`.",
    ])
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(json.dumps({'summary': summary_path, 'markdown': md_path, 'per_user': per_user_path}, indent=2))


if __name__ == '__main__':
    main()
