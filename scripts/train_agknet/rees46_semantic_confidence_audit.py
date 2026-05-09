#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, OrderedDict

import pandas as pd

import rees46_exact_source_protected_list_fusion as exact
import rees46_no_training_baselines as base
import rees46_semantic_bridge_audit as semantic
import rees46_source_aware_promotion_gate as promo
import rees46_validation_safe_gate as gate


def parse_ints(text):
    return [int(x) for x in text.split(',') if x.strip()]


def semantic_events(record, scope):
    if scope == 'history':
        return record['history']
    if scope == 'context':
        return record['val_feedback']
    if scope == 'history_context':
        return record['history'] + record['val_feedback']
    raise ValueError(scope)


def entropy_from_counts(counts):
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return float(-sum(p * math.log(p) for p in probs) / math.log(len(probs)))


def semantic_confidence_features(record, scheme, scope, meta, bucket_ranks):
    semids = Counter(semantic.event_semantic_ids(semantic_events(record, scope), meta, scheme))
    total = sum(semids.values())
    if not semids:
        return OrderedDict([
            ('top_semid_count', 0),
            ('top_semid_share', 0.0),
            ('top_semid_margin', 0.0),
            ('semid_entropy', 0.0),
            ('n_semids', 0),
            ('top_bucket_size', 0),
            ('top_bucket_specificity', 0.0),
        ])
    ranked = semids.most_common()
    top_sid, top_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    bucket_size = len(bucket_ranks[scheme].get(top_sid, []))
    return OrderedDict([
        ('top_semid_count', int(top_count)),
        ('top_semid_share', float(top_count / max(total, 1))),
        ('top_semid_margin', float((top_count - second_count) / max(total, 1))),
        ('semid_entropy', entropy_from_counts(list(semids.values()))),
        ('n_semids', int(len(semids))),
        ('top_bucket_size', int(bucket_size)),
        ('top_bucket_specificity', float(1.0 / math.log1p(bucket_size)) if bucket_size > 1 else 0.0),
    ])


def first_hit_rank(items, targets):
    target_set = set(str(t) for t in targets)
    for idx, item in enumerate(items):
        if item in target_set:
            return idx
    return -1


def evaluate_phase(artifact, args, phase, scheme, scope, existing_weights, promotion_config):
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
    max_k = max(parse_ints(args.ks))
    max_list = max(max_k, args.source_topk, args.existing_pool_k)
    rows = []
    for idx, (raw_record, record) in enumerate(zip(users, records)):
        if idx and idx % 1000 == 0:
            print(f'[{phase}] processed {idx}/{len(records)} users', flush=True)
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
        existing_rank = first_hit_rank(existing_list[:max_k], record['targets'])
        fused_rank = first_hit_rank(fused, record['targets'])
        features = semantic_confidence_features(record['semantic_record'], scheme, scope, state['sem_meta'], bucket_ranks)
        out = OrderedDict([
            ('user_id', record['user_id']),
            ('phase', phase),
            ('existing_rank', int(existing_rank)),
            ('fused_rank', int(fused_rank)),
            ('semantic_candidate_count', int(len(semantic_list))),
        ])
        out.update(features)
        for k in parse_ints(args.ks):
            base_hit = 0 <= existing_rank < k
            fused_hit = 0 <= fused_rank < k
            out[f'gross@{k}'] = int((not base_hit) and fused_hit)
            out[f'cannibal@{k}'] = int(base_hit and (not fused_hit))
            out[f'net@{k}'] = int(fused_hit) - int(base_hit)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_rows(df, ks):
    out = OrderedDict()
    out['n_users'] = int(len(df))
    for k in ks:
        gross = int(df[f'gross@{k}'].sum())
        cann = int(df[f'cannibal@{k}'].sum())
        out[f'gross@{k}'] = gross
        out[f'cannibal@{k}'] = cann
        out[f'net@{k}'] = gross - cann
        out[f'ratio@{k}'] = float(cann / gross) if gross else None
    return out


def bin_feature(df, feature, bins, ks):
    values = df[feature]
    if values.nunique(dropna=True) <= 1:
        return []
    try:
        labels = pd.qcut(values, q=bins, duplicates='drop')
    except ValueError:
        return []
    rows = []
    for label, part in df.groupby(labels, observed=True):
        row = OrderedDict([
            ('feature', feature),
            ('bin', str(label)),
            ('n_users', int(len(part))),
            ('mean_value', float(part[feature].mean())),
        ])
        for k in ks:
            gross = int(part[f'gross@{k}'].sum())
            cann = int(part[f'cannibal@{k}'].sum())
            row[f'gross@{k}'] = gross
            row[f'cannibal@{k}'] = cann
            row[f'net@{k}'] = gross - cann
            row[f'ratio@{k}'] = float(cann / gross) if gross else None
        rows.append(row)
    return rows


def write_markdown(path, args, summary, bin_table):
    lines = [
        '# REES46 Semantic Confidence Audit',
        '',
        f"- Semantic source: `{summary['semantic_source']}`",
        f"- Promotion policy: `{summary['promotion_config']['name']}`",
        '- Note: current bridge has no generative logits/probabilities; these are internal confidence proxies from semantic-ID votes and bucket specificity.',
        '',
        '## Overall',
        '',
        '| Split | Users | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for split in ['validation', 'test']:
        row = summary[split]
        ratio = row.get('ratio@100')
        ratio_text = '' if ratio is None else f'{ratio:.3f}'
        lines.append(
            f"| {split} | {row['n_users']} | {row.get('gross@100', 0)} | "
            f"{row.get('cannibal@100', 0)} | {row.get('net@100', 0)} | {ratio_text} |"
        )
    lines.extend([
        '',
        '## Test Feature Bins',
        '',
        '| Feature | Bin | Users | Mean | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in bin_table:
        if row['phase'] != 'test':
            continue
        ratio = row.get('ratio@100')
        ratio_text = '' if ratio is None else f'{ratio:.3f}'
        lines.append(
            f"| `{row['feature']}` | {row['bin']} | {row['n_users']} | {row['mean_value']:.4f} | "
            f"{row.get('gross@100', 0)} | {row.get('cannibal@100', 0)} | "
            f"{row.get('net@100', 0)} | {ratio_text} |"
        )
    lines.extend(['', '## Interpretation', '', summary['interpretation']])
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Audit semantic-internal confidence proxies against promotion cannibalization.')
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
    parser.add_argument('--bins', type=int, default=5)
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
    promotion_config = OrderedDict([
        ('name', args.promotion_name),
        ('quota', args.promotion_quota),
        ('semantic_budget', args.promotion_semantic_budget),
        ('insert_start', args.promotion_insert_start),
        ('filter', args.promotion_filter),
    ])
    artifact = base.load_artifact(args.artifact)
    val_df = evaluate_phase(artifact, args, 'val', scheme, scope, existing_weights, promotion_config)
    test_df = evaluate_phase(artifact, args, 'test', scheme, scope, existing_weights, promotion_config)
    feature_names = [
        'top_semid_count',
        'top_semid_share',
        'top_semid_margin',
        'semid_entropy',
        'n_semids',
        'top_bucket_size',
        'top_bucket_specificity',
        'semantic_candidate_count',
    ]
    bin_rows = []
    for phase, df in [('validation', val_df), ('test', test_df)]:
        for feature in feature_names:
            for row in bin_feature(df, feature, args.bins, ks):
                row['phase'] = phase
                bin_rows.append(row)
    summary = OrderedDict([
        ('args', vars(args)),
        ('semantic_source', semantic_source_name),
        ('promotion_config', promotion_config),
        ('validation', summarize_rows(val_df, ks)),
        ('test', summarize_rows(test_df, ks)),
        ('interpretation', 'No true top-1 probability exists in the current metadata semantic-ID bridge. Use this audit to decide whether vote/bucket confidence proxies can substitute for semantic self-confidence.'),
    ])
    summary_json = os.path.join(args.output_dir, 'rees46_semantic_confidence_audit_summary.json')
    summary_md = os.path.join(args.output_dir, 'rees46_semantic_confidence_audit_summary.md')
    per_user_path = os.path.join(args.output_dir, 'rees46_semantic_confidence_audit_per_user.csv')
    bin_path = os.path.join(args.output_dir, 'rees46_semantic_confidence_audit_bins.csv')
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    pd.concat([val_df, test_df], ignore_index=True).to_csv(per_user_path, index=False)
    pd.DataFrame(bin_rows).to_csv(bin_path, index=False)
    write_markdown(summary_md, args, summary, bin_rows)
    print(json.dumps({'summary': summary_json, 'markdown': summary_md, 'bins': bin_path, 'per_user': per_user_path}, indent=2))


if __name__ == '__main__':
    main()
