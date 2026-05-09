# Synerise No-Training Support Audit

- Gate: `pre_any_gap_silent_val_proxy_test_buy`
- Gate users: `17465`
- Evaluated users with test purchases: `17465`
- Decision: PASS: bridge sources expand no-training support beyond existing replay.

## Source Hits

| Source | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| existing_user_replay | 1507 | 1629 | 1630 | 1631 |
| global_deployable_popularity | 155 | 383 | 571 | 1388 |
| bridge_category | 563 | 1089 | 1332 | 1864 |
| bridge_price | 208 | 451 | 600 | 1012 |
| bridge_category_price | 1127 | 1565 | 1696 | 1826 |
| bridge_name_prefix | 1334 | 1657 | 1736 | 1857 |
| bridge_union_oracle | 1629 | 2080 | 2263 | 2555 |
| existing_plus_bridge_oracle | 1826 | 2132 | 2284 | 2556 |
| existing_plus_bridge_all | 1560 | 1920 | 2038 | 2374 |

## Existing Miss Recovery

| Source@K | Recovered existing misses |
|---|---:|
| bridge_category@10 | 127 |
| bridge_category@50 | 280 |
| bridge_category@100 | 384 |
| bridge_category@500 | 609 |
| bridge_category_price@10 | 104 |
| bridge_category_price@50 | 148 |
| bridge_category_price@100 | 178 |
| bridge_category_price@500 | 214 |
| bridge_name_prefix@10 | 135 |
| bridge_name_prefix@50 | 151 |
| bridge_name_prefix@100 | 185 |
| bridge_name_prefix@500 | 250 |
| bridge_price@10 | 52 |
| bridge_price@50 | 104 |
| bridge_price@100 | 148 |
| bridge_price@500 | 268 |

## Interpretation Stub

This is a no-training support audit. `bridge_union_oracle` and `existing_plus_bridge_oracle` are audit-only headroom, not deployable methods.
Proceed to naive insertion / CASP only if support expansion is positive and the bridge source is not just price-bucket popularity.