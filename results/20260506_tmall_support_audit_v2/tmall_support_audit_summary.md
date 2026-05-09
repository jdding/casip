# Tmall No-Training Support Audit

- Gate: `pre_any_gap_silent_val_proxy_test_purchase`
- Gate users: `45574`
- Evaluated users with test purchases: `45574`

## Source Hits

| Source | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| existing_user_replay | 21220 | 24853 | 25010 | 25034 |
| global_deployable_popularity | 248 | 1290 | 2169 | 7172 |
| semantic_category_val_proxy | 3660 | 8601 | 11468 | 18043 |
| semantic_brand_val_proxy | 9288 | 16834 | 19846 | 24505 |
| semantic_category_brand_val_proxy | 13179 | 21076 | 23301 | 25843 |
| semantic_merchant_val_proxy | 9960 | 17528 | 20468 | 24788 |
| semantic_union_oracle | 15282 | 22848 | 25234 | 28236 |
| existing_plus_semantic_oracle | 23775 | 27087 | 27734 | 28853 |
| existing_plus_semantic_all | 19289 | 23460 | 24826 | 28159 |

## Existing Miss Recovery

| Source@K | Recovered existing misses |
|---|---:|
| semantic_brand_val_proxy@10 | 1258 |
| semantic_brand_val_proxy@50 | 1120 |
| semantic_brand_val_proxy@100 | 1374 |
| semantic_brand_val_proxy@500 | 1958 |
| semantic_category_brand_val_proxy@10 | 1842 |
| semantic_category_brand_val_proxy@50 | 1173 |
| semantic_category_brand_val_proxy@100 | 1325 |
| semantic_category_brand_val_proxy@500 | 1660 |
| semantic_category_val_proxy@10 | 643 |
| semantic_category_val_proxy@50 | 1024 |
| semantic_category_val_proxy@100 | 1464 |
| semantic_category_val_proxy@500 | 2505 |
| semantic_merchant_val_proxy@10 | 1326 |
| semantic_merchant_val_proxy@50 | 1079 |
| semantic_merchant_val_proxy@100 | 1299 |
| semantic_merchant_val_proxy@500 | 1760 |