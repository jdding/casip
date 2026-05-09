# REES46 No-Training Baselines

- Artifact: `results/20260505_rees46_protocol_parallel_all_protocol/rees46_protocol_artifact.pkl`
- Users: `16954`
- Active catalog: `test` / `294234` products

## Baselines

| System | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| `brand_overlap_seen_allowed` | 2093 | 3494 | 4133 | 5264 | 3712.2 |
| `brand_replacement_unseen` | 1393 | 2397 | 2894 | 3747 | 3698.9 |
| `category_code_family_replacement_unseen` | 1360 | 2572 | 3239 | 4894 | 73448.6 |
| `category_or_brand_replacement_unseen` | 1484 | 2710 | 3360 | 4782 | 8858.0 |
| `category_overlap_seen_allowed` | 1549 | 2786 | 3338 | 4380 | 5880.5 |
| `category_replacement_unseen` | 1087 | 2045 | 2491 | 3373 | 5867.5 |
| `history_product_replay` | 632 | 684 | 685 | 685 | 9.0 |
| `seen_product_replay` | 2062 | 2473 | 2500 | 2506 | 17.7 |
| `test_purchase_popularity_oracle_upper` | 2607 | 4876 | 6045 | 9512 | 294234.0 |
| `val_feedback_replay` | 1868 | 2097 | 2108 | 2112 | 9.0 |
| `val_popularity` | 2394 | 4320 | 5366 | 7936 | 294234.0 |

## Oracle / Complementarity Probe

The `test_purchase_popularity_oracle_upper` row is intentionally leaky and must not be used as a deployable baseline.
It is included only to expose upper-bound replacement signal under the test purchase distribution.

| Pair | Union Hit@10 | Left-only | Right-only | Jaccard |
|---|---:|---:|---:|---:|
| `brand_overlap_seen_allowed__vs__brand_replacement_unseen` | 2171 | 778 | 78 | 0.6057 |
| `brand_overlap_seen_allowed__vs__category_code_family_replacement_unseen` | 2656 | 1296 | 563 | 0.3001 |
| `brand_overlap_seen_allowed__vs__category_or_brand_replacement_unseen` | 2425 | 941 | 332 | 0.4751 |
| `brand_overlap_seen_allowed__vs__category_overlap_seen_allowed` | 2584 | 1035 | 491 | 0.4094 |
| `brand_overlap_seen_allowed__vs__category_replacement_unseen` | 2535 | 1448 | 442 | 0.2544 |
| `brand_overlap_seen_allowed__vs__history_product_replay` | 2388 | 1756 | 295 | 0.1411 |
| `brand_overlap_seen_allowed__vs__seen_product_replay` | 3219 | 1157 | 1126 | 0.2908 |
| `brand_overlap_seen_allowed__vs__test_purchase_popularity_oracle_upper` | 3204 | 597 | 1111 | 0.4669 |
| `brand_overlap_seen_allowed__vs__val_feedback_replay` | 3183 | 1315 | 1090 | 0.2444 |
| `brand_overlap_seen_allowed__vs__val_popularity` | 2982 | 588 | 889 | 0.5047 |
| `brand_replacement_unseen__vs__category_code_family_replacement_unseen` | 1968 | 608 | 575 | 0.3989 |
| `brand_replacement_unseen__vs__category_or_brand_replacement_unseen` | 1710 | 226 | 317 | 0.6825 |
| `brand_replacement_unseen__vs__category_overlap_seen_allowed` | 2331 | 782 | 938 | 0.2621 |
| `brand_replacement_unseen__vs__category_replacement_unseen` | 1841 | 754 | 448 | 0.3471 |
| `brand_replacement_unseen__vs__history_product_replay` | 1965 | 1333 | 572 | 0.0305 |
| `brand_replacement_unseen__vs__seen_product_replay` | 3295 | 1233 | 1902 | 0.0486 |
| `brand_replacement_unseen__vs__test_purchase_popularity_oracle_upper` | 3043 | 436 | 1650 | 0.3145 |
| `brand_replacement_unseen__vs__val_feedback_replay` | 3121 | 1253 | 1728 | 0.0449 |
| `brand_replacement_unseen__vs__val_popularity` | 2872 | 478 | 1479 | 0.3186 |
| `category_code_family_replacement_unseen__vs__category_or_brand_replacement_unseen` | 1769 | 285 | 409 | 0.6077 |
| `category_code_family_replacement_unseen__vs__category_overlap_seen_allowed` | 2003 | 454 | 643 | 0.4523 |
| `category_code_family_replacement_unseen__vs__category_replacement_unseen` | 1515 | 428 | 155 | 0.6152 |
| `category_code_family_replacement_unseen__vs__history_product_replay` | 1937 | 1305 | 577 | 0.0284 |
| `category_code_family_replacement_unseen__vs__seen_product_replay` | 3282 | 1220 | 1922 | 0.0427 |
| `category_code_family_replacement_unseen__vs__test_purchase_popularity_oracle_upper` | 2804 | 197 | 1444 | 0.4148 |
| `category_code_family_replacement_unseen__vs__val_feedback_replay` | 3110 | 1242 | 1750 | 0.0379 |
| `category_code_family_replacement_unseen__vs__val_popularity` | 2540 | 146 | 1180 | 0.4780 |
| `category_or_brand_replacement_unseen__vs__category_overlap_seen_allowed` | 2119 | 570 | 635 | 0.4313 |
| `category_or_brand_replacement_unseen__vs__category_replacement_unseen` | 1641 | 554 | 157 | 0.5667 |
| `category_or_brand_replacement_unseen__vs__history_product_replay` | 2052 | 1420 | 568 | 0.0312 |
| `category_or_brand_replacement_unseen__vs__seen_product_replay` | 3376 | 1314 | 1892 | 0.0504 |
| `category_or_brand_replacement_unseen__vs__test_purchase_popularity_oracle_upper` | 2916 | 309 | 1432 | 0.4029 |
| `category_or_brand_replacement_unseen__vs__val_feedback_replay` | 3210 | 1342 | 1726 | 0.0442 |
| `category_or_brand_replacement_unseen__vs__val_popularity` | 2684 | 290 | 1200 | 0.4449 |
| `category_overlap_seen_allowed__vs__category_replacement_unseen` | 1617 | 530 | 68 | 0.6302 |
| `category_overlap_seen_allowed__vs__history_product_replay` | 1858 | 1226 | 309 | 0.1738 |
| `category_overlap_seen_allowed__vs__seen_product_replay` | 2949 | 887 | 1400 | 0.2245 |
| `category_overlap_seen_allowed__vs__test_purchase_popularity_oracle_upper` | 2960 | 353 | 1411 | 0.4041 |
| `category_overlap_seen_allowed__vs__val_feedback_replay` | 2904 | 1036 | 1355 | 0.1767 |
| `category_overlap_seen_allowed__vs__val_popularity` | 2681 | 287 | 1132 | 0.4707 |
| `category_replacement_unseen__vs__history_product_replay` | 1658 | 1026 | 571 | 0.0368 |
| `category_replacement_unseen__vs__seen_product_replay` | 3008 | 946 | 1921 | 0.0469 |
| `category_replacement_unseen__vs__test_purchase_popularity_oracle_upper` | 2863 | 256 | 1776 | 0.2903 |
| `category_replacement_unseen__vs__val_feedback_replay` | 2841 | 973 | 1754 | 0.0401 |
| `category_replacement_unseen__vs__val_popularity` | 2640 | 246 | 1553 | 0.3186 |
| `history_product_replay__vs__seen_product_replay` | 2108 | 46 | 1476 | 0.2780 |
| `history_product_replay__vs__test_purchase_popularity_oracle_upper` | 2965 | 358 | 2333 | 0.0924 |
| `history_product_replay__vs__val_feedback_replay` | 2237 | 369 | 1605 | 0.1176 |
| `history_product_replay__vs__val_popularity` | 2746 | 352 | 2114 | 0.1020 |
| `seen_product_replay__vs__test_purchase_popularity_oracle_upper` | 3895 | 1288 | 1833 | 0.1987 |
| `seen_product_replay__vs__val_feedback_replay` | 2199 | 331 | 137 | 0.7872 |
| `seen_product_replay__vs__val_popularity` | 3680 | 1286 | 1618 | 0.2109 |
| `test_purchase_popularity_oracle_upper__vs__val_feedback_replay` | 3820 | 1952 | 1213 | 0.1715 |
| `test_purchase_popularity_oracle_upper__vs__val_popularity` | 2791 | 397 | 184 | 0.7918 |
| `val_feedback_replay__vs__val_popularity` | 3614 | 1220 | 1746 | 0.1793 |

Union over all no-training systems: Hit@10 `4639`, Hit@50 `7025`, Hit@100 `8092`, Hit@500 `11155`.
