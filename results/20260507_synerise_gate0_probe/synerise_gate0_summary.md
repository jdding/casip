# Synerise Gate-0 Probe

## Inputs

- Input dir: `data/synerise_dataset`
- Product properties: `data/synerise_dataset/product_properties.parquet`
- Unique users in windows: `22298361`

## Gate Counts

| Gate | Users | Mean pre | Mean gap | Mean val proxy | Mean val cart | Mean val search | Mean test buy |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre_any_gap_silent_val_proxy_test_buy | 17465 | 55.29 | 0.00 | 25.55 | 1.61 | 2.33 | 2.55 |
| pre_buy_gap_silent_val_proxy_test_buy | 8375 | 90.62 | 0.00 | 27.27 | 1.92 | 2.78 | 2.79 |
| pre_any_gap_low5_val_proxy_test_buy | 29200 | 66.50 | 0.99 | 25.48 | 1.70 | 2.32 | 2.59 |
| pre_any_gap_silent_val_cart_search_test_buy | 8347 | 72.15 | 0.00 | 47.75 | 3.36 | 4.88 | 2.98 |

## Target-Level Gate Counts

| Gate | Level | Users | Test purchase targets | OOV vs user history | OOV rate |
|---|---|---:|---:|---:|---:|
| pre_any_gap_silent_val_proxy_test_buy | sku | 17465 | 26549 | 20706 | 0.780 |
| pre_any_gap_silent_val_proxy_test_buy | category | 17465 | 3065 | 409 | 0.133 |
| pre_any_gap_silent_val_proxy_test_buy | price | 17465 | 100 | 0 | 0.000 |
| pre_buy_gap_silent_val_proxy_test_buy | sku | 8375 | 15290 | 12114 | 0.792 |
| pre_buy_gap_silent_val_proxy_test_buy | category | 8375 | 2456 | 326 | 0.133 |
| pre_buy_gap_silent_val_proxy_test_buy | price | 8375 | 100 | 0 | 0.000 |
| pre_any_gap_low5_val_proxy_test_buy | sku | 29200 | 41406 | 30567 | 0.738 |
| pre_any_gap_low5_val_proxy_test_buy | category | 29200 | 3569 | 357 | 0.100 |
| pre_any_gap_low5_val_proxy_test_buy | price | 29200 | 100 | 0 | 0.000 |
| pre_any_gap_silent_val_cart_search_test_buy | sku | 8347 | 16206 | 12152 | 0.750 |
| pre_any_gap_silent_val_cart_search_test_buy | category | 8347 | 2525 | 318 | 0.126 |
| pre_any_gap_silent_val_cart_search_test_buy | price | 8347 | 100 | 0 | 0.000 |

## Interpretation Stub

This is metadata-only Gate-0. It does not train a recommender, run CASP, or use test-aware source selection.
If item-level `sku` targets are too sparse, inspect category and price rows before rejecting Synerise.