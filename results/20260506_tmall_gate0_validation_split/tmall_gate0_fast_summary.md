# Tmall Gate-0 Fast Probe

- User log: `data/tmall/data_format1/user_log_format1.csv`
- Train labels: `data/tmall/data_format1/train_format1.csv`
- Test labels: `data/tmall/data_format1/test_format1.csv`
- Rows seen: `54925330`
- Rows used in windows: `44342651`
- Unique users in windows: `424170`

## Gate Counts

| Gate | Users | Mean pre | Mean gap | Mean val proxy | Mean test purchase |
|---|---:|---:|---:|---:|---:|
| pre_any_gap_silent_val_proxy_test_purchase | 1648 | 29.48 | 0.00 | 10.46 | 1.55 |
| pre_any_gap_low3_val_proxy_test_purchase | 3871 | 33.87 | 1.11 | 10.12 | 1.54 |
| pre_purchase_gap_silent_val_proxy_test_purchase | 1186 | 37.44 | 0.00 | 10.45 | 1.57 |
| pre_any_gap_silent_val_cartfav_test_purchase | 481 | 34.01 | 0.00 | 16.10 | 1.65 |

## Target-Level Gate Counts

| Gate | Level | Users | Test purchase targets | OOV vs user history | OOV rate |
|---|---|---:|---:|---:|---:|
| pre_any_gap_silent_val_proxy_test_purchase | item | 1648 | 1844 | 1044 | 0.566 |
| pre_any_gap_silent_val_proxy_test_purchase | merchant | 1648 | 1188 | 129 | 0.109 |
| pre_any_gap_silent_val_proxy_test_purchase | category | 1648 | 308 | 12 | 0.039 |
| pre_any_gap_silent_val_proxy_test_purchase | brand | 1648 | 1057 | 139 | 0.132 |
| pre_any_gap_low3_val_proxy_test_purchase | item | 3871 | 3768 | 1762 | 0.468 |
| pre_any_gap_low3_val_proxy_test_purchase | merchant | 3871 | 1973 | 69 | 0.035 |
| pre_any_gap_low3_val_proxy_test_purchase | category | 3871 | 446 | 7 | 0.016 |
| pre_any_gap_low3_val_proxy_test_purchase | brand | 3871 | 1766 | 104 | 0.059 |
| pre_purchase_gap_silent_val_proxy_test_purchase | item | 1186 | 1393 | 823 | 0.591 |
| pre_purchase_gap_silent_val_proxy_test_purchase | merchant | 1186 | 942 | 130 | 0.138 |
| pre_purchase_gap_silent_val_proxy_test_purchase | category | 1186 | 277 | 10 | 0.036 |
| pre_purchase_gap_silent_val_proxy_test_purchase | brand | 1186 | 848 | 123 | 0.145 |
| pre_any_gap_silent_val_cartfav_test_purchase | item | 481 | 623 | 384 | 0.616 |
| pre_any_gap_silent_val_cartfav_test_purchase | merchant | 481 | 452 | 106 | 0.235 |
| pre_any_gap_silent_val_cartfav_test_purchase | category | 481 | 190 | 8 | 0.042 |
| pre_any_gap_silent_val_cartfav_test_purchase | brand | 481 | 418 | 96 | 0.230 |

## Target Turnover

| Pair | Level | Left targets | Right targets | Overlap | Jaccard | Right-new rate |
|---|---|---:|---:|---:|---:|---:|
| pre_to_val | item | 220955 | 20876 | 11779 | 0.0512 | 0.436 |
| pre_to_val | merchant | 4907 | 4271 | 4205 | 0.8456 | 0.015 |
| pre_to_val | category | 1265 | 844 | 834 | 0.6541 | 0.012 |
| pre_to_val | brand | 6290 | 4041 | 3910 | 0.6089 | 0.032 |
| pre_to_test | item | 220955 | 22348 | 11747 | 0.0507 | 0.474 |
| pre_to_test | merchant | 4907 | 4285 | 4215 | 0.8469 | 0.016 |
| pre_to_test | category | 1265 | 852 | 843 | 0.6617 | 0.011 |
| pre_to_test | brand | 6290 | 4117 | 3966 | 0.6157 | 0.037 |
| val_to_test | item | 20876 | 22348 | 7654 | 0.2152 | 0.658 |
| val_to_test | merchant | 4271 | 4285 | 3887 | 0.8325 | 0.093 |
| val_to_test | category | 844 | 852 | 734 | 0.7630 | 0.138 |
| val_to_test | brand | 4041 | 4117 | 3492 | 0.7484 | 0.152 |