# Tmall Gate-0 Fast Probe

- User log: `data/tmall/data_format1/user_log_format1.csv`
- Train labels: `data/tmall/data_format1/train_format1.csv`
- Test labels: `data/tmall/data_format1/test_format1.csv`
- Rows seen: `54925330`
- Rows used in windows: `54925284`
- Unique users in windows: `424170`

## Gate Counts

| Gate | Users | Mean pre | Mean gap | Mean val proxy | Mean test purchase |
|---|---:|---:|---:|---:|---:|
| pre_any_gap_silent_val_proxy_test_purchase | 45574 | 24.33 | 0.00 | 17.65 | 2.69 |
| pre_any_gap_low3_val_proxy_test_purchase | 92428 | 28.10 | 0.95 | 17.63 | 2.70 |
| pre_purchase_gap_silent_val_proxy_test_purchase | 28497 | 32.74 | 0.00 | 17.48 | 2.83 |
| pre_any_gap_silent_val_cartfav_test_purchase | 16720 | 26.96 | 0.00 | 26.50 | 2.90 |

## Target-Level Gate Counts

| Gate | Level | Users | Test purchase targets | OOV vs user history | OOV rate |
|---|---|---:|---:|---:|---:|
| pre_any_gap_silent_val_proxy_test_purchase | item | 45574 | 49192 | 7205 | 0.146 |
| pre_any_gap_silent_val_proxy_test_purchase | merchant | 45574 | 4761 | 0 | 0.000 |
| pre_any_gap_silent_val_proxy_test_purchase | category | 45574 | 896 | 5 | 0.006 |
| pre_any_gap_silent_val_proxy_test_purchase | brand | 45574 | 4689 | 55 | 0.012 |
| pre_any_gap_low3_val_proxy_test_purchase | item | 92428 | 75801 | 8208 | 0.108 |
| pre_any_gap_low3_val_proxy_test_purchase | merchant | 92428 | 4943 | 0 | 0.000 |
| pre_any_gap_low3_val_proxy_test_purchase | category | 92428 | 990 | 4 | 0.004 |
| pre_any_gap_low3_val_proxy_test_purchase | brand | 92428 | 5155 | 42 | 0.008 |
| pre_purchase_gap_silent_val_proxy_test_purchase | item | 28497 | 37130 | 6417 | 0.173 |
| pre_purchase_gap_silent_val_proxy_test_purchase | merchant | 28497 | 4513 | 3 | 0.001 |
| pre_purchase_gap_silent_val_proxy_test_purchase | category | 28497 | 834 | 4 | 0.005 |
| pre_purchase_gap_silent_val_proxy_test_purchase | brand | 28497 | 4322 | 53 | 0.012 |
| pre_any_gap_silent_val_cartfav_test_purchase | item | 16720 | 25985 | 4092 | 0.157 |
| pre_any_gap_silent_val_cartfav_test_purchase | merchant | 16720 | 4045 | 5 | 0.001 |
| pre_any_gap_silent_val_cartfav_test_purchase | category | 16720 | 738 | 4 | 0.005 |
| pre_any_gap_silent_val_cartfav_test_purchase | brand | 16720 | 3751 | 54 | 0.014 |

## Target Turnover

| Pair | Level | Left targets | Right targets | Overlap | Jaccard | Right-new rate |
|---|---|---:|---:|---:|---:|---:|
| pre_to_val | item | 220955 | 35570 | 18259 | 0.0766 | 0.487 |
| pre_to_val | merchant | 4907 | 4669 | 4593 | 0.9217 | 0.016 |
| pre_to_val | category | 1265 | 962 | 948 | 0.7412 | 0.015 |
| pre_to_val | brand | 6290 | 4666 | 4473 | 0.6900 | 0.041 |
| pre_to_test | item | 220955 | 177737 | 56146 | 0.1639 | 0.684 |
| pre_to_test | merchant | 4907 | 4992 | 4905 | 0.9822 | 0.017 |
| pre_to_test | category | 1265 | 1157 | 1116 | 0.8545 | 0.035 |
| pre_to_test | brand | 6290 | 5926 | 5414 | 0.7959 | 0.086 |
| val_to_test | item | 35570 | 177737 | 22669 | 0.1189 | 0.872 |
| val_to_test | merchant | 4669 | 4992 | 4669 | 0.9353 | 0.065 |
| val_to_test | category | 962 | 1157 | 929 | 0.7807 | 0.197 |
| val_to_test | brand | 4666 | 5926 | 4550 | 0.7531 | 0.232 |