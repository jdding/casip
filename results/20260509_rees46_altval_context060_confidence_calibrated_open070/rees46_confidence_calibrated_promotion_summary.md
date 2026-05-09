# REES46 Confidence-Calibrated Promotion

- Input per-user audit: `results/20260509_rees46_altval_context060_semantic_confidence/rees46_semantic_confidence_audit_per_user.csv`
- Base promotion policy: `q5__sem50__slot10__all`
- Selected gate: `top_semid_share_ge_0.666667`
- Selection: validation `net_gain@100` with Top-10 no-regression and ratio <= `0.0`
- Accounting rule: overlap is not gross recovery; it may only help as a confidence feature in future list-level calibrators.

## Selected Result

| Split | Open rate | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.514 | 3627 | 4094 | 4212 | 4462 | 32 | 0 | 32 | 0.000 |
| test | 0.451 | 3044 | 5349 | 6435 | 8832 | 71 | 28 | 43 | 0.394 |

## Top Validation Gates

| Gate | Open rate | Net@100 | Gross@100 | Cannibal@100 | Ratio@100 | Hit@10 | Hit@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `top_semid_share_ge_0.666667` | 0.514 | 32 | 32 | 0 | 0.000 | 3627 | 4212 |
| `top_semid_share_ge_0.666667__semid_entropy_le_0.815797` | 0.466 | 26 | 26 | 0 | 0.000 | 3627 | 4206 |
| `top_semid_share_ge_0.666667__semid_entropy_le_0.73086` | 0.430 | 24 | 24 | 0 | 0.000 | 3627 | 4204 |
| `top_semid_margin_ge_0.6` | 0.403 | 22 | 22 | 0 | 0.000 | 3627 | 4202 |
| `top_semid_margin_ge_0.6__semid_entropy_le_0.73086` | 0.403 | 22 | 22 | 0 | 0.000 | 3627 | 4202 |
| `top_semid_margin_ge_0.6__semid_entropy_le_0.815797` | 0.403 | 22 | 22 | 0 | 0.000 | 3627 | 4202 |
| `top_semid_share_ge_0.75__semid_entropy_le_0.73086` | 0.399 | 21 | 21 | 0 | 0.000 | 3627 | 4201 |
| `top_semid_share_ge_0.75` | 0.426 | 21 | 21 | 0 | 0.000 | 3627 | 4201 |
| `top_semid_share_ge_0.75__semid_entropy_le_0.815797` | 0.426 | 21 | 21 | 0 | 0.000 | 3627 | 4201 |
| `semantic_candidate_count_le_308` | 0.700 | 48 | 49 | 1 | 0.020 | 3627 | 4228 |
| `n_semids_le_6` | 0.819 | 48 | 49 | 1 | 0.020 | 3627 | 4228 |
| `n_semids_le_4` | 0.727 | 44 | 45 | 1 | 0.022 | 3627 | 4224 |
| `n_semids_le_3` | 0.646 | 40 | 41 | 1 | 0.024 | 3627 | 4220 |
| `semantic_candidate_count_le_219` | 0.601 | 39 | 40 | 1 | 0.025 | 3627 | 4219 |
| `top_semid_share_ge_0.529412` | 0.601 | 38 | 39 | 1 | 0.026 | 3627 | 4218 |
| `semantic_candidate_count_le_191` | 0.502 | 33 | 34 | 1 | 0.029 | 3627 | 4213 |
| `n_semids_le_2` | 0.530 | 30 | 31 | 1 | 0.032 | 3627 | 4210 |
| `top_semid_margin_ge_0.4` | 0.510 | 29 | 30 | 1 | 0.033 | 3627 | 4209 |
| `top_semid_share_ge_0.529412__semid_entropy_le_0.815797` | 0.509 | 27 | 28 | 1 | 0.036 | 3627 | 4207 |
| `top_semid_margin_ge_0.4__semid_entropy_le_0.815797` | 0.491 | 26 | 27 | 1 | 0.037 | 3627 | 4206 |

## Interpretation

Selected `top_semid_share_ge_0.666667`. Test net@100=43, cannibal/gross=0.39436619718309857, gate_pass=True. This is a Stage P-A policy over a fixed promotion candidate list; it validates whether semantic confidence proxies can decide when to open promotion, but it is not yet a candidate-level learned scorer.
