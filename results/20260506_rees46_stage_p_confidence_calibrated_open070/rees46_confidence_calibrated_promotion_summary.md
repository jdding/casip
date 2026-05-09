# REES46 Confidence-Calibrated Promotion

- Input per-user audit: `results/20260506_rees46_semantic_confidence_full/rees46_semantic_confidence_audit_per_user.csv`
- Base promotion policy: `q5__sem50__slot10__all`
- Selected gate: `top_semid_share_ge_0.545455`
- Selection: validation `net_gain@100` with Top-10 no-regression and ratio <= `0.0`
- Accounting rule: overlap is not gross recovery; it may only help as a confidence feature in future list-level calibrators.

## Selected Result

| Split | Open rate | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.601 | 3700 | 4173 | 4320 | 4605 | 49 | 0 | 49 | 0.000 |
| test | 0.522 | 3044 | 5350 | 6438 | 8840 | 83 | 37 | 46 | 0.446 |

## Top Validation Gates

| Gate | Open rate | Net@100 | Gross@100 | Cannibal@100 | Ratio@100 | Hit@10 | Hit@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `top_semid_margin_ge_0.0714286` | 0.801 | 55 | 55 | 0 | 0.000 | 3700 | 4326 |
| `top_semid_margin_ge_0.166667` | 0.708 | 51 | 51 | 0 | 0.000 | 3700 | 4322 |
| `semid_entropy_le_0.919611` | 0.800 | 50 | 50 | 0 | 0.000 | 3700 | 4321 |
| `top_semid_share_ge_0.545455` | 0.601 | 49 | 49 | 0 | 0.000 | 3700 | 4320 |
| `top_semid_margin_ge_0.29044` | 0.600 | 45 | 45 | 0 | 0.000 | 3700 | 4316 |
| `semid_entropy_le_0.880482` | 0.701 | 43 | 43 | 0 | 0.000 | 3700 | 4314 |
| `top_semid_share_ge_0.666667` | 0.524 | 42 | 42 | 0 | 0.000 | 3700 | 4313 |
| `top_semid_share_ge_0.545455__semid_entropy_le_0.812319` | 0.515 | 39 | 39 | 0 | 0.000 | 3700 | 4310 |
| `top_semid_count_ge_4` | 0.548 | 39 | 39 | 0 | 0.000 | 3700 | 4310 |
| `semid_entropy_le_0.812319` | 0.600 | 39 | 39 | 0 | 0.000 | 3700 | 4310 |
| `top_semid_margin_ge_0.29044__semid_entropy_le_0.812319` | 0.525 | 38 | 38 | 0 | 0.000 | 3700 | 4309 |
| `top_semid_margin_ge_0.428571` | 0.507 | 37 | 37 | 0 | 0.000 | 3700 | 4308 |
| `top_semid_margin_ge_0.428571__semid_entropy_le_0.812319` | 0.497 | 36 | 36 | 0 | 0.000 | 3700 | 4307 |
| `top_semid_share_ge_0.666667__semid_entropy_le_0.812319` | 0.478 | 35 | 35 | 0 | 0.000 | 3700 | 4306 |
| `top_semid_share_ge_0.666667__semid_entropy_le_0.721928` | 0.439 | 32 | 32 | 0 | 0.000 | 3700 | 4303 |
| `top_semid_margin_ge_0.428571__semid_entropy_le_0.721928` | 0.449 | 32 | 32 | 0 | 0.000 | 3700 | 4303 |
| `top_semid_share_ge_0.545455__semid_entropy_le_0.721928` | 0.455 | 32 | 32 | 0 | 0.000 | 3700 | 4303 |
| `top_semid_margin_ge_0.29044__semid_entropy_le_0.721928` | 0.457 | 32 | 32 | 0 | 0.000 | 3700 | 4303 |
| `semid_entropy_le_0.721928` | 0.506 | 32 | 32 | 0 | 0.000 | 3700 | 4303 |
| `top_semid_share_ge_0.8` | 0.405 | 30 | 30 | 0 | 0.000 | 3700 | 4301 |

## Interpretation

Selected `top_semid_share_ge_0.545455`. Test net@100=46, cannibal/gross=0.4457831325301205, gate_pass=True. This is a Stage P-A policy over a fixed promotion candidate list; it validates whether semantic confidence proxies can decide when to open promotion, but it is not yet a candidate-level learned scorer.
