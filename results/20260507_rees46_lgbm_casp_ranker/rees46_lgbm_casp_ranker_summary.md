# REES46 LGBMRanker CASP Action Baseline

- Selected threshold: `1.526433803090632`
- Feasible validation thresholds: `0`
- Candidate actions per user: `keep_existing` vs `open_semantic`.
- Selection rule: validation margin threshold under the same CASP constraints.

## Selected Result

| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.870 | 3700 | 4187 | 4333 | 4614 | 62 | 0 | 62 | 0.000 |
| test | 0.825 | 3044 | 5336 | 6444 | 8859 | 115 | 63 | 52 | 0.548 |

## Interpretation

LGBMRanker had no validation-feasible threshold under the CASP constraints; the reported row is the best fallback by validation net utility. Test net@100=52, ratio=0.5478260869565217, gate_pass=False. This is a LambdaMART-style learned action-ranker baseline over the same fixed semantic insertion action, not a new semantic source.
