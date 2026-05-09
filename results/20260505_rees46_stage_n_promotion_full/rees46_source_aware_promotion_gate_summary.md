# REES46 Source-Aware Promotion Gate

- Semantic source: `semid_category_brand_context`
- Selection target: validation net gain @`100` with Top-10 no-regression preference
- Selected: `q5__sem50__slot10__all`

## Selected Result

| Split | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| validation | 3700 | 4185 | 4331 | 4614 |
| test | 3044 | 5330 | 6437 | 8862 |

## Net Gain Accounting

| Split | K | Gross | Cannibalized | Net | Cannibal/Gross | Inserted | Net / 100 inserted |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 10 | 0 | 0 | 0 |  | 0 |  |
| validation | 50 | 64 | 7 | 57 | 0.109 | 26844 | 0.212 |
| validation | 100 | 62 | 2 | 60 | 0.032 | 26844 | 0.224 |
| validation | 500 | 26 | 0 | 26 | 0.000 | 13493 | 0.193 |
| test | 10 | 0 | 0 | 0 |  | 0 |  |
| test | 50 | 130 | 119 | 11 | 0.915 | 78560 | 0.014 |
| test | 100 | 122 | 77 | 45 | 0.631 | 78560 | 0.057 |
| test | 500 | 52 | 6 | 46 | 0.115 | 44086 | 0.104 |

## Top Validation Policies

| Policy | Hit@10 | Hit@100 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|
| `q5__sem50__slot10__all` | 3700 | 4331 | 62 | 2 | 60 | 0.032 |
| `q5__sem50__slot50__all` | 3700 | 4331 | 62 | 2 | 60 | 0.032 |
| `q5__sem50__slot75__all` | 3700 | 4331 | 62 | 2 | 60 | 0.032 |
| `q5__sem50__slot90__all` | 3700 | 4331 | 62 | 2 | 60 | 0.032 |
| `q5__sem100__slot10__all` | 3700 | 4331 | 62 | 2 | 60 | 0.032 |
| `q5__sem100__slot50__all` | 3700 | 4331 | 62 | 2 | 60 | 0.032 |
| `q5__sem100__slot75__all` | 3700 | 4331 | 62 | 2 | 60 | 0.032 |
| `q5__sem100__slot90__all` | 3700 | 4331 | 62 | 2 | 60 | 0.032 |
| `q5__sem25__slot10__all` | 3700 | 4317 | 48 | 2 | 46 | 0.042 |
| `q5__sem25__slot50__all` | 3700 | 4317 | 48 | 2 | 46 | 0.042 |
| `q5__sem25__slot75__all` | 3700 | 4317 | 48 | 2 | 46 | 0.042 |
| `q5__sem25__slot90__all` | 3700 | 4317 | 48 | 2 | 46 | 0.042 |
| `q5__sem100__slot10__overlap` | 3700 | 4313 | 43 | 1 | 42 | 0.023 |
| `q5__sem100__slot10__tail_overlap` | 3700 | 4313 | 43 | 1 | 42 | 0.023 |
| `q5__sem100__slot50__overlap` | 3700 | 4313 | 43 | 1 | 42 | 0.023 |
| `q5__sem100__slot50__tail_overlap` | 3700 | 4313 | 43 | 1 | 42 | 0.023 |
| `q5__sem100__slot75__overlap` | 3700 | 4313 | 43 | 1 | 42 | 0.023 |
| `q5__sem100__slot75__tail_overlap` | 3700 | 4313 | 43 | 1 | 42 | 0.023 |
| `q5__sem100__slot90__overlap` | 3700 | 4313 | 43 | 1 | 42 | 0.023 |
| `q5__sem100__slot90__tail_overlap` | 3700 | 4313 | 43 | 1 | 42 | 0.023 |

## Interpretation

Validation selected `q5__sem50__slot10__all`. Test net@100 is 45 with cannibal/gross ratio 0.6311475409836066. Gate pass=False. This gate evaluates conservative physical promotion from the deep semantic pool, not a unified ranker.
