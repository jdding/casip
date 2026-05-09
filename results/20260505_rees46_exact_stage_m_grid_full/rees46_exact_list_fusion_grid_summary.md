# REES46 Exact List-Fusion Grid Selection

- Semantic source: `semid_category_brand_context`
- Selected on validation Hit@10: `a0.5__b20__budget50`
- Selected alpha/beta/budget: `0.5` / `20.0` / `50`

## Selected Exact Result

| Split | System | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---|---:|---:|---:|---:|
| validation | existing | 3700 | 4128 | 4271 | 4588 |
| validation | semantic | 329 | 601 | 706 | 917 |
| validation | fused | 3700 | 4219 | 4355 | 4658 |
| test | existing | 3044 | 5319 | 6392 | 8816 |
| test | semantic | 676 | 1100 | 1253 | 1706 |
| test | fused | 3044 | 5079 | 5999 | 8858 |

## Selected Displacement

| Split | K | Gross recovery | Cannibalized hit | Net gain | Inserted items | Net gain / 100 inserted |
|---|---:|---:|---:|---:|---:|---:|
| validation | 10 | 0 | 0 | 0 | 0 |  |
| validation | 50 | 139 | 48 | 91 | 92392 | 0.098 |
| validation | 100 | 155 | 71 | 84 | 197873 | 0.042 |
| validation | 500 | 82 | 12 | 70 | 147056 | 0.048 |
| test | 10 | 0 | 0 | 0 | 0 |  |
| test | 50 | 283 | 523 | -240 | 273812 | -0.088 |
| test | 100 | 318 | 711 | -393 | 611306 | -0.064 |
| test | 500 | 155 | 113 | 42 | 472243 | 0.009 |

## Top Validation Configs

| Config | Hit@10 | Hit@50 | Hit@100 | Gross@10 | Cannibal@10 | Net@10 | Inserted@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `a0.5__b20__budget50` | 3700 | 4219 | 4355 | 0 | 0 | 0 | 0 |
| `a0.5__b20__budget100` | 3700 | 4219 | 4351 | 0 | 0 | 0 | 0 |
| `a0.5__b20__budget500` | 3700 | 4219 | 4351 | 0 | 0 | 0 | 0 |
| `a0.5__b20__budget25` | 3700 | 4218 | 4344 | 0 | 0 | 0 | 0 |
| `a1__b20__budget50` | 3700 | 4215 | 4356 | 0 | 0 | 0 | 0 |
| `a1__b20__budget100` | 3700 | 4215 | 4356 | 0 | 0 | 0 | 0 |
| `a1__b20__budget500` | 3700 | 4215 | 4356 | 0 | 0 | 0 | 0 |
| `a1__b20__budget25` | 3700 | 4215 | 4344 | 0 | 0 | 0 | 0 |
| `a0.25__b20__budget50` | 3700 | 4208 | 4355 | 0 | 0 | 0 | 0 |
| `a0.25__b20__budget100` | 3700 | 4208 | 4340 | 0 | 0 | 0 | 0 |
| `a0.25__b20__budget500` | 3700 | 4208 | 4340 | 0 | 0 | 0 | 0 |
| `a0.25__b20__budget25` | 3700 | 4203 | 4344 | 0 | 0 | 0 | 0 |
| `a1__b20__budget10` | 3700 | 4195 | 4332 | 0 | 0 | 0 | 0 |
| `a0.5__b20__budget10` | 3700 | 4195 | 4332 | 0 | 0 | 0 | 0 |
| `a0.25__b20__budget10` | 3700 | 4195 | 4332 | 0 | 0 | 0 | 0 |
| `a0.25__b20__budget5` | 3700 | 4173 | 4310 | 0 | 0 | 0 | 0 |
| `a0.5__b20__budget5` | 3700 | 4173 | 4310 | 0 | 0 | 0 | 0 |
| `a1__b20__budget5` | 3700 | 4173 | 4310 | 0 | 0 | 0 | 0 |
| `a1__b50__budget50` | 3700 | 4128 | 4368 | 0 | 0 | 0 | 0 |
| `a1__b50__budget100` | 3700 | 4128 | 4368 | 0 | 0 | 0 | 0 |

## Interpretation

Exact validation selected `a0.5__b20__budget50`. On test, fused Hit@10 changes by +0, with gross recovery 0 and cannibalized hits 0. This selection accounts for fixed-length displacement, unlike the earlier rank-only proxy.
