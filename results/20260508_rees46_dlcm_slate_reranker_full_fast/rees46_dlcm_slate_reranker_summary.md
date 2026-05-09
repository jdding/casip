# REES46 DLCM Slate-Aware Reranker

- Candidate pool: existing top `500` + semantic top `500`; slate length `1000`
- Semantic source: `semid_category_brand_context`
- Selected epoch: `5`
- Test latency: `0.6315` ms/user on `cuda`

## Test Result

| System | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DLCM | 2657 | 4741 | 5773 | 8473 | 486 | 1105 | -619 | 2.274 |

## Training History

| Epoch | Loss | Dev Hit@100 | Dev Net@100 | Dev Ratio@100 |
|---:|---:|---:|---:|---:|
| 1 | 0.618597 | 2205 | +85 | 0.356 |
| 2 | 0.547203 | 2205 | +85 | 0.351 |
| 3 | 0.541660 | 2210 | +90 | 0.338 |
| 4 | 0.541098 | 2212 | +92 | 0.324 |
| 5 | 0.536683 | 2215 | +95 | 0.312 |
| 6 | 0.532238 | 2214 | +94 | 0.361 |
