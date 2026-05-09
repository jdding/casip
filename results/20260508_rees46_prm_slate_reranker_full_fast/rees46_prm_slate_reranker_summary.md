# REES46 PRM Slate-Aware Reranker

- Candidate pool: existing top `500` + semantic top `500`; slate length `1000`
- Semantic source: `semid_category_brand_context`
- Selected epoch: `4`
- Test latency: `0.6797` ms/user on `cuda`

## Test Result

| System | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PRM | 2654 | 4888 | 5866 | 8474 | 489 | 1015 | -526 | 2.076 |

## Training History

| Epoch | Loss | Dev Hit@100 | Dev Net@100 | Dev Ratio@100 |
|---:|---:|---:|---:|---:|
| 1 | 0.568058 | 2197 | +77 | 0.425 |
| 2 | 0.523184 | 2202 | +82 | 0.397 |
| 3 | 0.511153 | 2206 | +86 | 0.323 |
| 4 | 0.504931 | 2207 | +87 | 0.331 |
| 5 | 0.502294 | 2205 | +85 | 0.320 |
| 6 | 0.498749 | 2202 | +82 | 0.406 |
