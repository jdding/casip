# REES46 Stage P Transparent Calibrator

- Model: `tree`
- Selected threshold: `0.0748839`
- Training labels: gross@100=positive, cannibal@100=negative; neutral weight `0.05`
- Negative/cannibal penalty weight: `3.0`

## Selected Result

| Split | Open rate | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.586 | 3700 | 4186 | 4331 | 4614 | 61 | 1 | 60 | 0.016 |
| test | 0.649 | 3044 | 5331 | 6422 | 8842 | 83 | 53 | 30 | 0.639 |

## Tree

```text
|--- top_bucket_specificity <= 0.19
|   |--- confidence_inv_history <= 0.21
|   |   |--- class: 0
|   |--- confidence_inv_history >  0.21
|   |   |--- confidence_inv_history <= 0.24
|   |   |   |--- class: 0
|   |   |--- confidence_inv_history >  0.24
|   |   |   |--- class: 0
|--- top_bucket_specificity >  0.19
|   |--- inv_n_context <= 0.05
|   |   |--- top_semid_share <= 0.18
|   |   |   |--- class: 0
|   |   |--- top_semid_share >  0.18
|   |   |   |--- class: 0
|   |--- inv_n_context >  0.05
|   |   |--- confidence_context <= 1.48
|   |   |   |--- class: 0
|   |   |--- confidence_context >  1.48
|   |   |   |--- class: 0

```

## Interpretation

Selected threshold 0.0748839. Test net@100=30, cannibal/gross=0.6385542168674698, gate_pass=False. This is the transparent P-B calibrator; compare it against P-A before escalating to a richer model.
