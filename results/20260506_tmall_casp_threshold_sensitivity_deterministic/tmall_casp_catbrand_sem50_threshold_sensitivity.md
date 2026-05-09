# Tmall CASP Threshold Sensitivity

- Family: `cat_brand`, `sem_n=50`
- Test users: `45574`

| Gate | Val open | Val net@50 | Val net@100 | Test open | Test net@10 | Test net@50 | Test net@100 | Test ratio@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n_cat_brand_le_1 | 0.302 | 5 | 8 | 0.178 | 56 | 210 | 238 | 0.008 |
| n_cat_brand_le_2 | 0.471 | 9 | 12 | 0.312 | 80 | 349 | 429 | 0.012 |
| n_cat_brand_le_3 | 0.588 | 10 | 15 | 0.417 | 90 | 436 | 595 | 0.018 |
| n_cat_brand_le_5 | 0.742 | 14 | 23 | 0.572 | 100 | 474 | 782 | 0.019 |
| n_cat_brand_le_10 | 0.895 | 14 | 25 | 0.789 | 101 | 330 | 1000 | 0.028 |
| n_cat_brand_le_20 | 0.973 | 11 | 29 | 0.933 | 101 | 67 | 1070 | 0.039 |
| all | 1.000 | 7 | 28 | 1.000 | 101 | -137 | 1063 | 0.056 |