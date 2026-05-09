# REES46 Dense-Feedback Ranker

- Train/dev/test users: `11820/2956/16954`
- Train examples / positives: `1241860` / `29980`
- Selected epoch: `3`
- Candidate top-k: `2500`; source top-k: `500`

## Result

| System | Hit@10 | Hit@50 | Hit@100 | Hit@500 |
|---|---:|---:|---:|---:|
| learned dev | 1831 | 2121 | 2190 | 2372 |
| learned test | 2766 | 4785 | 5629 | 8356 |
| fixed rule test | 2910 | 5211 | 6199 | 8517 |

## Training History

| Epoch | Dev Hit@10 | Dev Hit@50 | Dev Hit@100 | Dev Hit@500 |
|---:|---:|---:|---:|---:|
| 1 | 1830 | 2123 | 2191 | 2368 |
| 2 | 1828 | 2123 | 2191 | 2369 |
| 3 | 1831 | 2121 | 2190 | 2372 |
| 4 | 1825 | 2123 | 2190 | 2372 |
| 5 | 1825 | 2120 | 2191 | 2370 |
| 6 | 1825 | 2120 | 2188 | 2377 |
