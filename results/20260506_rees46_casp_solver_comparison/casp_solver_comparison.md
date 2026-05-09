# CASP Solver-Family Comparison

All rows are test Hit@100 against the purchase-aligned existing source.

| Solver | Open | Hit@100 | Gross | Cann. | Net | Ratio | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Existing source | 0.000 | 6392 | 0 | 0 | +0 | -- | reference |
| Exact validation fusion | -- | 5999 | 318 | 711 | -393 | 2.236 | net fail |
| Always-on promotion | 1.000 | 6437 | 122 | 77 | +45 | 0.631 | ratio fail |
| CASP P-A: confidence solver | 0.522 | 6438 | 83 | 37 | +46 | 0.446 | feasible |
| CASP-L2: compact logistic gate | 0.640 | 6446 | 97 | 43 | +54 | 0.443 | feasible |
| P-B: shallow tree | 0.649 | 6422 | 83 | 53 | +30 | 0.639 | ratio fail |
| P-B: interaction rules | 0.541 | 6430 | 81 | 43 | +38 | 0.531 | ratio fail |
| P-B: list-residual rules | 0.522 | 6438 | 83 | 37 | +46 | 0.446 | returns P-A |
| P-C: learned two-head HGB | 0.416 | 6417 | 64 | 39 | +25 | 0.609 | ratio fail |
| P-C: learned utility HGB | 0.449 | 6421 | 69 | 40 | +29 | 0.580 | ratio fail |
| P-C: LGBM action ranker | 0.825 | 6444 | 115 | 63 | +52 | 0.548 | ratio fail |
| DLCM slate reranker | -- | 5773 | 486 | 1105 | -619 | 2.274 | net fail |
| PRM slate reranker | -- | 5866 | 489 | 1015 | -526 | 2.076 | net fail |

## Source Artifacts

- Existing source: `results/20260506_rees46_stage_p_confidence_calibrated_open070/rees46_confidence_calibrated_promotion_summary.json`
- Exact validation fusion: `results/20260505_rees46_exact_stage_m_grid_full/rees46_exact_list_fusion_grid_summary.json`
- Always-on promotion: `results/20260505_rees46_stage_n_promotion_full/rees46_source_aware_promotion_gate_summary.json`
- CASP P-A: confidence solver: `results/20260506_rees46_stage_p_confidence_calibrated_open070/rees46_confidence_calibrated_promotion_summary.json`
- CASP-L2: compact logistic gate: `results/20260508_rees46_learned_casp_gate_logistic_compact/rees46_learned_casp_gate_logistic_compact_summary.json`
- P-B: shallow tree: `results/20260506_rees46_stage_p_transparent_tree/rees46_stage_p_tree_calibrator_summary.json`
- P-B: interaction rules: `results/20260506_rees46_stage_p_interaction_rule_grid/rees46_stage_p_interaction_rule_grid_summary.json`
- P-B: list-residual rules: `results/20260506_rees46_stage_p_list_residual/rees46_stage_p_list_residual_summary.json`
- P-C: learned two-head HGB: `results/20260507_rees46_learned_casp_gate_two_head_hgb/rees46_learned_casp_gate_two_head_hgb_summary.json`
- P-C: learned utility HGB: `results/20260507_rees46_learned_casp_gate_utility_hgb/rees46_learned_casp_gate_utility_hgb_summary.json`
- P-C: LGBM action ranker: `results/20260507_rees46_lgbm_casp_ranker/rees46_lgbm_casp_ranker_summary.json`
- DLCM slate reranker: `results/20260508_rees46_dlcm_slate_reranker_full_fast/rees46_dlcm_slate_reranker_summary.json`
- PRM slate reranker: `results/20260508_rees46_prm_slate_reranker_full_fast/rees46_prm_slate_reranker_summary.json`
