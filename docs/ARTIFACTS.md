# Artifact Ledger

This ledger maps the reviewer-facing result files to the paper-facing evidence
groups. Paths are relative to the package root.

## Table Groups

| Group | Fixed artifacts |
|---|---|
| Protocol summary | `results/20260505_rees46_protocol_parallel_all_no_training/`, `results/20260506_tmall_gate0_probe/`, `results/20260507_synerise_gate0_probe/` |
| REES46 baselines | `results/20260505_rees46_validation_safe_gate/`, `results/20260505_rees46_fine_blend/`, `results/20260505_rees46_dense_feedback_ranker/`, `results/20260505_rees46_purchase_event_gate_local/` |
| REES46 support audit | `results/20260505_rees46_semantic_bridge_audit/` |
| REES46 physical insertion | `results/20260506_rees46_casp_solver_comparison/`, `results/20260506_rees46_stage_p_confidence_calibrated_open070/`, `results/20260508_rees46_learned_casp_gate_logistic_compact/` |
| REES46 alternate-validation stability | `results/20260509_rees46_altval_context060_confidence_calibrated_open070/`, `results/20260509_rees46_altval_context060_learned_casp_gate_logistic_compact/` |
| REES46 learned and reranker baselines | `results/20260507_rees46_learned_casp_gate_two_head_hgb/`, `results/20260507_rees46_learned_casp_gate_utility_hgb/`, `results/20260507_rees46_lgbm_casp_ranker/`, `results/20260508_rees46_dlcm_slate_reranker_full_fast/`, `results/20260508_rees46_prm_slate_reranker_full_fast/` |
| Tmall Protocol B | `results/20260506_tmall_support_audit_v2/`, `results/20260506_tmall_casp_policy_probe_net50_10_deterministic/`, `results/20260506_tmall_casp_threshold_sensitivity_deterministic/`, `results/20260509_tmall_casp_policy_probe_hit50_primary/`, `results/20260507_tmall_learned_casp_gate_two_head_hgb/`, `results/20260507_tmall_learned_casp_gate_utility_hgb/` |
| Synerise Protocol C | `results/20260507_synerise_support_audit/`, `results/20260507_synerise_casp_policy_probe/`, `results/20260507_synerise_casp_policy_probe_altval_1031/` |
| Slice and uncertainty audit | `results/20260507_ccfa_evidence_audit/` |

## Excluded Artifacts

The package excludes raw logs, generated protocol pickles, per-user rows,
checkpoints, paper source files, compiled PDFs, and local run logs. These are
not needed for the reviewer smoke path.
