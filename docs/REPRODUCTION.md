# Reproduction Notes

The following commands are the paper-facing command families. They assume the
external data layout in `data/README.md`.

## REES46

```bash
python3 scripts/train_agknet/build_rees46_protocol_artifact_parallel.py \
  --input data/Rees46 \
  --output-dir results/20260505_rees46_protocol_parallel_all_protocol

python3 scripts/train_agknet/rees46_semantic_bridge_audit.py \
  --artifact results/20260505_rees46_protocol_parallel_all_protocol/rees46_protocol_artifact.pkl \
  --output-dir results/20260505_rees46_semantic_bridge_audit

python3 scripts/train_agknet/rees46_confidence_calibrated_promotion.py \
  --input results/20260506_rees46_semantic_confidence_full/rees46_semantic_confidence_audit_per_user.csv \
  --output-dir results/20260506_rees46_stage_p_confidence_calibrated_open070 \
  --max-validation-ratio 0.0 \
  --min-validation-net 20 \
  --max-open-rate 0.70

python3 scripts/train_agknet/rees46_casp_solver_comparison.py \
  --output-dir results/20260506_rees46_casp_solver_comparison

python3 scripts/train_agknet/rees46_learned_casp_gate.py \
  --model logistic \
  --feature-set compact \
  --output-dir results/20260508_rees46_learned_casp_gate_logistic_compact

# Reviewer-risk alternate validation summaries use the same held-out test split
# after rebuilding the val_context_frac=0.6 validation rows.
python3 scripts/train_agknet/rees46_confidence_calibrated_promotion.py \
  --input results/20260509_rees46_altval_context060_semantic_confidence/rees46_semantic_confidence_audit_per_user.csv \
  --output-dir results/20260509_rees46_altval_context060_confidence_calibrated_open070 \
  --max-validation-ratio 0.0 \
  --min-validation-net 20 \
  --max-open-rate 0.70

python3 scripts/train_agknet/rees46_learned_casp_gate.py \
  --val-rows results/20260509_rees46_altval_context060_list_residual/rees46_stage_p_list_residual_val_rows.csv \
  --test-rows results/20260509_rees46_altval_context060_list_residual/rees46_stage_p_list_residual_test_rows.csv \
  --model logistic \
  --feature-set compact \
  --output-dir results/20260509_rees46_altval_context060_learned_casp_gate_logistic_compact
```

## Tmall

```bash
python3 scripts/train_agknet/tmall_gate0_fast_probe.py \
  --input-dir data/tmall \
  --output-dir results/20260506_tmall_gate0_probe \
  --chunksize 1000000 \
  --windows pre:501:1001,gap:1001:1101,val:1101:1111,test:1111:1112

bash scripts/run_tmall_support_audit.sh

python3 scripts/train_agknet/tmall_casp_policy_probe.py \
  --input-dir data/tmall \
  --output-dir results/20260506_tmall_casp_policy_probe_net50_10_deterministic \
  --chunksize 1000000 \
  --min-validation-net50 10

python3 scripts/train_agknet/tmall_casp_threshold_sensitivity.py \
  --output-dir results/20260506_tmall_casp_threshold_sensitivity_deterministic \
  --chunksize 1000000 \
  --validation-grid results/20260506_tmall_casp_policy_probe_net50_10_deterministic/tmall_casp_validation_grid.csv

python3 scripts/train_agknet/tmall_casp_policy_probe.py \
  --input-dir data/tmall \
  --output-dir results/20260509_tmall_casp_policy_probe_hit50_primary \
  --chunksize 1000000 \
  --selection-k 50 \
  --min-validation-net 10 \
  --min-validation-net50 10
```

## Synerise

```bash
python3 scripts/train_agknet/synerise_gate0_probe.py \
  --input-dir data/synerise_dataset \
  --output-dir results/20260507_synerise_gate0_probe \
  --batch-size 1000000 \
  --windows pre:2022-06-23:2022-10-10,gap:2022-10-10:2022-10-23,val:2022-10-23:2022-11-12,test:2022-11-12:2022-12-09

python3 scripts/train_agknet/synerise_no_training_support_audit.py \
  --input-dir data/synerise_dataset \
  --output-dir results/20260507_synerise_support_audit \
  --batch-size 1000000 \
  --windows pre:2022-06-23:2022-10-10,gap:2022-10-10:2022-10-23,val:2022-10-23:2022-11-12,test:2022-11-12:2022-12-09

python3 scripts/train_agknet/synerise_casp_policy_probe.py \
  --input-dir data/synerise_dataset \
  --output-dir results/20260507_synerise_casp_policy_probe \
  --batch-size 1000000
```

## Slice And Uncertainty Audit

```bash
python3 scripts/train_agknet/ccfa_evidence_audit.py \
  --output-dir results/20260507_ccfa_evidence_audit \
  --chunksize 1000000 \
  --bootstrap-samples 2000
```
