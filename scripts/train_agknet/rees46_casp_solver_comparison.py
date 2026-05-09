#!/usr/bin/env python3
"""Aggregate CASP solver-family results into one paper-facing artifact."""

import argparse
import csv
import json
import os
from collections import OrderedDict


DEFAULT_OUTPUT = "results/20260506_rees46_casp_solver_comparison"


ARTIFACTS = OrderedDict([
    (
        "Existing source",
        {
            "kind": "promotion_base",
            "status": "reference",
            "path": "results/20260506_rees46_stage_p_confidence_calibrated_open070/rees46_confidence_calibrated_promotion_summary.json",
        },
    ),
    (
        "Exact validation fusion",
        {
            "kind": "exact_selected",
            "status": "net fail",
            "path": "results/20260505_rees46_exact_stage_m_grid_full/rees46_exact_list_fusion_grid_summary.json",
        },
    ),
    (
        "Always-on promotion",
        {
            "kind": "selected",
            "status": "ratio fail",
            "open_rate": 1.0,
            "path": "results/20260505_rees46_stage_n_promotion_full/rees46_source_aware_promotion_gate_summary.json",
        },
    ),
    (
        "CASP P-A: confidence solver",
        {
            "kind": "selected",
            "status": "feasible",
            "path": "results/20260506_rees46_stage_p_confidence_calibrated_open070/rees46_confidence_calibrated_promotion_summary.json",
        },
    ),
    (
        "CASP-L2: compact logistic gate",
        {
            "kind": "selected",
            "status": "feasible",
            "path": "results/20260508_rees46_learned_casp_gate_logistic_compact/rees46_learned_casp_gate_logistic_compact_summary.json",
        },
    ),
    (
        "P-B: shallow tree",
        {
            "kind": "selected",
            "status": "ratio fail",
            "path": "results/20260506_rees46_stage_p_transparent_tree/rees46_stage_p_tree_calibrator_summary.json",
        },
    ),
    (
        "P-B: interaction rules",
        {
            "kind": "selected",
            "status": "ratio fail",
            "path": "results/20260506_rees46_stage_p_interaction_rule_grid/rees46_stage_p_interaction_rule_grid_summary.json",
        },
    ),
    (
        "P-B: list-residual rules",
        {
            "kind": "selected",
            "status": "returns P-A",
            "path": "results/20260506_rees46_stage_p_list_residual/rees46_stage_p_list_residual_summary.json",
        },
    ),
    (
        "P-C: learned two-head HGB",
        {
            "kind": "selected",
            "status": "ratio fail",
            "path": "results/20260507_rees46_learned_casp_gate_two_head_hgb/rees46_learned_casp_gate_two_head_hgb_summary.json",
        },
    ),
    (
        "P-C: learned utility HGB",
        {
            "kind": "selected",
            "status": "ratio fail",
            "path": "results/20260507_rees46_learned_casp_gate_utility_hgb/rees46_learned_casp_gate_utility_hgb_summary.json",
        },
    ),
    (
        "P-C: LGBM action ranker",
        {
            "kind": "selected",
            "status": "ratio fail",
            "path": "results/20260507_rees46_lgbm_casp_ranker/rees46_lgbm_casp_ranker_summary.json",
        },
    ),
    (
        "DLCM slate reranker",
        {
            "kind": "slate_reranker",
            "status": "net fail",
            "path": "results/20260508_rees46_dlcm_slate_reranker_full_fast/rees46_dlcm_slate_reranker_summary.json",
        },
    ),
    (
        "PRM slate reranker",
        {
            "kind": "slate_reranker",
            "status": "net fail",
            "path": "results/20260508_rees46_prm_slate_reranker_full_fast/rees46_prm_slate_reranker_summary.json",
        },
    ),
])


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ratio(gross, cannibal):
    if gross == 0:
        return None
    return cannibal / gross


def from_metrics(name, status, metrics, artifact, open_rate=None):
    hit100 = metrics["hit_counts"]["hit@100"]
    gross = metrics["gross_recovery"]["@100"]
    cannibal = metrics["cannibalized_hit"]["@100"]
    net = metrics["net_gain"]["@100"]
    r = metrics.get("cannibalization_ratio", {}).get("@100")
    if r is None:
        r = ratio(gross, cannibal)
    if open_rate is None:
        open_rate = metrics.get("gate_open_rate")
    return OrderedDict([
        ("solver", name),
        ("open_rate", open_rate),
        ("hit100", hit100),
        ("gross100", gross),
        ("cannibal100", cannibal),
        ("net100", net),
        ("ratio100", r),
        ("status", status),
        ("artifact", artifact),
    ])


def collect_row(name, spec):
    data = load_json(spec["path"])
    kind = spec["kind"]
    if kind == "promotion_base":
        return from_metrics(name, spec["status"], data["base_test"], spec["path"], open_rate=0.0)
    if kind == "selected":
        return from_metrics(name, spec["status"], data["selected"]["test"], spec["path"], open_rate=spec.get("open_rate"))
    if kind == "slate_reranker":
        return from_metrics(name, spec["status"], data["test_summary"], spec["path"], open_rate=None)
    if kind == "exact_selected":
        test = data["selected_exact"]["test"]
        metrics = {
            "hit_counts": test["fused"]["hit_counts"],
            "gross_recovery": {"@100": test["displacement"]["@100"]["gross_recovery"]},
            "cannibalized_hit": {"@100": test["displacement"]["@100"]["cannibalized_hit"]},
            "net_gain": {"@100": test["displacement"]["@100"]["net_gain"]},
            "cannibalization_ratio": {
                "@100": ratio(
                    test["displacement"]["@100"]["gross_recovery"],
                    test["displacement"]["@100"]["cannibalized_hit"],
                )
            },
        }
        return from_metrics(name, spec["status"], metrics, spec["path"], open_rate=None)
    raise ValueError(f"unknown kind: {kind}")


def fmt_float(value):
    if value is None:
        return "--"
    return f"{value:.3f}"


def write_markdown(rows, path):
    with open(path, "w") as f:
        f.write("# CASP Solver-Family Comparison\n\n")
        f.write("All rows are test Hit@100 against the purchase-aligned existing source.\n\n")
        f.write("| Solver | Open | Hit@100 | Gross | Cann. | Net | Ratio | Status |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['solver']} | {fmt_float(row['open_rate'])} | {row['hit100']} | "
                f"{row['gross100']} | {row['cannibal100']} | {row['net100']:+d} | "
                f"{fmt_float(row['ratio100'])} | {row['status']} |\n"
            )
        f.write("\n## Source Artifacts\n\n")
        for row in rows:
            f.write(f"- {row['solver']}: `{row['artifact']}`\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = [collect_row(name, spec) for name, spec in ARTIFACTS.items()]

    csv_path = os.path.join(args.output_dir, "casp_solver_comparison.csv")
    json_path = os.path.join(args.output_dir, "casp_solver_comparison.json")
    md_path = os.path.join(args.output_dir, "casp_solver_comparison.md")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump({"rows": rows, "artifacts": ARTIFACTS}, f, indent=2)
    write_markdown(rows, md_path)

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
