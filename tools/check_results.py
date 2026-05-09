#!/usr/bin/env python3
"""Lightweight consistency checks for fixed reviewer artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str):
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"expected {expected}, got {actual}")


def check_file(path: str) -> None:
    if not (ROOT / path).is_file():
        raise AssertionError(f"missing artifact: {path}")


def check_rees46_solver_table() -> None:
    path = "results/20260506_rees46_casp_solver_comparison/casp_solver_comparison.json"
    data = load_json(path)
    rows = data["rows"]
    if len(rows) < 8:
        raise AssertionError("solver comparison has too few rows")

    # Representative paper-facing rows by order in the materialized comparison.
    existing, exact, always_on, casp_pa, casp_l2 = rows[:5]
    assert existing["hit100"] == 6392
    assert exact["net100"] == -393
    assert always_on["net100"] == 45
    assert casp_pa["net100"] == 46
    assert casp_l2["hit100"] == 6446
    assert casp_l2["net100"] == 54
    assert_close(casp_l2["ratio100"], 0.44329896907216493)


def check_rees46_l2() -> None:
    path = (
        "results/20260508_rees46_learned_casp_gate_logistic_compact/"
        "rees46_learned_casp_gate_logistic_compact_summary.json"
    )
    selected = load_json(path)["selected"]["test"]
    assert selected["hit_counts"]["hit@100"] == 6446
    assert selected["net_gain"]["@100"] == 54
    assert_close(selected["cannibalization_ratio"]["@100"], 0.44329896907216493)


def check_tmall() -> None:
    path = (
        "results/20260506_tmall_casp_policy_probe_net50_10_deterministic/"
        "tmall_casp_policy_summary.json"
    )
    selected = load_json(path)["selected_test"]
    assert selected["policy"] == "cat_brand__sem50__n_cat_brand_le_20"
    assert selected["base@100"] == 25010
    assert selected["hit@100"] == 26080
    assert selected["net@100"] == 1070

    threshold_path = (
        "results/20260506_tmall_casp_threshold_sensitivity_deterministic/"
        "tmall_casp_catbrand_sem50_threshold_sensitivity.csv"
    )
    with (ROOT / threshold_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not any(row.get("gate") == "n_cat_brand_le_20" for row in rows):
        raise AssertionError("missing selected Tmall threshold row")


def check_synerise() -> None:
    path = "results/20260507_synerise_casp_policy_probe/synerise_casp_policy_summary.json"
    selected = load_json(path)["selected_test"]
    assert selected["policy"] == "category__bridge100__slot20__n_category_price_le_20"
    assert selected["base@100"] == 1630
    assert selected["hit@100"] == 1992
    assert selected["net@100"] == 362


def main() -> None:
    required = [
        "docs/ARTIFACTS.md",
        "docs/REPRODUCTION.md",
        "scripts/train_agknet/rees46_casp_solver_comparison.py",
        "scripts/train_agknet/tmall_casp_policy_probe.py",
        "scripts/train_agknet/synerise_casp_policy_probe.py",
        "figures/generate_casp_solver_feasibility.py",
    ]
    for path in required:
        check_file(path)
    check_rees46_solver_table()
    check_rees46_l2()
    check_tmall()
    check_synerise()
    print("OK: fixed reviewer artifacts match expected CASP table values.")


if __name__ == "__main__":
    main()
