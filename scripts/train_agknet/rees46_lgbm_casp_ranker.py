#!/usr/bin/env python3
"""LambdaMART-style learned action-ranker baseline for REES46 CASP.

The unit of ranking is a per-user intervention action: keep the protected
existing list, or open the fixed CASP semantic insertion action.  The learned
ranker is selected on validation with the same source-protected constraints as
CASP, then reported once on test.
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


BASE_FEATURES = [
    "selected_count",
    "inserted_count@100",
    "displaced_count@100",
    "selected_semantic_only_count",
    "selected_tail_overlap_count",
    "selected_overlap_top500_count",
    "selected_semantic_rank_mean",
    "selected_semantic_rank_min",
    "selected_existing_rank_mean_fill",
    "selected_existing_rank_median_fill",
    "selected_existing_rank_min_fill",
    "selected_rank_gap_mean",
    "selected_rank_gap_min",
    "displaced_rank_mean",
    "displaced_rank_min",
    "displaced_rank_max",
    "collision_top10_semid",
    "collision_top100_semid",
    "selected_semid_count",
    "semantic_top50_semid_entropy",
    "semantic_top50_n_semids",
    "top_semid_count",
    "top_semid_share",
    "top_semid_margin",
    "semid_entropy",
    "n_semids",
    "top_bucket_size",
    "top_bucket_specificity",
]


def parse_ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


def add_derived_features(df):
    out = df.copy()
    out["inv_selected_existing_rank_mean"] = 1.0 / (1.0 + out["selected_existing_rank_mean_fill"].astype(float))
    out["inv_selected_existing_rank_min"] = 1.0 / (1.0 + out["selected_existing_rank_min_fill"].astype(float))
    out["log_top_bucket_size"] = np.log1p(np.maximum(out["top_bucket_size"].astype(float), 0.0))
    out["log_n_semids"] = np.log1p(np.maximum(out["n_semids"].astype(float), 0.0))
    out["confidence_entropy"] = out["top_semid_share"].astype(float) * (1.0 - out["semid_entropy"].astype(float))
    out["confidence_specificity"] = out["top_semid_share"].astype(float) * out["top_bucket_specificity"].astype(float)
    out["tail_overlap_share"] = out["selected_tail_overlap_count"].astype(float) / np.maximum(out["selected_count"].astype(float), 1.0)
    out["semantic_only_share"] = out["selected_semantic_only_count"].astype(float) / np.maximum(out["selected_count"].astype(float), 1.0)
    return out


DERIVED_FEATURES = [
    "inv_selected_existing_rank_mean",
    "inv_selected_existing_rank_min",
    "log_top_bucket_size",
    "log_n_semids",
    "confidence_entropy",
    "confidence_specificity",
    "tail_overlap_share",
    "semantic_only_share",
]


def feature_names():
    return BASE_FEATURES + DERIVED_FEATURES + ["is_open_action", "is_keep_action"]


def make_action_frame(per_user_df, selection_k):
    open_df = add_derived_features(per_user_df.copy())
    open_df["action"] = "open_semantic"
    open_df["is_open_action"] = 1
    open_df["is_keep_action"] = 0
    open_df["label"] = np.where(
        open_df[f"gross@{selection_k}"].astype(int) > 0,
        2,
        np.where(open_df[f"cannibal@{selection_k}"].astype(int) > 0, 0, 1),
    )

    keep_df = open_df.copy()
    keep_df["action"] = "keep_existing"
    keep_df["is_open_action"] = 0
    keep_df["is_keep_action"] = 1
    for col in BASE_FEATURES + DERIVED_FEATURES:
        keep_df[col] = 0.0
    keep_df["fused_rank"] = keep_df["existing_rank"]
    for col in [c for c in keep_df.columns if c.startswith("gross@") or c.startswith("cannibal@") or c.startswith("net@")]:
        keep_df[col] = 0
    keep_df["label"] = 1

    actions = pd.concat([keep_df, open_df], ignore_index=True)
    actions["user_id"] = actions["user_id"].astype(str)
    actions = actions.sort_values(["user_id", "is_open_action"], ascending=[True, True]).reset_index(drop=True)
    return actions


def fit_ranker(train_actions, args):
    x = train_actions[feature_names()]
    y = train_actions["label"].astype(int)
    group = train_actions.groupby("user_id", sort=False).size().to_numpy()
    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        verbose=-1,
        n_jobs=args.n_jobs,
    )
    model.fit(x, y, group=group)
    return model


def per_user_scores(actions, model):
    scored = actions.copy()
    scored["_score"] = model.predict(scored[feature_names()])
    piv = scored.pivot(index="user_id", columns="action", values="_score").reset_index()
    piv["margin"] = piv["open_semantic"] - piv["keep_existing"]
    return piv


def threshold_grid(margins, n):
    vals = sorted(set(float(x) for x in np.quantile(margins, np.linspace(0.0, 1.0, n))))
    vals.append(float(np.max(margins) + 1e-9))
    vals.append(float(np.min(margins) - 1e-9))
    return sorted(set(vals))


def evaluate_threshold(per_user_df, margins, threshold, ks):
    df = per_user_df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df = df.merge(margins[["user_id", "margin"]], on="user_id", how="left")
    df["_open"] = df["margin"] >= threshold
    out = OrderedDict([
        ("threshold", float(threshold)),
        ("n_users", int(len(df))),
        ("gate_open_users", int(df["_open"].sum())),
        ("gate_open_rate", float(df["_open"].mean()) if len(df) else 0.0),
        ("hit_counts", OrderedDict()),
        ("gross_recovery", OrderedDict()),
        ("cannibalized_hit", OrderedDict()),
        ("net_gain", OrderedDict()),
        ("cannibalization_ratio", OrderedDict()),
    ])
    for k in ks:
        existing_hit = (df["existing_rank"] >= 0) & (df["existing_rank"] < k)
        fused_hit = (df["fused_rank"] >= 0) & (df["fused_rank"] < k)
        effective_hit = existing_hit.where(~df["_open"], fused_hit)
        gross = int((df["_open"] & (~existing_hit) & effective_hit).sum())
        cann = int((df["_open"] & existing_hit & (~effective_hit)).sum())
        out["hit_counts"][f"hit@{k}"] = int(effective_hit.sum())
        out["gross_recovery"][f"@{k}"] = gross
        out["cannibalized_hit"][f"@{k}"] = cann
        out["net_gain"][f"@{k}"] = gross - cann
        out["cannibalization_ratio"][f"@{k}"] = float(cann / gross) if gross else None
    return out


def select_threshold(rows, args):
    key = f"@{args.selection_k}"
    base_hit10 = rows[0]["hit_counts"]["hit@10"] if rows else 0
    feasible = []
    for row in rows:
        ratio = row["cannibalization_ratio"].get(key)
        ratio_ok = ratio is None or ratio <= args.max_validation_ratio
        net_ok = row["net_gain"].get(key, 0) >= args.min_validation_net
        top10_ok = row["hit_counts"]["hit@10"] >= base_hit10
        open_ok = args.max_open_rate < 0 or row["gate_open_rate"] <= args.max_open_rate
        if ratio_ok and net_ok and top10_ok and open_ok:
            feasible.append(row)
    pool = feasible if feasible else rows
    selected = max(
        pool,
        key=lambda r: (
            r["net_gain"].get(key, 0),
            -(r["cannibalization_ratio"].get(key) if r["cannibalization_ratio"].get(key) is not None else 0.0),
            r["gross_recovery"].get(key, 0),
            -r["gate_open_rate"],
        ),
    )
    return selected, len(feasible)


def write_markdown(path, summary):
    lines = [
        "# REES46 LGBMRanker CASP Action Baseline",
        "",
        f"- Selected threshold: `{summary['selected_threshold']}`",
        f"- Feasible validation thresholds: `{summary['n_feasible_thresholds']}`",
        "- Candidate actions per user: `keep_existing` vs `open_semantic`.",
        "- Selection rule: validation margin threshold under the same CASP constraints.",
        "",
        "## Selected Result",
        "",
        "| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["validation", "test"]:
        row = summary["selected"][split]
        ratio = row["cannibalization_ratio"].get("@100")
        ratio_text = "" if ratio is None else f"{ratio:.3f}"
        lines.append(
            f"| {split} | {row['gate_open_rate']:.3f} | {row['hit_counts']['hit@10']} | "
            f"{row['hit_counts']['hit@50']} | {row['hit_counts']['hit@100']} | {row['hit_counts']['hit@500']} | "
            f"{row['gross_recovery']['@100']} | {row['cannibalized_hit']['@100']} | "
            f"{row['net_gain']['@100']} | {ratio_text} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-rows", default="results/20260506_rees46_stage_p_list_residual/rees46_stage_p_list_residual_val_rows.csv")
    parser.add_argument("--test-rows", default="results/20260506_rees46_stage_p_list_residual/rees46_stage_p_list_residual_test_rows.csv")
    parser.add_argument("--output-dir", default="results/20260507_rees46_lgbm_casp_ranker")
    parser.add_argument("--ks", default="10,50,100,500")
    parser.add_argument("--selection-k", type=int, default=100)
    parser.add_argument("--min-validation-net", type=int, default=20)
    parser.add_argument("--max-validation-ratio", type=float, default=0.0)
    parser.add_argument("--max-open-rate", type=float, default=0.70)
    parser.add_argument("--thresholds", type=int, default=101)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ks = parse_ints(args.ks)
    val_user = pd.read_csv(args.val_rows)
    test_user = pd.read_csv(args.test_rows)
    train_actions = make_action_frame(val_user, args.selection_k)
    test_actions = make_action_frame(test_user, args.selection_k)
    model = fit_ranker(train_actions, args)
    val_margins = per_user_scores(train_actions, model)
    test_margins = per_user_scores(test_actions, model)
    val_rows = [evaluate_threshold(val_user, val_margins, th, ks) for th in threshold_grid(val_margins["margin"], args.thresholds)]
    selected_val, n_feasible = select_threshold(val_rows, args)
    threshold = float(selected_val["threshold"])
    selected_test = evaluate_threshold(test_user, test_margins, threshold, ks)
    key = f"@{args.selection_k}"
    ratio = selected_test["cannibalization_ratio"].get(key)
    gate_pass = (
        selected_test["net_gain"].get(key, 0) > 0
        and (ratio is None or ratio < 0.5)
        and selected_test["hit_counts"]["hit@10"] >= int(((test_user["existing_rank"] >= 0) & (test_user["existing_rank"] < 10)).sum())
    )
    if n_feasible:
        feasibility_text = "selected a validation-feasible action-margin threshold"
    else:
        feasibility_text = "had no validation-feasible threshold under the CASP constraints; the reported row is the best fallback by validation net utility"
    interpretation = (
        f"LGBMRanker {feasibility_text}. "
        f"Test net@{args.selection_k}={selected_test['net_gain'].get(key, 0)}, "
        f"ratio={ratio}, gate_pass={gate_pass}. This is a LambdaMART-style learned "
        "action-ranker baseline over the same fixed semantic insertion action, not "
        "a new semantic source."
    )
    summary = OrderedDict([
        ("args", vars(args)),
        ("feature_names", feature_names()),
        ("n_validation_users", int(val_user["user_id"].nunique())),
        ("n_test_users", int(test_user["user_id"].nunique())),
        ("n_train_actions", int(len(train_actions))),
        ("label_counts", OrderedDict((str(k), int(v)) for k, v in train_actions["label"].value_counts().sort_index().items())),
        ("selected_threshold", threshold),
        ("n_feasible_thresholds", int(n_feasible)),
        ("selected", OrderedDict([("validation", selected_val), ("test", selected_test)])),
        ("top_validation_thresholds", sorted(
            val_rows,
            key=lambda r: (
                r["net_gain"].get(key, 0),
                -(r["cannibalization_ratio"].get(key) if r["cannibalization_ratio"].get(key) is not None else 0.0),
                r["gross_recovery"].get(key, 0),
            ),
            reverse=True,
        )[:20]),
        ("gate_pass", bool(gate_pass)),
        ("interpretation", interpretation),
    ])
    prefix = "rees46_lgbm_casp_ranker"
    summary_json = os.path.join(args.output_dir, f"{prefix}_summary.json")
    summary_md = os.path.join(args.output_dir, f"{prefix}_summary.md")
    grid_csv = os.path.join(args.output_dir, f"{prefix}_validation_grid.csv")
    margins_csv = os.path.join(args.output_dir, f"{prefix}_margins.csv")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    pd.json_normalize(val_rows).to_csv(grid_csv, index=False)
    pd.concat([
        val_margins.assign(split="validation"),
        test_margins.assign(split="test"),
    ], ignore_index=True).to_csv(margins_csv, index=False)
    write_markdown(summary_md, summary)
    print(json.dumps({"summary": summary_json, "markdown": summary_md, "grid": grid_csv, "margins": margins_csv}, indent=2))


if __name__ == "__main__":
    main()
