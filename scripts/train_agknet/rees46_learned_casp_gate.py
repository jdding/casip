#!/usr/bin/env python3
"""Learned constrained opening baselines for REES46 CASP.

This script trains a deployable user-level gate for the fixed CASP promotion
action.  It is intentionally different from the transparent threshold solver:
the model learns a score from validation action outcomes, then a validation
threshold is selected under the same Top-10/net/cannibalization/open-rate
constraints used by CASP.
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rees46_confidence_calibrated_promotion as gate_eval


FEATURES = [
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


FEATURE_SETS = {
    "all": FEATURES + DERIVED_FEATURES,
    "compact": [
        "top_semid_share",
        "top_semid_margin",
        "semid_entropy",
        "log_n_semids",
        "top_bucket_specificity",
        "semantic_only_share",
    ],
    "share_only": [
        "top_semid_share",
    ],
}


def feature_names(args):
    return FEATURE_SETS[args.feature_set]


def utility_labels(df, k):
    return df[f"net@{k}"].astype(float)


def event_weights(df, k, neutral_weight):
    y = utility_labels(df, k)
    return np.where(y != 0.0, 1.0, float(neutral_weight))


def make_model(kind, args):
    if kind == "utility_hgb":
        return HistGradientBoostingRegressor(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2,
            random_state=args.seed,
        )
    if kind == "two_head_hgb":
        gross = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2,
            random_state=args.seed,
        )
        cann = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2,
            random_state=args.seed + 1,
        )
        return {"gross": gross, "cannibal": cann}
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=args.logistic_c,
                class_weight={0: args.cannibal_penalty, 1: 1.0},
                max_iter=2000,
                solver="liblinear",
                random_state=args.seed,
            ),
        )
    raise ValueError(kind)


def fit_model(kind, model, train_df, args, k):
    x = train_df[feature_names(args)]
    weights = event_weights(train_df, k, args.neutral_weight)
    if kind == "utility_hgb":
        model.fit(x, utility_labels(train_df, k), sample_weight=weights)
        return model
    if kind == "two_head_hgb":
        gross_y = (train_df[f"gross@{k}"].astype(int) > 0).astype(int)
        cann_y = (train_df[f"cannibal@{k}"].astype(int) > 0).astype(int)
        model["gross"].fit(x, gross_y, sample_weight=np.where(gross_y > 0, 1.0, args.neutral_weight))
        model["cannibal"].fit(x, cann_y, sample_weight=np.where(cann_y > 0, args.cannibal_penalty, args.neutral_weight))
        return model
    if kind == "logistic":
        gross = train_df[f"gross@{k}"].astype(int) > 0
        cann = train_df[f"cannibal@{k}"].astype(int) > 0
        mask = gross | cann
        if args.keep_neutral:
            mask = pd.Series(True, index=train_df.index)
        y = gross.astype(int)
        sample_weight = np.where(cann, args.cannibal_penalty, np.where(gross, 1.0, args.neutral_weight))
        model.fit(
            x.loc[mask],
            y.loc[mask],
            logisticregression__sample_weight=sample_weight[mask],
        )
        return model
    raise ValueError(kind)


def score_model(kind, model, df, args):
    x = df[feature_names(args)]
    if kind == "utility_hgb":
        return model.predict(x)
    if kind == "two_head_hgb":
        gross_prob = model["gross"].predict_proba(x)[:, 1]
        cann_prob = model["cannibal"].predict_proba(x)[:, 1]
        return gross_prob - args.cannibal_penalty * cann_prob
    if kind == "logistic":
        return model.predict_proba(x)[:, 1]
    raise ValueError(kind)


def eval_threshold(df, score, threshold, ks):
    tmp = df.copy()
    tmp["_learned_score"] = score
    config = OrderedDict([
        ("name", f"learned_score_ge_{threshold:.8g}"),
        ("conditions", [OrderedDict([
            ("feature", "_learned_score"),
            ("direction", "ge"),
            ("threshold", float(threshold)),
        ])]),
    ])
    return gate_eval.evaluate_config(tmp, config, ks)


def threshold_grid(score, n):
    quantiles = np.linspace(0.0, 1.0, n)
    vals = sorted(set(float(x) for x in np.quantile(score, quantiles)))
    vals.append(float(np.max(score) + 1e-9))
    vals.append(float(np.min(score) - 1e-9))
    return sorted(set(vals))


def select_threshold(df, score, args, ks):
    rows = [eval_threshold(df, score, threshold, ks) for threshold in threshold_grid(score, args.thresholds)]
    base_hit10 = int(((df["existing_rank"] >= 0) & (df["existing_rank"] < 10)).sum())
    key = f"@{args.selection_k}"
    feasible = []
    for row in rows:
        ratio = row["cannibalization_ratio"].get(key)
        ratio_ok = ratio is None or ratio <= args.max_validation_ratio
        top10_ok = row["hit_counts"]["hit@10"] >= base_hit10
        net_ok = row["net_gain"].get(key, 0) >= args.min_validation_net
        open_ok = args.max_open_rate < 0 or row["gate_open_rate"] <= args.max_open_rate
        if ratio_ok and top10_ok and net_ok and open_ok:
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
    return selected, rows, len(feasible)


def selected_per_user(df, score, threshold, ks):
    out = df[["user_id", "phase", "existing_rank", "fused_rank"]].copy()
    out["score"] = score
    out["opened"] = (out["score"] >= threshold).astype(int)
    effective_rank = out["existing_rank"].where(out["opened"] == 0, out["fused_rank"])
    out["effective_rank"] = effective_rank
    for k in ks:
        out[f"base_hit@{k}"] = ((out["existing_rank"] >= 0) & (out["existing_rank"] < k)).astype(int)
        out[f"casp_hit@{k}"] = ((out["effective_rank"] >= 0) & (out["effective_rank"] < k)).astype(int)
        out[f"delta@{k}"] = out[f"casp_hit@{k}"] - out[f"base_hit@{k}"]
    return out


def bootstrap_ci(values, n_boot, seed):
    arr = np.asarray(values, dtype=np.int16)
    n = len(arr)
    if n == 0:
        return OrderedDict([("observed", 0), ("ci_low", 0.0), ("ci_high", 0.0), ("p_boot_le_zero", None)])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, n), replace=True).sum(axis=1)
    return OrderedDict([
        ("observed", int(arr.sum())),
        ("ci_low", float(np.percentile(samples, 2.5))),
        ("ci_high", float(np.percentile(samples, 97.5))),
        ("p_boot_le_zero", float(np.mean(samples <= 0))),
    ])


def model_details(kind, model, args):
    if kind != "logistic":
        return []
    lr = model.named_steps["logisticregression"]
    return sorted(
        [OrderedDict([("feature", name), ("coef", float(coef))]) for name, coef in zip(feature_names(args), lr.coef_[0])],
        key=lambda row: abs(row["coef"]),
        reverse=True,
    )


def write_markdown(path, summary):
    lines = [
        "# REES46 Learned CASP Gate",
        "",
        f"- Model: `{summary['model_kind']}`",
        f"- Selected threshold: `{summary['selected_threshold']}`",
        f"- Feasible validation thresholds: `{summary['n_feasible_thresholds']}`",
        "- Training target: validation promotion-action utility, with target-derived columns excluded from features.",
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
    lines.extend([
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
    ])
    if summary["model_details"]:
        lines.extend(["", "## Model Details", "", "| Feature | Coef |", "|---|---:|"])
        for row in summary["model_details"][:20]:
            lines.append(f"| `{row['feature']}` | {row['coef']:.4f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-rows", default="results/20260506_rees46_stage_p_list_residual/rees46_stage_p_list_residual_val_rows.csv")
    parser.add_argument("--test-rows", default="results/20260506_rees46_stage_p_list_residual/rees46_stage_p_list_residual_test_rows.csv")
    parser.add_argument("--output-dir", default="results/20260507_rees46_learned_casp_gate")
    parser.add_argument("--model", choices=["utility_hgb", "two_head_hgb", "logistic"], default="two_head_hgb")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="all")
    parser.add_argument("--ks", default="10,50,100,500")
    parser.add_argument("--selection-k", type=int, default=100)
    parser.add_argument("--min-validation-net", type=int, default=20)
    parser.add_argument("--max-validation-ratio", type=float, default=0.0)
    parser.add_argument("--max-open-rate", type=float, default=0.70)
    parser.add_argument("--thresholds", type=int, default=101)
    parser.add_argument("--neutral-weight", type=float, default=0.03)
    parser.add_argument("--cannibal-penalty", type=float, default=3.0)
    parser.add_argument("--keep-neutral", action="store_true")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=8)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ks = parse_ints(args.ks)
    val_df = add_derived_features(pd.read_csv(args.val_rows))
    test_df = add_derived_features(pd.read_csv(args.test_rows))
    model = fit_model(args.model, make_model(args.model, args), val_df, args, args.selection_k)
    val_score = score_model(args.model, model, val_df, args)
    selected_val, val_rows, n_feasible = select_threshold(val_df, val_score, args, ks)
    threshold = float(selected_val["conditions"][0]["threshold"])
    test_score = score_model(args.model, model, test_df, args)
    selected_test = eval_threshold(test_df, test_score, threshold, ks)
    selected_val_per_user = selected_per_user(val_df, val_score, threshold, ks)
    selected_test_per_user = selected_per_user(test_df, test_score, threshold, ks)
    key = f"@{args.selection_k}"
    ratio = selected_test["cannibalization_ratio"].get(key)
    gate_pass = (
        selected_test["net_gain"].get(key, 0) > 0
        and (ratio is None or ratio < 0.5)
        and selected_test["hit_counts"]["hit@10"] >= int(((test_df["existing_rank"] >= 0) & (test_df["existing_rank"] < 10)).sum())
    )
    if args.model == "logistic":
        interpretation = (
            f"L2-logistic selected a validation-feasible score threshold from feature set "
            f"{args.feature_set}. Test net@{args.selection_k}={selected_test['net_gain'].get(key, 0)}, "
            f"ratio={ratio}, gate_pass={gate_pass}. This is a regularized CASP solver "
            "for the fixed promotion action; transparent thresholds remain the "
            "interpretability anchor."
        )
    else:
        interpretation = (
            f"Learned {args.model} selected a validation-feasible score threshold. "
            f"Test net@{args.selection_k}={selected_test['net_gain'].get(key, 0)}, "
            f"ratio={ratio}, gate_pass={gate_pass}. This is a learned constrained "
            "baseline for the CASP solver-family comparison."
        )
    summary = OrderedDict([
        ("args", vars(args)),
        ("model_kind", args.model),
        ("feature_set", args.feature_set),
        ("feature_names", feature_names(args)),
        ("train_counts", OrderedDict([
            ("n_validation_rows", int(len(val_df))),
            ("gross_positive", int((val_df[f"gross@{args.selection_k}"] > 0).sum())),
            ("cannibal_positive", int((val_df[f"cannibal@{args.selection_k}"] > 0).sum())),
            ("neutral", int(((val_df[f"gross@{args.selection_k}"] == 0) & (val_df[f"cannibal@{args.selection_k}"] == 0)).sum())),
        ])),
        ("selected_threshold", threshold),
        ("n_feasible_thresholds", int(n_feasible)),
        ("selected", OrderedDict([("validation", selected_val), ("test", selected_test)])),
        ("bootstrap", OrderedDict([
            (f"net@{k}", bootstrap_ci(selected_test_per_user[f"delta@{k}"].to_numpy(), args.bootstrap_samples, args.seed + k))
            for k in ks
        ])),
        ("top_validation_thresholds", sorted(
            val_rows,
            key=lambda r: (
                r["net_gain"].get(key, 0),
                -(r["cannibalization_ratio"].get(key) if r["cannibalization_ratio"].get(key) is not None else 0.0),
                r["gross_recovery"].get(key, 0),
            ),
            reverse=True,
        )[:20]),
        ("model_details", model_details(args.model, model, args)),
        ("gate_pass", bool(gate_pass)),
        ("interpretation", interpretation),
    ])
    prefix = f"rees46_learned_casp_gate_{args.model}"
    if args.model == "logistic":
        prefix = f"{prefix}_{args.feature_set}"
    summary_json = os.path.join(args.output_dir, f"{prefix}_summary.json")
    summary_md = os.path.join(args.output_dir, f"{prefix}_summary.md")
    grid_csv = os.path.join(args.output_dir, f"{prefix}_validation_grid.csv")
    score_csv = os.path.join(args.output_dir, f"{prefix}_scores.csv")
    selected_val_csv = os.path.join(args.output_dir, f"{prefix}_selected_validation.csv")
    selected_test_csv = os.path.join(args.output_dir, f"{prefix}_selected_test.csv")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    pd.json_normalize(val_rows).to_csv(grid_csv, index=False)
    pd.DataFrame({
        "phase": ["val"] * len(val_score) + ["test"] * len(test_score),
        "user_id": list(val_df["user_id"].astype(str)) + list(test_df["user_id"].astype(str)),
        "score": list(val_score) + list(test_score),
    }).to_csv(score_csv, index=False)
    selected_val_per_user.to_csv(selected_val_csv, index=False)
    selected_test_per_user.to_csv(selected_test_csv, index=False)
    write_markdown(summary_md, summary)
    print(json.dumps({
        "summary": summary_json,
        "markdown": summary_md,
        "grid": grid_csv,
        "scores": score_csv,
        "selected_validation": selected_val_csv,
        "selected_test": selected_test_csv,
    }, indent=2))


if __name__ == "__main__":
    main()
