#!/usr/bin/env python3
"""Learned constrained opening baselines for Tmall CASP."""

import argparse
import json
import os
import platform
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tmall_casp_policy_probe as probe


FEATURES = [
    "n_cat_brand",
    "n_brand",
    "n_cat",
    "existing_len",
    "semantic_len",
    "selected_count",
    "semantic_existing_overlap10",
    "semantic_existing_overlap50",
    "semantic_existing_overlap100",
    "semantic_existing_overlap500",
    "selected_existing_overlap",
    "selected_new_count",
    "selected_new_share",
]


def parse_ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


def make_promoted(existing, semantic, sem_n):
    return probe.make_promoted(existing, semantic, sem_n)


def row_features(row, source, sem_n):
    existing = row["existing"]
    semantic = row["semantic"][source]
    selected = []
    blocked = set(existing[:10])
    for item in semantic:
        if item not in blocked:
            selected.append(item)
            blocked.add(item)
        if len(selected) >= sem_n:
            break
    existing_sets = {k: set(existing[:k]) for k in (10, 50, 100, 500)}
    sem_set = set(semantic[:max(probe.KS)])
    selected_set = set(selected)
    existing_selected = selected_set & set(existing[:max(probe.KS)])
    out = OrderedDict([
        ("n_cat_brand", int(row["features"]["n_cat_brand"])),
        ("n_brand", int(row["features"]["n_brand"])),
        ("n_cat", int(row["features"]["n_cat"])),
        ("existing_len", int(len(existing))),
        ("semantic_len", int(len(semantic))),
        ("selected_count", int(len(selected))),
        ("semantic_existing_overlap10", int(len(sem_set & existing_sets[10]))),
        ("semantic_existing_overlap50", int(len(sem_set & existing_sets[50]))),
        ("semantic_existing_overlap100", int(len(sem_set & existing_sets[100]))),
        ("semantic_existing_overlap500", int(len(sem_set & existing_sets[500]))),
        ("selected_existing_overlap", int(len(existing_selected))),
        ("selected_new_count", int(len(selected_set - set(existing[:max(probe.KS)])))),
    ])
    out["selected_new_share"] = float(out["selected_new_count"] / max(out["selected_count"], 1))
    return out


def action_row(row, source, sem_n, ks):
    existing = row["existing"]
    targets = row["targets"]
    promoted = make_promoted(existing, row["semantic"][source], sem_n)
    out = OrderedDict([
        ("user_id", row["user_id"]),
        ("source", source),
        ("sem_n", int(sem_n)),
    ])
    out.update(row_features(row, source, sem_n))
    for k in ks:
        base = probe.hit(existing, targets, k)
        new = probe.hit(promoted, targets, k)
        out[f"base@{k}"] = int(base)
        out[f"hit@{k}"] = int(new)
        out[f"gross@{k}"] = int(new and not base)
        out[f"cannibal@{k}"] = int(base and not new)
        out[f"net@{k}"] = int(new) - int(base)
    return out


def build_action_frame(rows, sources, sem_ns, ks):
    records = []
    for row in rows:
        for source in sources:
            for sem_n in sem_ns:
                records.append(action_row(row, source, sem_n, ks))
    return pd.DataFrame(records)


def model_features():
    return FEATURES + [
        "is_cat_brand",
        "is_brand",
        "is_cat",
        "sem_n",
        "log_sem_n",
        "log_n_cat_brand",
        "log_n_brand",
        "log_n_cat",
    ]


def add_model_features(df):
    out = df.copy()
    out["is_cat_brand"] = (out["source"] == "cat_brand").astype(int)
    out["is_brand"] = (out["source"] == "brand").astype(int)
    out["is_cat"] = (out["source"] == "cat").astype(int)
    out["log_sem_n"] = np.log1p(out["sem_n"].astype(float))
    out["log_n_cat_brand"] = np.log1p(out["n_cat_brand"].astype(float))
    out["log_n_brand"] = np.log1p(out["n_brand"].astype(float))
    out["log_n_cat"] = np.log1p(out["n_cat"].astype(float))
    return out


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
        return {
            "gross": HistGradientBoostingClassifier(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                max_leaf_nodes=args.max_leaf_nodes,
                l2_regularization=args.l2,
                random_state=args.seed,
            ),
            "cannibal": HistGradientBoostingClassifier(
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                max_leaf_nodes=args.max_leaf_nodes,
                l2_regularization=args.l2,
                random_state=args.seed + 1,
            ),
        }
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


def fit_model(kind, model, train_df, args):
    x = train_df[model_features()]
    k = args.selection_k
    if kind == "utility_hgb":
        y = train_df[f"net@{k}"].astype(float)
        weights = np.where(y != 0.0, 1.0, args.neutral_weight)
        model.fit(x, y, sample_weight=weights)
        return model
    if kind == "two_head_hgb":
        gross = (train_df[f"gross@{k}"].astype(int) > 0).astype(int)
        cann = (train_df[f"cannibal@{k}"].astype(int) > 0).astype(int)
        model["gross"].fit(x, gross, sample_weight=np.where(gross > 0, 1.0, args.neutral_weight))
        model["cannibal"].fit(x, cann, sample_weight=np.where(cann > 0, args.cannibal_penalty, args.neutral_weight))
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
    x = df[model_features()]
    if kind == "utility_hgb":
        return model.predict(x)
    if kind == "two_head_hgb":
        gross = model["gross"].predict_proba(x)[:, 1]
        cann = model["cannibal"].predict_proba(x)[:, 1]
        return gross - args.cannibal_penalty * cann
    if kind == "logistic":
        return model.predict_proba(x)[:, 1]
    raise ValueError(kind)


def evaluate_threshold(df, score, threshold, ks):
    work = df.copy()
    work["_score"] = score
    eligible = work[work["_score"] >= threshold]
    if eligible.empty:
        selected = eligible
    else:
        selected = eligible.sort_values(["user_id", "_score"], ascending=[True, False]).drop_duplicates("user_id", keep="first")
    user_base = work.drop_duplicates("user_id", keep="first")
    open_users = set(selected["user_id"].astype(str))
    counts = OrderedDict([
        ("n_users", int(len(user_base))),
        ("open_users", int(len(open_users))),
        ("open_rate", float(len(open_users) / len(user_base)) if len(user_base) else 0.0),
    ])
    for k in ks:
        base = int(user_base[f"base@{k}"].sum())
        gross = int(selected[f"gross@{k}"].sum()) if len(selected) else 0
        cann = int(selected[f"cannibal@{k}"].sum()) if len(selected) else 0
        counts[f"base@{k}"] = base
        counts[f"hit@{k}"] = base + gross - cann
        counts[f"gross@{k}"] = gross
        counts[f"cannibal@{k}"] = cann
        counts[f"net@{k}"] = gross - cann
        counts[f"ratio@{k}"] = float(cann / gross) if gross else None
    counts["threshold"] = float(threshold)
    return counts


def threshold_grid(score, n):
    vals = sorted(set(float(x) for x in np.quantile(score, np.linspace(0.0, 1.0, n))))
    vals.append(float(np.max(score) + 1e-9))
    vals.append(float(np.min(score) - 1e-9))
    return sorted(set(vals))


def select_threshold(df, score, args, ks):
    rows = [evaluate_threshold(df, score, threshold, ks) for threshold in threshold_grid(score, args.thresholds)]
    k = args.selection_k
    feasible = []
    for row in rows:
        ratio = row[f"ratio@{k}"]
        ratio_ok = ratio is None or ratio <= args.max_validation_ratio
        top10_ok = row["hit@10"] >= row["base@10"]
        net_ok = row[f"net@{k}"] >= args.min_validation_net
        net50_ok = row["net@50"] >= args.min_validation_net50
        open_ok = args.max_open_rate < 0 or row["open_rate"] <= args.max_open_rate
        if ratio_ok and top10_ok and net_ok and net50_ok and open_ok:
            feasible.append(row)
    pool = feasible if feasible else rows
    selected = max(
        pool,
        key=lambda r: (
            r[f"net@{k}"],
            -(r[f"ratio@{k}"] if r[f"ratio@{k}"] is not None else 0.0),
            r[f"gross@{k}"],
            -r["open_rate"],
        ),
    )
    return selected, rows, len(feasible)


def write_markdown(path, summary):
    lines = [
        "# Tmall Learned CASP Gate",
        "",
        f"- Model: `{summary['model_kind']}`",
        f"- Selected threshold: `{summary['selected_threshold']}`",
        f"- Feasible validation thresholds: `{summary['n_feasible_thresholds']}`",
        "- Unit of learning: one user-source-semantic-budget action row.",
        "",
        "## Selected Result",
        "",
        "| Split | Open | Hit@10 | Hit@50 | Hit@100 | Hit@500 | Gross@100 | Cannibal@100 | Net@100 | Ratio@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["validation", "test"]:
        row = summary["selected"][split]
        ratio = row["ratio@100"]
        ratio_text = "" if ratio is None else f"{ratio:.3f}"
        lines.append(
            f"| {split} | {row['open_rate']:.3f} | {row['hit@10']} | {row['hit@50']} | "
            f"{row['hit@100']} | {row['hit@500']} | {row['gross@100']} | {row['cannibal@100']} | "
            f"{row['net@100']} | {ratio_text} |"
        )
    lines.extend(["", "## Interpretation", "", summary["interpretation"]])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/tmall")
    parser.add_argument("--output-dir", default="results/20260507_tmall_learned_casp_gate")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--validation-windows", default="pre:501:1001,gap:1001:1101,val:1101:1106,test:1106:1111")
    parser.add_argument("--test-windows", default="pre:501:1001,gap:1001:1101,val:1101:1111,test:1111:1112")
    parser.add_argument("--gate", default="pre_any_gap_silent_val_proxy_test_purchase")
    parser.add_argument("--sources", default="cat_brand,brand,cat")
    parser.add_argument("--sem-ns", default="10,25,50,100")
    parser.add_argument("--model", choices=["utility_hgb", "two_head_hgb", "logistic"], default="two_head_hgb")
    parser.add_argument("--ks", default="10,50,100,500")
    parser.add_argument("--selection-k", type=int, default=100)
    parser.add_argument("--min-validation-net", type=int, default=5)
    parser.add_argument("--min-validation-net50", type=int, default=10)
    parser.add_argument("--max-validation-ratio", type=float, default=0.5)
    parser.add_argument("--max-open-rate", type=float, default=-1.0)
    parser.add_argument("--thresholds", type=int, default=101)
    parser.add_argument("--neutral-weight", type=float, default=0.03)
    parser.add_argument("--cannibal-penalty", type=float, default=3.0)
    parser.add_argument("--keep-neutral", action="store_true")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=8)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    ks = parse_ints(args.ks)
    sources = [x.strip() for x in args.sources.split(",") if x.strip()]
    sem_ns = parse_ints(args.sem_ns)
    user_log = probe.find_file(args.input_dir, "user_log_format1.csv")
    if not user_log:
        raise FileNotFoundError("Missing user_log_format1.csv")

    validation_rows = probe.build_eval_state(user_log, probe.parse_windows(args.validation_windows), args.chunksize, args.gate, 200)
    test_rows = probe.build_eval_state(user_log, probe.parse_windows(args.test_windows), args.chunksize, args.gate, 200)
    val_df = add_model_features(build_action_frame(validation_rows, sources, sem_ns, ks))
    test_df = add_model_features(build_action_frame(test_rows, sources, sem_ns, ks))

    model = fit_model(args.model, make_model(args.model, args), val_df, args)
    val_score = score_model(args.model, model, val_df, args)
    selected_val, val_grid, n_feasible = select_threshold(val_df, val_score, args, ks)
    threshold = float(selected_val["threshold"])
    test_score = score_model(args.model, model, test_df, args)
    selected_test = evaluate_threshold(test_df, test_score, threshold, ks)
    ratio = selected_test[f"ratio@{args.selection_k}"]
    gate_pass = (
        selected_test[f"net@{args.selection_k}"] > 0
        and selected_test["hit@10"] >= selected_test["base@10"]
        and (ratio is None or ratio <= args.max_validation_ratio)
    )
    interpretation = (
        f"Learned {args.model} selected a validation-feasible action threshold over user-source-budget rows. "
        f"Test net@{args.selection_k}={selected_test[f'net@{args.selection_k}']}, ratio={ratio}, "
        f"gate_pass={gate_pass}. This baseline tests whether a learned constrained gate can replace the transparent CASP rule."
    )
    summary = OrderedDict([
        ("args", vars(args)),
        ("provenance", OrderedDict([
            ("cwd", os.getcwd()),
            ("argv", sys.argv),
            ("hostname", platform.node()),
            ("python", sys.version.split()[0]),
        ])),
        ("model_kind", args.model),
        ("feature_names", model_features()),
        ("n_validation_users", len(validation_rows)),
        ("n_test_users", len(test_rows)),
        ("n_validation_action_rows", int(len(val_df))),
        ("n_test_action_rows", int(len(test_df))),
        ("train_counts", OrderedDict([
            ("gross_positive", int((val_df[f"gross@{args.selection_k}"] > 0).sum())),
            ("cannibal_positive", int((val_df[f"cannibal@{args.selection_k}"] > 0).sum())),
            ("neutral", int(((val_df[f"gross@{args.selection_k}"] == 0) & (val_df[f"cannibal@{args.selection_k}"] == 0)).sum())),
        ])),
        ("selected_threshold", threshold),
        ("n_feasible_thresholds", int(n_feasible)),
        ("selected", OrderedDict([("validation", selected_val), ("test", selected_test)])),
        ("top_validation_thresholds", sorted(
            val_grid,
            key=lambda r: (
                r[f"net@{args.selection_k}"],
                -(r[f"ratio@{args.selection_k}"] if r[f"ratio@{args.selection_k}"] is not None else 0.0),
                r[f"gross@{args.selection_k}"],
            ),
            reverse=True,
        )[:20]),
        ("gate_pass", bool(gate_pass)),
        ("runtime_seconds", time.time() - started),
        ("interpretation", interpretation),
    ])
    prefix = f"tmall_learned_casp_gate_{args.model}"
    summary_json = os.path.join(args.output_dir, f"{prefix}_summary.json")
    summary_md = os.path.join(args.output_dir, f"{prefix}_summary.md")
    grid_csv = os.path.join(args.output_dir, f"{prefix}_validation_grid.csv")
    selected_val_csv = os.path.join(args.output_dir, f"{prefix}_selected_validation.csv")
    selected_test_csv = os.path.join(args.output_dir, f"{prefix}_selected_test.csv")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(val_grid).to_csv(grid_csv, index=False)
    pd.DataFrame([selected_val]).to_csv(selected_val_csv, index=False)
    pd.DataFrame([selected_test]).to_csv(selected_test_csv, index=False)
    write_markdown(summary_md, summary)
    print(json.dumps({"summary": summary_json, "markdown": summary_md, "grid": grid_csv}, indent=2))


if __name__ == "__main__":
    main()
