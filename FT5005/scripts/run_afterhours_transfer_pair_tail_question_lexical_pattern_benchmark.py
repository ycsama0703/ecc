#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import site
import sys
from pathlib import Path

import numpy as np

user_site = site.getusersitepackages()
user_site_removed = False
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)
    user_site_removed = True

try:
    from sklearn.decomposition import PCA
except ModuleNotFoundError:
    if user_site_removed and isinstance(user_site, str) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    from sklearn.decomposition import PCA

from dj30_qc_utils import load_csv_rows, write_csv, write_json
from run_afterhours_transfer_agreement_signal_benchmark import (
    MODEL_AGREED,
    MODEL_HARD_ABSTENTION,
    MODEL_PRE_ONLY,
    agreement_gain_target,
    build_temporal_rows,
    fit_ridge,
    summarize_significance,
)
from run_afterhours_transfer_pair_tail_text_benchmark import GEOMETRY_FEATURES, feature_matrix
from run_structured_baselines import metrics

TEXT_COL = "tail_top1_question_text"
PATTERN_GROUPS = {
    "clarify_modeling": {
        "wondering": ["wondering"],
        "wondering_if": ["wondering if"],
        "if_you_could": ["if you could"],
        "could_you": ["could you"],
        "can_you": ["can you"],
        "how_should_think": ["how should we think", "how should i think"],
        "how_do_think": ["how do we think", "how do i think"],
        "help_understand": ["help us understand", "help me understand"],
        "help_think": ["help us think"],
        "follow_up": ["follow up", "to clarify"],
        "give_color": ["give us color", "give any color", "color on", "color"],
        "talk_about": ["talk about", "can you talk"],
        "walk_through": ["walk us through", "walk through"],
        "frame_model": ["frame", "model"],
    },
    "quant_bridge": {
        "around": ["around"],
        "million": ["million"],
        "billion": ["billion"],
        "percent": ["percent", "%"],
        "basis_points": ["basis points", "bps"],
        "sequential": ["sequential"],
        "year_over_year": ["year over year", "yoy"],
        "quarter": ["quarter"],
        "cadence": ["cadence"],
        "bridge": ["bridge"],
        "trajectory": ["trajectory"],
        "outlook": ["outlook"],
        "visibility": ["visibility"],
    },
    "structural_probe": {
        "platform": ["platform"],
        "assets": ["assets"],
        "content": ["content"],
        "sports": ["sports"],
        "portfolio": ["portfolio"],
        "services": ["services"],
        "distribution": ["distribution"],
        "channel": ["channel"],
    },
}
FAMILY_SPECS = {
    "clarify_modeling_lex": ["clarify_modeling"],
    "quant_bridge_lex": ["quant_bridge"],
    "structural_probe_lex": ["structural_probe"],
    "clarify_quant_lex": ["clarify_modeling", "quant_bridge"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compact lexical-pattern families for the hardest-question transfer signal."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--text-views-csv",
        type=Path,
        default=Path("results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv"),
    )
    parser.add_argument(
        "--reference-predictions-csv",
        type=Path,
        default=Path(
            "results/afterhours_transfer_pair_tail_question_encoding_benchmark_real/"
            "afterhours_transfer_pair_tail_question_encoding_benchmark_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real"),
    )
    parser.add_argument("--train-split", default="val2020_test_post2020")
    parser.add_argument("--val-split", default="val2021_test_post2021")
    parser.add_argument("--test-split", default="val2022_test_post2022")
    parser.add_argument("--refit-train-splits", default="val2020_test_post2020,val2021_test_post2021")
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_regex(phrase: str):
    if phrase == "%":
        return re.compile(r"%")
    escaped = re.escape(phrase.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


PATTERN_REGEX = {
    group: {feature: [build_regex(pat) for pat in pats] for feature, pats in feature_map.items()}
    for group, feature_map in PATTERN_GROUPS.items()
}


def family_feature_names(family: str) -> list[str]:
    names = []
    for group in FAMILY_SPECS[family]:
        names.extend([f"{group}__{feature}" for feature in PATTERN_GROUPS[group]])
    return names


FAMILY_FEATURES = {family: family_feature_names(family) for family in FAMILY_SPECS}


def count_feature(text: str, regexes: list[re.Pattern]) -> float:
    return float(sum(len(regex.findall(text)) for regex in regexes))


def attach_lexical_features(rows, text_views_csv: Path):
    lookup = {row["event_key"]: row for row in load_csv_rows(text_views_csv.resolve()) if row.get("event_key")}
    coverage = {"temporal_rows": len(rows), "with_text": 0}
    for family in FAMILY_FEATURES:
        coverage[f"with_{family}"] = 0
    feature_activation = []
    for row in rows:
        trow = lookup.get(str(row["event_key"]))
        text = (trow.get(TEXT_COL, "") if trow else "").lower()
        if text:
            coverage["with_text"] += 1
        activated = {family: False for family in FAMILY_FEATURES}
        for group in PATTERN_GROUPS:
            for feature, regexes in PATTERN_REGEX[group].items():
                key = f"{group}__{feature}"
                value = count_feature(text, regexes)
                row[key] = value
        for family, names in FAMILY_FEATURES.items():
            if any(float(row.get(name, 0.0)) > 0.0 for name in names):
                coverage[f"with_{family}"] += 1
                activated[family] = True
        feature_activation.append({
            "event_key": row["event_key"],
            **{name: float(row.get(name, 0.0)) for names in FAMILY_FEATURES.values() for name in names},
            **{f"active__{family}": int(flag) for family, flag in activated.items()},
        })
    return coverage, feature_activation


def zscore_fit(train_x: np.ndarray, mats: list[np.ndarray]):
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1.0
    train_z = (train_x - means) / stds
    other = [(m - means) / stds for m in mats]
    return train_z, other


def safe_feature_matrix(rows, feature_names: list[str]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(feature_names)), dtype=float)
    return feature_matrix(rows, feature_names)


def build_factor(train_rows, val_rows, test_rows, feature_names: list[str]):
    train_x = safe_feature_matrix(train_rows, feature_names)
    val_x = safe_feature_matrix(val_rows, feature_names)
    test_x = safe_feature_matrix(test_rows, feature_names)
    train_z, [val_z, test_z] = zscore_fit(train_x, [val_x, test_x])
    if train_z.size == 0 or np.allclose(train_z, 0.0):
        zero_loadings = [{"feature": feature, "loading": 0.0} for feature in feature_names]
        return {
            "train_factor": np.zeros((len(train_rows), 1), dtype=float),
            "val_factor": np.zeros((len(val_rows), 1), dtype=float),
            "test_factor": np.zeros((len(test_rows), 1), dtype=float),
            "explained_variance_ratio": 0.0,
            "loadings": zero_loadings,
        }
    pca = PCA(n_components=1, random_state=42)
    train_f = pca.fit_transform(train_z)
    val_f = pca.transform(val_z) if len(val_z) else np.zeros((0, 1), dtype=float)
    test_f = pca.transform(test_z) if len(test_z) else np.zeros((0, 1), dtype=float)
    loadings = [
        {"feature": feature, "loading": float(loading)}
        for feature, loading in sorted(zip(feature_names, pca.components_[0].tolist()), key=lambda x: abs(x[1]), reverse=True)
    ]
    return {
        "train_factor": train_f,
        "val_factor": val_f,
        "test_factor": test_f,
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "loadings": loadings,
    }


def as_array(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=float)


def load_reference_predictions(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    lookup = {row["event_key"]: row for row in csv.DictReader(path.open())}
    refs = {}
    for name, col in [
        (MODEL_PRE_ONLY, MODEL_PRE_ONLY),
        (MODEL_HARD_ABSTENTION, MODEL_HARD_ABSTENTION),
        ("geometry_only", "geometry_only"),
        ("question_lsa4_bi", "question_lsa4_bi"),
    ]:
        refs[name] = np.asarray([float(lookup[key][col]) for key in keys], dtype=float)
    return refs


def top_coefficients(model, feature_names, limit=10):
    ridge = model.named_steps["ridge"]
    return [
        {"feature": f, "coefficient": float(c)}
        for f, c in sorted(zip(feature_names, ridge.coef_.tolist()), key=lambda x: abs(x[1]), reverse=True)[:limit]
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_temporal_rows(args.temporal_root.resolve())
    coverage, feature_activation_rows = attach_lexical_features(rows, args.text_views_csv.resolve())
    refit_train_splits = parse_list(args.refit_train_splits)
    alpha_grid = [float(x) for x in parse_list(args.alphas)]

    train_rows = [r for r in rows if r["split"] == args.train_split and int(r["agreement"]) == 1]
    val_rows = [r for r in rows if r["split"] == args.val_split and int(r["agreement"]) == 1]
    refit_rows = [r for r in rows if r["split"] in refit_train_splits and int(r["agreement"]) == 1]
    test_agreement_rows = [r for r in rows if r["split"] == args.test_split and int(r["agreement"]) == 1]
    test_full_rows = [r for r in rows if r["split"] == args.test_split]
    if not train_rows or not val_rows or not refit_rows or not test_agreement_rows or not test_full_rows:
        raise SystemExit("missing rows for lexical-pattern benchmark")

    train_gain = agreement_gain_target(train_rows)
    val_target = as_array(val_rows, "target")
    test_full_target = as_array(test_full_rows, "target")
    test_full_pre = as_array(test_full_rows, MODEL_PRE_ONLY)
    test_full_agreed = as_array(test_full_rows, MODEL_AGREED)
    test_full_hard = as_array(test_full_rows, MODEL_HARD_ABSTENTION)

    geometry_train = feature_matrix(train_rows, GEOMETRY_FEATURES)
    geometry_val = feature_matrix(val_rows, GEOMETRY_FEATURES)
    geometry_refit = feature_matrix(refit_rows, GEOMETRY_FEATURES)
    geometry_test = feature_matrix(test_agreement_rows, GEOMETRY_FEATURES)

    families = {
        "geometry_only": {
            "train_x": geometry_train,
            "val_x": geometry_val,
            "refit_x": geometry_refit,
            "test_x": geometry_test,
            "feature_names": GEOMETRY_FEATURES,
            "meta": None,
        }
    }
    loading_rows = []
    pattern_catalog_rows = []
    for group, feature_map in PATTERN_GROUPS.items():
        for feature, patterns in feature_map.items():
            pattern_catalog_rows.append({"group": group, "feature": feature, "patterns": " | ".join(patterns)})

    for family, features in FAMILY_FEATURES.items():
        tune = build_factor(train_rows, val_rows, test_agreement_rows, features)
        refit = build_factor(refit_rows, [], test_agreement_rows, features)
        factor_name = f"{family}_factor_pca1"
        factor_feature = [f"{factor_name}__score"]
        families[family] = {
            "train_x": safe_feature_matrix(train_rows, features),
            "val_x": safe_feature_matrix(val_rows, features),
            "refit_x": safe_feature_matrix(refit_rows, features),
            "test_x": safe_feature_matrix(test_agreement_rows, features),
            "feature_names": features,
            "meta": {
                "source_features": features,
                "source_groups": FAMILY_SPECS[family],
                "route_variant": "direct",
            },
        }
        families[factor_name] = {
            "train_x": tune["train_factor"],
            "val_x": tune["val_factor"],
            "refit_x": refit["train_factor"],
            "test_x": refit["test_factor"],
            "feature_names": factor_feature,
            "meta": {
                "source_features": features,
                "source_groups": FAMILY_SPECS[family],
                "route_variant": "factor_pca1",
                "explained_variance_ratio_tune": tune["explained_variance_ratio"],
                "explained_variance_ratio_refit": refit["explained_variance_ratio"],
            },
        }
        families[f"geometry_plus_{factor_name}"] = {
            "train_x": np.hstack([geometry_train, tune["train_factor"]]),
            "val_x": np.hstack([geometry_val, tune["val_factor"]]),
            "refit_x": np.hstack([geometry_refit, refit["train_factor"]]),
            "test_x": np.hstack([geometry_test, refit["test_factor"]]),
            "feature_names": GEOMETRY_FEATURES + factor_feature,
            "meta": {
                "source_features": features,
                "source_groups": FAMILY_SPECS[family],
                "route_variant": "geometry_plus_factor_pca1",
                "explained_variance_ratio_tune": tune["explained_variance_ratio"],
                "explained_variance_ratio_refit": refit["explained_variance_ratio"],
            },
        }
        for row in tune["loadings"]:
            loading_rows.append({"family": factor_name, **row})
        for row in refit["loadings"]:
            loading_rows.append({"family": f"{factor_name}__refit", **row})

    family_rows = []
    tuning_rows = []
    prediction_rows = [dict(r) for r in test_full_rows]
    family_predictions = {
        MODEL_PRE_ONLY: test_full_pre,
        MODEL_AGREED: test_full_agreed,
        MODEL_HARD_ABSTENTION: test_full_hard,
    }
    summaries = {}
    best_family = None
    best_r2 = None

    for family, payload in families.items():
        train_x = payload["train_x"]
        val_x = payload["val_x"]
        refit_x = payload["refit_x"]
        test_x = payload["test_x"]
        fnames = payload["feature_names"]
        best_alpha = None
        best_val_r2 = None
        best_val_payload = None
        for alpha in alpha_grid:
            model = fit_ridge(train_x, train_gain, alpha)
            val_signal = model.predict(val_x)
            val_pred = np.where(val_signal > 0.0, as_array(val_rows, MODEL_AGREED), as_array(val_rows, MODEL_PRE_ONLY))
            score = metrics(val_target, val_pred)
            row = {
                "family": family,
                "alpha": float(alpha),
                "val_r2": score["r2"],
                "val_rmse": score["rmse"],
                "val_mae": score["mae"],
                "val_use_agreed_share": float(np.mean(val_signal > 0.0)),
                "feature_count": len(fnames),
            }
            tuning_rows.append(row)
            if best_val_r2 is None or score["r2"] > best_val_r2:
                best_val_r2 = score["r2"]
                best_alpha = float(alpha)
                best_val_payload = row
        model = fit_ridge(refit_x, agreement_gain_target(refit_rows), best_alpha)
        test_signal = model.predict(test_x)
        test_agreement_pred = np.where(test_signal > 0.0, as_array(test_agreement_rows, MODEL_AGREED), as_array(test_agreement_rows, MODEL_PRE_ONLY))
        pred_lookup = {r["event_key"]: float(p) for r, p in zip(test_agreement_rows, test_agreement_pred)}
        signal_lookup = {r["event_key"]: float(s) for r, s in zip(test_agreement_rows, test_signal)}
        use_lookup = {r["event_key"]: int(s > 0.0) for r, s in zip(test_agreement_rows, test_signal)}
        full_pred = np.asarray([pred_lookup.get(r["event_key"], float(r[MODEL_PRE_ONLY])) for r in test_full_rows], dtype=float)
        family_predictions[family] = full_pred
        for row in prediction_rows:
            key = row["event_key"]
            row[family] = pred_lookup.get(key, float(row[MODEL_PRE_ONLY]))
            row[f"{family}__predicted_gain_signal"] = signal_lookup.get(key, 0.0)
            row[f"{family}__use_agreed"] = use_lookup.get(key, 0)
        full_metrics = metrics(test_full_target, full_pred)
        sig_vs_hard = summarize_significance(test_full_target, test_full_hard, full_pred, args.bootstrap_iters, args.perm_iters, args.seed)
        sig_vs_pre = summarize_significance(test_full_target, test_full_pre, full_pred, args.bootstrap_iters, args.perm_iters, args.seed)
        family_row = {
            "family": family,
            "feature_count": len(fnames),
            "selected_alpha": best_alpha,
            "val_r2": best_val_payload["val_r2"],
            "val_use_agreed_share": best_val_payload["val_use_agreed_share"],
            "test_full_r2": full_metrics["r2"],
            "test_full_rmse": full_metrics["rmse"],
            "test_full_mae": full_metrics["mae"],
            "test_full_p_mse_vs_hard": sig_vs_hard["mse_gain_pvalue"],
            "test_full_p_mse_vs_pre": sig_vs_pre["mse_gain_pvalue"],
            "test_use_agreed_share": float(np.mean(test_signal > 0.0)),
        }
        family_rows.append(family_row)
        summaries[family] = {
            "feature_names": fnames,
            "selected_alpha": best_alpha,
            "best_validation": best_val_payload,
            "test_full_metrics": full_metrics,
            "significance_vs_hard": sig_vs_hard,
            "significance_vs_pre": sig_vs_pre,
            "coef_rows": top_coefficients(model, fnames),
            "meta": payload["meta"],
        }
        if best_r2 is None or full_metrics["r2"] > best_r2:
            best_r2 = full_metrics["r2"]
            best_family = family

    keys = [str(r["event_key"]) for r in test_full_rows]
    refs = load_reference_predictions(args.reference_predictions_csv.resolve(), keys)
    best_vs_refs = {
        name: summarize_significance(test_full_target, ref, family_predictions[best_family], args.bootstrap_iters, args.perm_iters, args.seed)
        for name, ref in refs.items()
    }
    ref_lookup = {key: {name: float(refs[name][idx]) for name in refs} for idx, key in enumerate(keys)}
    for row in prediction_rows:
        for name, value in ref_lookup[row["event_key"]].items():
            row[name] = value

    activation_summary = []
    for family, features in FAMILY_FEATURES.items():
        feat_rows = [r for r in feature_activation_rows if any(float(r.get(name, 0.0)) > 0.0 for name in features)]
        activation_summary.append({
            "family": family,
            "active_row_share": float(len(feat_rows) / len(feature_activation_rows)) if feature_activation_rows else 0.0,
            "feature_count": len(features),
        })
        for name in features:
            vals = np.asarray([float(r.get(name, 0.0)) for r in feature_activation_rows], dtype=float)
            activation_summary.append({
                "family": family,
                "feature": name,
                "nonzero_share": float(np.mean(vals > 0.0)) if len(vals) else 0.0,
                "mean_value": float(np.mean(vals)) if len(vals) else 0.0,
                "max_value": float(np.max(vals)) if len(vals) else 0.0,
            })

    summary = {
        "config": {
            "temporal_root": str(args.temporal_root.resolve()),
            "text_views_csv": str(args.text_views_csv.resolve()),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "refit_train_splits": refit_train_splits,
            "test_split": args.test_split,
            "alpha_grid": alpha_grid,
            "text_col": TEXT_COL,
        },
        "coverage": coverage,
        "split_sizes": {
            "train_agreement_size": len(train_rows),
            "val_agreement_size": len(val_rows),
            "refit_agreement_size": len(refit_rows),
            "test_agreement_size": len(test_agreement_rows),
            "test_full_size": len(test_full_rows),
        },
        "reference": {
            "test_full_pre": metrics(test_full_target, test_full_pre),
            "test_full_agreed": metrics(test_full_target, test_full_agreed),
            "test_full_hard_abstention": metrics(test_full_target, test_full_hard),
        },
        "families": family_rows,
        "best_family": best_family,
        "best_family_summary": summaries.get(best_family),
        "best_vs_refs": best_vs_refs,
    }

    write_json(output_dir / "afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_family_overview.csv", family_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_tuning.csv", tuning_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_test_predictions.csv", prediction_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_loadings.csv", loading_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_pattern_catalog.csv", pattern_catalog_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_activation_summary.csv", activation_summary)


if __name__ == "__main__":
    main()
