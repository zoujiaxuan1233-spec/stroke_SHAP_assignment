"""Reproduce Tasks 1 to 6, including all data and SHAP verification checks.

Run from any working directory: python /path/to/submission/code/run_assignment.py
No test outcomes are used to train models or choose decision thresholds.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED = 2021
NUM = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CAT = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURES = NUM + CAT
LABELS = {"age": "Age", "avg_glucose_level": "Average glucose", "bmi": "BMI",
          "hypertension": "Hypertension", "heart_disease": "Heart disease", "gender": "Gender",
          "ever_married": "Ever married", "work_type": "Work type", "Residence_type": "Residence",
          "smoking_status": "Smoking status"}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)), encoding="utf-8")


def scores(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0), "macro_f1": f1_score(y, pred, average="macro"),
            "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "n": len(y), "positive_support": int(np.sum(y)), "prevalence": float(np.mean(y))}


def best_threshold(y, p):
    grid = np.arange(1, 100) / 100
    values = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    # np.argmax selects the lowest threshold in an exact tie.
    ix = int(np.argmax(values))
    return float(grid[ix]), pd.DataFrame({"threshold": grid, "validation_f1": values})


def choose_class1(values, index):
    # SHAP <0.45 returned a list; >=0.45 can return (n, p, outputs).
    if isinstance(values, list):
        return np.asarray(values[index])
    values = np.asarray(values)
    return values[:, :, index] if values.ndim == 3 else values


def plot_force(base, phi, values, names, path, title):
    shap.force_plot(float(base), phi, values, feature_names=names, matplotlib=True,
                    show=False, figsize=(16, 3.8), contribution_threshold=.08,
                    text_rotation=15, out_names="Stroke class score")
    # The native force layout occupies the top margin; identify the model in
    # report captions to avoid a title overlapping the final output label.
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")
    # Standalone HTML works without IPython or internet access.
    plot = shap.force_plot(float(base), phi, values, feature_names=names, out_names="Stroke class score")
    shap.save_html(str(Path(path).with_suffix(".html")), plot)


def plot_summary(phi, x, path, title, max_display=15):
    np.random.seed(SEED)
    shap.summary_plot(phi, x, show=False, max_display=max_display, plot_size=(9, 6))
    plt.xlabel("SHAP value (stroke class score units)")
    plt.title(title, fontsize=13)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")


def local_table(phi, raw, transformed=None):
    names = list(raw.index)
    values = ["missing; training median imputed" if pd.isna(v) else str(v) for v in raw]
    rows = [{"feature": n, "value": v, "shap_value": float(s)} for n, v, s in zip(names, values, phi)]
    return sorted(rows, key=lambda r: -abs(r["shap_value"]))


def main(a):
    started = time.perf_counter()
    np.random.seed(SEED)
    out = a.output
    for sub in ["eda", "figures", "tables", "models", "explanations"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("assignment_eda", Path(__file__).with_name("00_data_exploration.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("Task 1: running the retained teaching script and extended EDA", flush=True)
    eda = module.run_eda(a.data, out / "eda")
    d = pd.read_csv(a.data)
    assert set(d.stroke.unique()) == {0, 1}
    assert not d.id.duplicated().any()
    X, y = d[FEATURES], d.stroke
    dev, test = train_test_split(np.arange(len(d)), test_size=.2, random_state=SEED, stratify=y)
    train, val = train_test_split(dev, test_size=.25, random_state=SEED, stratify=y.iloc[dev])
    assert not (set(train) & set(test) or set(val) & set(test) or set(train) & set(val))
    split = np.full(len(d), "train", dtype=object)
    split[val], split[test] = "validation", "test"
    pd.DataFrame({"row_index": np.arange(len(d)), "id": d.id, "split": split, "stroke": y}).to_csv(out / "tables/split_assignments.csv", index=False)
    # Explanation cohort and local case are fixed before observing model outputs.
    _, exp_rows = train_test_split(test, test_size=a.explain_samples, stratify=y.iloc[test], random_state=SEED)
    case_pos = int(np.flatnonzero(y.iloc[exp_rows].to_numpy() == 1)[0])
    case_row = int(exp_rows[case_pos])
    bg_rows = np.random.default_rng(SEED).choice(train, size=a.background_samples, replace=False)
    bg2_rows = np.random.default_rng(SEED + 1).choice(train, size=a.background_samples, replace=False)
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ], verbose_feature_names_out=False)
    Xt = prep.fit_transform(X.iloc[train]).astype(float)
    Xv = prep.transform(X.iloc[val]).astype(float)
    Xtest = prep.transform(X.iloc[test]).astype(float)
    Xe = prep.transform(X.iloc[exp_rows]).astype(float)
    bg = prep.transform(X.iloc[bg_rows]).astype(float)
    bg2 = prep.transform(X.iloc[bg2_rows]).astype(float)
    feature_names = prep.get_feature_names_out().tolist()
    assert feature_names[:len(NUM)] == NUM
    print(f"Training {len(train)} / validation {len(val)} / test {len(test)}; {len(feature_names)} transformed columns", flush=True)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                class_weight="balanced_subsample", random_state=SEED, n_jobs=1)
    rf.fit(Xt, y.iloc[train])
    rf_pipe = Pipeline([("preprocess", prep), ("classifier", rf)])
    # Alternative nonlinear neural network: scaling and oversampling use training only.
    scaler = StandardScaler().fit(Xt)
    ros = RandomOverSampler(random_state=SEED)
    Xm, ym = ros.fit_resample(scaler.transform(Xt), y.iloc[train])
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                        alpha=.01, batch_size=128, learning_rate_init=.001,
                        max_iter=600, early_stopping=False, random_state=SEED,
                        tol=1e-4, n_iter_no_change=25)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        mlp.fit(Xm, ym)
    mlp_warnings = sorted({str(w.message) for w in caught})
    assert all(np.isfinite(w).all() for w in mlp.coefs_)
    assert np.isfinite(mlp.loss_)
    mlp_pipe = Pipeline([("preprocess", prep), ("scale", scaler), ("classifier", mlp)])
    metrics = {}
    test_probs = {}
    for key, pipe in [("random_forest", rf_pipe), ("neural_network", mlp_pipe)]:
        pos = int(np.flatnonzero(pipe.classes_ == 1)[0])
        pv, pt = pipe.predict_proba(X.iloc[val])[:, pos], pipe.predict_proba(X.iloc[test])[:, pos]
        assert np.isfinite(pv).all() and np.isfinite(pt).all()
        threshold, curve = best_threshold(y.iloc[val], pv)
        curve.to_csv(out / f"tables/{key}_threshold_selection.csv", index=False)
        metrics[key] = {"validation": scores(y.iloc[val], pv, threshold), "test": scores(y.iloc[test], pt, threshold),
                        "test_at_0_5": scores(y.iloc[test], pt, .5), "positive_class_index": pos}
        test_probs[key] = pt
        joblib.dump(pipe, out / f"models/{key}_pipeline.joblib")
    # A no-skill comparator exposes why accuracy alone is misleading.
    metrics["always_negative"] = scores(y.iloc[test], np.zeros(len(test)), .5)
    print("Metrics:", json.dumps(metrics, indent=2), flush=True)
    table = pd.DataFrame([{ "model": k, **v["test"]} for k, v in metrics.items() if k != "always_negative"])
    table.to_csv(out / "tables/model_metrics.csv", index=False)
    dump(out / "metrics.json", metrics)
    predictions = pd.DataFrame({"row_index": test, "actual": y.iloc[test].to_numpy(), **test_probs})
    predictions.to_csv(out / "tables/test_predictions.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for key, p in test_probs.items():
        name = "Random forest" if key == "random_forest" else "Neural network"
        fpr, tpr, _ = roc_curve(y.iloc[test], p)
        precision, recall, _ = precision_recall_curve(y.iloc[test], p)
        axs[0].plot(fpr, tpr, label=f"{name} (AUC {metrics[key]['test']['roc_auc']:.3f})")
        axs[1].plot(recall, precision, label=f"{name} (AP {metrics[key]['test']['average_precision']:.3f})")
    axs[0].plot([0, 1], [0, 1], "k--", linewidth=.8)
    axs[1].axhline(y.iloc[test].mean(), color="black", linestyle="--", linewidth=.8, label="Test prevalence")
    axs[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Untouched test set ROC")
    axs[1].set(xlabel="Recall for recorded stroke", ylabel="Precision for recorded stroke", title="Untouched test set precision and recall")
    for ax in axs: ax.legend(fontsize=8)
    fig.savefig(out / "figures/performance_curves.png", dpi=200)
    plt.close(fig)
    print("Tasks 2 to 4: path-dependent Tree SHAP on class 1 probability", flush=True)
    t0 = time.perf_counter()
    # For sklearn RF, raw tree outputs are class probabilities. Path-dependent
    # Tree SHAP uses training node weights, which reflect class reweighting.
    tree = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent", model_output="raw")
    rf_index = metrics["random_forest"]["positive_class_index"]
    phi = choose_class1(tree.shap_values(Xe), rf_index)
    base = float(np.asarray(tree.expected_value)[rf_index])
    pe = rf_pipe.predict_proba(X.iloc[exp_rows])[:, rf_index]
    residual = float(np.max(np.abs(base + phi.sum(axis=1) - pe)))
    # For sklearn random forests, raw outputs are class probabilities. The
    # path-dependent explanation should be additive up to floating-point error.
    tree_additivity_tolerance = 1e-5
    assert phi.shape == Xe.shape and residual < tree_additivity_tolerance, (
        phi.shape, residual, tree_additivity_tolerance)
    tree_seconds = time.perf_counter() - t0
    # Sum signed encoded attributions first; then compute grouped mean |SHAP|.
    groups = []
    for name in feature_names:
        groups.append(next(c for c in FEATURES if name == c or name.startswith(c + "_")))
    grouped = np.column_stack([phi[:, np.array(groups) == c].sum(axis=1) for c in FEATURES])
    assert np.allclose(grouped.sum(axis=1), phi.sum(axis=1))
    rank = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(grouped).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    rank.to_csv(out / "tables/tree_global_importance.csv", index=False)
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(phi).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False).to_csv(out / "tables/tree_encoded_importance.csv", index=False)
    frame = pd.DataFrame(Xe, columns=feature_names)
    plot_force(base, phi[case_pos], Xe[case_pos], feature_names, out / "figures/tree_force.png",
               f"Random forest local explanation | held-out row {case_row} | class 1 probability")
    plot_summary(phi, frame, out / "figures/tree_summary.png", f"Random forest Tree SHAP | {len(exp_rows)} held-out records")
    tree_local = local_table(grouped[case_pos], X.iloc[case_row])
    pd.DataFrame(tree_local).to_csv(out / "tables/tree_local_contributions.csv", index=False)
    ex = shap.Explanation(values=grouped[case_pos], base_values=base,
                           data=np.array([LABELS[c] + " = " + str(X.iloc[case_row][c]) for c in FEATURES]),
                           feature_names=[LABELS[c] for c in FEATURES])
    # Display labels for mixed categorical values; additivity is unchanged.
    ex.data = None
    shap.plots.waterfall(ex, max_display=11, show=False)
    plt.xlabel("Stroke class score (probability scale)")
    plt.savefig(out / "figures/tree_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    # Same held-out cohort for the model-agnostic explainer. Kernel SHAP uses
    # empirical training rows; Tree SHAP uses the forest's training path weights.
    # Category codes are only a reversible SHAP transport format, never model inputs.
    # Entire categorical variables are replaced together, avoiding invalid one-hot masks.
    categories = {c: sorted(X.iloc[train][c].dropna().unique().tolist()) for c in CAT}
    def encode(raw):
        result = raw[FEATURES].copy()
        for c in CAT:
            result[c] = result[c].map({v: i for i, v in enumerate(categories[c])})
            if result[c].isna().any(): raise ValueError(f"Unseen category in explanation cohort: {c}")
        return result.to_numpy(dtype=float)
    def predict_mlp(codes):
        raw = pd.DataFrame(codes, columns=FEATURES)
        for c in CAT:
            integer = raw[c].to_numpy()
            if not np.allclose(integer, np.rint(integer)): raise ValueError("Non-integral category code")
            raw[c] = np.array(categories[c], dtype=object)[np.rint(integer).astype(int)]
        return mlp_pipe.predict_proba(raw)[:, metrics["neural_network"]["positive_class_index"]]
    Z, Zbg = encode(X.iloc[exp_rows]), encode(X.iloc[bg_rows])
    assert np.allclose(predict_mlp(Z), mlp_pipe.predict_proba(X.iloc[exp_rows])[:, 1])
    print("Task 6: Kernel SHAP for a different neural network model", flush=True)
    t0 = time.perf_counter()
    np.random.seed(SEED)
    kernel = shap.KernelExplainer(predict_mlp, Zbg, feature_names=FEATURES, link="identity")
    kphi = np.asarray(kernel.shap_values(Z, nsamples=a.kernel_nsamples, l1_reg=0.0, silent=True))
    kbase = float(kernel.expected_value)
    kp = predict_mlp(Z)
    kresidual = float(np.max(np.abs(kbase + kphi.sum(axis=1) - kp)))
    assert kphi.shape == Z.shape and kresidual < 1e-5
    assert np.isfinite(kphi).all()
    kernel_seconds = time.perf_counter() - t0
    # For this one case enumerate every nontrivial coalition (2**10 - 2).
    # This measures local sampling error; additivity alone cannot do so.
    exact_local = np.asarray(kernel.shap_values(Z[[case_pos]], nsamples=2**len(FEATURES)-2,
                                               l1_reg=0.0, silent=True))[0]
    kernel_local_sampling_error = float(np.max(np.abs(exact_local - kphi[case_pos])))
    krank = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(kphi).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    krank.to_csv(out / "tables/kernel_global_importance.csv", index=False)
    kernel_local = local_table(kphi[case_pos], X.iloc[case_row])
    pd.DataFrame(kernel_local).to_csv(out / "tables/kernel_local_contributions.csv", index=False)
    plot_force(kbase, kphi[case_pos], X.iloc[case_row].to_numpy(), FEATURES, out / "figures/kernel_force.png",
               f"Neural network local explanation | held-out row {case_row} | class 1 probability")
    display = pd.DataFrame(Z, columns=FEATURES)
    # Discrete category colours denote code values only; mapping is exported below.
    plot_summary(kphi, display, out / "figures/kernel_summary.png", f"Neural network Kernel SHAP | {len(exp_rows)} held-out records", 10)
    dump(out / "explanations/category_codes.json", categories)
    # Background sensitivity check for the local case only, with a new training sample.
    k2 = shap.KernelExplainer(predict_mlp, encode(X.iloc[bg2_rows]), feature_names=FEATURES, link="identity")
    np.random.seed(SEED)
    kphi2 = np.asarray(k2.shap_values(Z[[case_pos]], nsamples=a.kernel_nsamples, l1_reg=0.0, silent=True))[0]
    def stability(first, second):
        top = np.argsort(-np.abs(first))[:3]
        other = np.argsort(-np.abs(second))[:3]
        return {"original_top3": [FEATURES[i] for i in top], "second_top3": [FEATURES[i] for i in other],
                "top3_overlap": int(len(set(top) & set(other))),
                "original_top3_signs_preserved": bool(np.all(np.sign(first[top]) == np.sign(second[top]))),
                "max_abs_difference": float(np.max(np.abs(first-second)))}
    stability_results = {"kernel": stability(kphi[case_pos], kphi2)}
    np.savez_compressed(out / "explanations/shap_values.npz", tree=phi, tree_grouped=grouped,
                         kernel=kphi, tree_base=base, kernel_base=kbase, explanation_rows=exp_rows,
                         background_rows=bg_rows, second_background_rows=bg2_rows, case_position=case_pos,
                         tree_feature_names=np.array(feature_names), original_feature_names=np.array(FEATURES))
    for key in ["random_forest", "neural_network"]:
        m = metrics[key]["test"]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=17)
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["No stroke", "Stroke"], yticklabels=["No stroke", "Stroke"], xlabel="Predicted class", ylabel="Recorded class", title=key.replace("_", " ").title())
        fig.tight_layout();fig.savefig(out / f"figures/{key}_confusion.png", dpi=180);plt.close(fig)
    details = {"seed": SEED, "data_sha256": eda["sha256"],
        "split_counts": {s: {"n": int(sum(split==s)), "positive": int(y[split==s].sum())} for s in ["train", "validation", "test"]},
        "preprocessor_feature_names": feature_names, "training_numeric_medians": prep.named_transformers_["num"].statistics_.tolist(),
        "rf_params": rf.get_params(), "mlp_params": mlp.get_params(), "mlp_iterations": mlp.n_iter_,
        "mlp_loss": float(mlp.loss_), "mlp_warnings": mlp_warnings,
        "resampled_training_counts": {str(k): int(v) for k,v in pd.Series(ym).value_counts().items()},
        "explanation_sample_size": len(exp_rows), "explanation_positive_count": int(y.iloc[exp_rows].sum()),
        "background_size": len(bg_rows), "background_positive_count": int(y.iloc[bg_rows].sum()),
        "second_background_positive_count": int(y.iloc[bg2_rows].sum()),
        "kernel_nsamples": a.kernel_nsamples, "tree_seconds": tree_seconds, "kernel_seconds": kernel_seconds,
        "kernel_local_max_abs_sampling_difference": kernel_local_sampling_error,
        "tree_base": base, "kernel_base": kbase, "tree_max_residual": residual,
        "tree_additivity_tolerance": tree_additivity_tolerance, "kernel_max_residual": kresidual,
        "tree_global": rank.to_dict("records"), "kernel_global": krank.to_dict("records"),
        "case": {"row_index": case_row, "csv_line": case_row+2, "position_in_explanation": case_pos, "actual": int(y.iloc[case_row]),
                 "raw_features": X.iloc[case_row].to_dict(), "tree_score": float(pe[case_pos]), "kernel_score": float(kp[case_pos]),
                 "tree_sum": float(grouped[case_pos].sum()), "kernel_sum": float(kphi[case_pos].sum()),
                 "tree_local": tree_local, "kernel_local": kernel_local},
        "stability": stability_results,
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": {n: importlib.metadata.version(n) for n in ["numpy","pandas","scipy","scikit-learn","shap","imbalanced-learn","matplotlib","numba","joblib"]}},
        "total_seconds": time.perf_counter()-started}
    dump(out / "run_details.json", details)
    print("Completed:", json.dumps({k:details[k] for k in ["tree_seconds","kernel_seconds","tree_max_residual","kernel_max_residual","total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    warnings.simplefilter("once", RuntimeWarning)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/healthcare-dataset-stroke-data.csv")
    p.add_argument("--output", type=Path, default=ROOT / "outputs")
    p.add_argument("--explain-samples", type=int, default=120)
    p.add_argument("--background-samples", type=int, default=100)
    p.add_argument("--kernel-nsamples", type=int, default=512)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        main(args)
