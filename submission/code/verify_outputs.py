"""Independent checks of saved predictions, split membership and explanations."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

root = Path(__file__).resolve().parents[1]
o = root / "outputs"
d = pd.read_csv(root / "data/healthcare-dataset-stroke-data.csv")
r = json.loads((o / "run_details.json").read_text())
m = json.loads((o / "metrics.json").read_text())
s = pd.read_csv(o / "tables/split_assignments.csv")
p = pd.read_csv(o / "tables/test_predictions.csv")
z = np.load(o / "explanations/shap_values.npz")
assert hashlib.sha256((root / "data/healthcare-dataset-stroke-data.csv").read_bytes()).hexdigest() == r["data_sha256"]
assert len(s) == len(d) and s.row_index.is_unique
assert set(z["explanation_rows"]) <= set(s.loc[s.split.eq("test"), "row_index"])
assert set(z["background_rows"]) <= set(s.loc[s.split.eq("train"), "row_index"])
assert set(z["second_background_rows"]) <= set(s.loc[s.split.eq("train"), "row_index"])
for key, array, base in [("random_forest", "tree", "tree_base"), ("neural_network", "kernel", "kernel_base")]:
    pipe = joblib.load(o / f"models/{key}_pipeline.joblib")
    idx = int(np.flatnonzero(pipe.classes_ == 1)[0])
    score = pipe.predict_proba(d.iloc[p.row_index].drop(columns=["id", "stroke"]))[:, idx]
    assert np.isfinite(score).all() and np.allclose(score, p[key])
    pred = score >= m[key]["test"]["threshold"]
    assert list(confusion_matrix(p.actual, pred).ravel()) == [m[key]["test"][k] for k in ["tn", "fp", "fn", "tp"]]
    assert np.isclose(f1_score(p.actual, pred), m[key]["test"]["f1"])
    explained_score = pipe.predict_proba(d.iloc[z["explanation_rows"]].drop(columns=["id", "stroke"]))[:, idx]
    assert np.isfinite(z[array]).all()
    assert np.allclose(z[base] + z[array].sum(axis=1), explained_score, atol=1e-5)
assert np.allclose(z["tree"].sum(1), z["tree_grouped"].sum(1))
assert r["case"]["row_index"] == int(z["explanation_rows"][int(z["case_position"])])
assert len(list((o / "eda").glob("original_*.png"))) == 12
print("PASS: data fingerprint, all split memberships, saved predictions, confusion matrices, F1, both SHAP reconstructions, grouped contributions, local case and 12 original EDA figures")
