"""Evaluate a rule-based threshold and a logistic regression classifier for
binary disc herniation detection, using the geometric features produced by
extract_features.py. Cross-validation is grouped by patient (GroupKFold) so
that discs from the same patient never appear in both train and test folds.

Usage:
    python classify.py <features_csv> <output_report_txt>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score,
)

FEATURE_COLS = [
    "disc_height_mm",
    "max_bulge_mm",
    "mean_bulge_mm",
    "bulge_normalized",
    "asymmetry_mm",
    "ap_depth_mm",
    "canal_encroach_frac",
    "mean_disc_ap_extent_mm",
]


def sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return sens, spec, ppv, (tn, fp, fn, tp)


def main(features_csv, report_txt):
    df = pd.read_csv(features_csv)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    y = df["disc_herniation"].to_numpy()
    groups = df["patient"].to_numpy()
    X = df[FEATURE_COLS].to_numpy()

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=== Dataset ===")
    log(f"Total disc-level rows usable for classification: {len(df)}")
    log(f"Positive (herniation=1): {int(y.sum())}  Negative: {int((1 - y).sum())}")
    log(f"Prevalence: {y.mean():.3%}")
    log("")

    # --- Rule-based baseline: flag as herniated if canal encroachment OR
    # bulge_normalized exceeds an empirically chosen threshold.
    log("=== Rule-based threshold baseline ===")
    rule_pred = ((df["canal_encroach_frac"] > 0) | (df["bulge_normalized"] > 1.0)).astype(int).to_numpy()
    sens, spec, ppv, cm = sens_spec(y, rule_pred)
    acc = accuracy_score(y, rule_pred)
    log("Rule: canal_encroach_frac > 0  OR  bulge_normalized > 1.0")
    log(f"Accuracy={acc:.3f}  Sensitivity={sens:.3f}  Specificity={spec:.3f}  PPV={ppv:.3f}")
    log(f"Confusion (tn,fp,fn,tp) = {cm}")
    log("")

    # --- Logistic regression, 5-fold GroupKFold cross-validation ---
    log("=== Logistic regression (5-fold GroupKFold by patient) ===")
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_aucs = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler().fit(X[tr_idx])
        X_tr = scaler.transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_tr, y[tr_idx])
        prob = clf.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        all_y_true.append(y[te_idx])
        all_y_pred.append(pred)
        all_y_prob.append(prob)
        if len(np.unique(y[te_idx])) > 1:
            fold_aucs.append(roc_auc_score(y[te_idx], prob))

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_prob = np.concatenate(all_y_prob)

    sens, spec, ppv, cm = sens_spec(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    log(f"Pooled across {n_splits} folds, class_weight='balanced', threshold=0.5")
    log(f"Accuracy={acc:.3f}  Sensitivity={sens:.3f}  Specificity={spec:.3f}  PPV={ppv:.3f}  AUC={auc:.3f}")
    log(f"Per-fold AUC: {[round(a, 3) for a in fold_aucs]}")
    log(f"Confusion (tn,fp,fn,tp) = {cm}")
    log("")

    # Fit on full data once more just to report feature coefficients
    # (for interpretability only - not used for the CV metrics above).
    scaler_full = StandardScaler().fit(X)
    clf_full = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf_full.fit(scaler_full.transform(X), y)
    log("=== Feature coefficients (fit on all data, standardized features) ===")
    for name, coef in sorted(zip(FEATURE_COLS, clf_full.coef_[0]), key=lambda t: -abs(t[1])):
        log(f"  {name:28s} {coef:+.3f}")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
