"""
train_iscx.py
=============
Entrainement et evaluation sur le dataset REEL ISCX-URL2016 (All.csv),
version pre-featurisee (79 features numeriques + cible URL_Type_obf_Type).

- Scenario 1 : detection BINAIRE (benign / malicious)
- Scenario 2 : classification MULTI-CLASSE (5 classes :
               benign / Defacement / phishing / malware / spam)

Modeles : Random Forest et XGBoost.
Metriques : Accuracy, Precision, Recall, F1, FPR.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from xgboost import XGBClassifier

DATA = "/mnt/user-data/uploads/All.csv"
OUT = "/home/claude/iscx_out"
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 5
TARGET = "URL_Type_obf_Type"
sns.set_style("whitegrid")

results = {"binary": {}, "multiclass": {}, "meta": {}}


# --------------------------------------------------------------------------
# Nettoyage
# --------------------------------------------------------------------------
def load_and_clean():
    df = pd.read_csv(DATA)
    n0 = len(df)
    # Retirer doublons
    df = df.drop_duplicates().reset_index(drop=True)
    # Separer cible / features
    y_raw = df[TARGET].astype(str)
    X = df.drop(columns=[TARGET])
    # Ne garder que les colonnes numeriques (toutes le sont deja ici)
    X = X.select_dtypes(include=[np.number])
    # Remplacer les infinis par NaN puis imputer par la mediane
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    print(f"  Lignes : {n0} -> {len(df)} apres deduplication")
    print(f"  Features numeriques : {X.shape[1]}")
    return X, y_raw, list(X.columns)


def fpr_from_cm(cm):
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        return fp / (fp + tn) if (fp + tn) else 0.0
    fprs = []
    for i in range(cm.shape[0]):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return float(np.mean(fprs))


def plot_confusion(cm, labels, title, fname):
    plt.figure(figsize=(5.5, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.ylabel("Vraie classe"); plt.xlabel("Prediction"); plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close()


def plot_importance(model, cols, title, fname, top=15):
    imp = pd.Series(model.feature_importances_, index=cols)
    imp = imp.sort_values(ascending=False).head(top)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=imp.values, y=imp.index, color="#2a6f97")
    plt.xlabel("Importance"); plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close()
    return imp


def metrics_block(y_te, pred, labels):
    cm = confusion_matrix(y_te, pred)
    avg = "binary" if len(labels) == 2 else "macro"
    return {
        "accuracy": round(accuracy_score(y_te, pred) * 100, 2),
        "precision": round(precision_score(y_te, pred, average=avg, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_te, pred, average=avg, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_te, pred, average=avg, zero_division=0) * 100, 2),
        "fpr": round(fpr_from_cm(cm) * 100, 2),
    }, cm


def train_pair(X, y, labels, prefix, title_suffix, cols):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, shuffle=True,
        random_state=RANDOM_STATE, stratify=y)

    binary = len(labels) == 2
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                random_state=RANDOM_STATE)
    rf.fit(X_tr, y_tr)
    m_rf, cm_rf = metrics_block(y_te, rf.predict(X_te), labels)
    plot_confusion(cm_rf, labels, f"Matrice de confusion - RF ({title_suffix})",
                   f"{prefix}_rf_cm.png")
    print(f"  RF  {title_suffix}: Acc={m_rf['accuracy']}%  F1={m_rf['f1']}%  FPR={m_rf['fpr']}%")

    xgb = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.2,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss" if binary else "mlogloss",
                        n_jobs=-1, random_state=RANDOM_STATE)
    xgb.fit(X_tr, y_tr)
    m_xgb, cm_xgb = metrics_block(y_te, xgb.predict(X_te), labels)
    plot_confusion(cm_xgb, labels, f"Matrice de confusion - XGB ({title_suffix})",
                   f"{prefix}_xgb_cm.png")
    print(f"  XGB {title_suffix}: Acc={m_xgb['accuracy']}%  F1={m_xgb['f1']}%  FPR={m_xgb['fpr']}%")

    imp = plot_importance(rf, cols, f"Importance des features - RF ({title_suffix})",
                          f"{prefix}_importance.png")
    return m_rf, m_xgb, imp, rf


def main():
    print("=== Chargement & nettoyage du dataset ISCX-URL2016 ===")
    X, y_raw, cols = load_and_clean()
    results["meta"] = {"source": "ISCX-URL2016 (All.csv)", "n_features": X.shape[1],
                       "random_state": RANDOM_STATE}

    # ---- Scenario 1 : BINAIRE ----
    print("\n=== SCENARIO 1 : DETECTION BINAIRE ===")
    y_bin = (y_raw.str.lower() != "benign").astype(int).values  # 1 = malicious
    labels_bin = ["benign", "malicious"]
    m_rf, m_xgb, imp, rf_bin = train_pair(X, y_bin, labels_bin, "bin", "binaire", cols)
    results["binary"] = {
        "n_samples": int(len(X)),
        "RandomForest": m_rf, "XGBoost": m_xgb,
        "top_features": {k: round(float(v), 4) for k, v in imp.head(10).items()},
    }

    # ---- Scenario 2 : MULTI-CLASSE ----
    print("\n=== SCENARIO 2 : MULTI-CLASSE (5 classes) ===")
    classes = sorted(y_raw.unique())
    cmap = {c: i for i, c in enumerate(classes)}
    y_mc = y_raw.map(cmap).values
    m_rf2, m_xgb2, imp2, _ = train_pair(X, y_mc, classes, "mc", "multi-classe", cols)
    results["multiclass"] = {
        "available": True, "classes": classes, "n_samples": int(len(X)),
        "RandomForest": m_rf2, "XGBoost": m_xgb2,
        "distribution": {k: int(v) for k, v in y_raw.value_counts().items()},
    }

    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
