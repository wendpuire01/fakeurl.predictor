"""
app.py — Interface Streamlit (detection d'URLs malveillantes)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from url_features import extract_features, FEATURE_NAMES

st.set_page_config(page_title="Detection d'URLs malveillantes", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
ISCX_MODELS = os.path.join(HERE, "iscx_models.joblib")
TARGET = "URL_Type_obf_Type"
RANDOM_STATE = 5


# --------------------------------------------------------------------------
# Style CSS sobre
# --------------------------------------------------------------------------
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; }
    .verdict-safe {
        background: #e6f4ea; border-left: 4px solid #34a853;
        padding: 16px; border-radius: 4px; margin: 8px 0;
    }
    .verdict-warning {
        background: #fff8e1; border-left: 4px solid #f9a825;
        padding: 16px; border-radius: 4px; margin: 8px 0;
    }
    .verdict-danger {
        background: #fce8e6; border-left: 4px solid #ea4335;
        padding: 16px; border-radius: 4px; margin: 8px 0;
    }
    .signal-item { padding: 3px 0; color: #555; font-size: 0.9em; font-family: monospace; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
@st.cache_resource
def load_iscx_models():
    if HAS_JOBLIB and os.path.exists(ISCX_MODELS):
        return joblib.load(ISCX_MODELS)
    return None


def clean_features(X, medians=None):
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    if medians is not None:
        X = X.fillna(pd.Series(medians))
        X = X.fillna(0)
    else:
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X


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


def plot_cm(cm, labels, title):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Prediction"); ax.set_ylabel("Vraie classe"); ax.set_title(title)
    thr = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=8)
    fig.tight_layout()
    return fig


SIGNAL_WEIGHTS = [
    ("ip_adress",          lambda f: f["ip_adress"] == 1,                3, "Adresse IP utilisee a la place d'un domaine"),
    ("number_at",          lambda f: f["number_at"] > 0,                 3, "Symbole '@' present dans l'URL (redirection trompeuse)"),
    ("https_domaine",      lambda f: f["https_domaine"] == 1,            2, "'https' dans le nom de domaine (imitation de securite)"),
    ("num_double_slash",   lambda f: f["num_double_slash"] > 0,          2, "Double slash de redirection detecte"),
    ("suspicious_words",   lambda f: f["suspicious_words"] == 1,         2, "Mot suspect detecte (login, paypal, secure, banking...)"),
    ("short_link",         lambda f: f["short_link"] == 1,               2, "Service de raccourcissement d'URL"),
    ("space_in_url",       lambda f: f["space_in_url"] == 1,             2, "Espace encode (%20) dans l'URL"),
    ("dash_in_domain",     lambda f: f["dash_in_domain"] == 1,           1, "Tiret dans le nom de domaine"),
    ("number_sub_domain",  lambda f: f["number_sub_domain"] >= 3,        2, f"Sous-domaines excessifs"),
    ("length_url",         lambda f: f["length_url"] > 75,               1, f"URL tres longue"),
    ("num_of_percentage",  lambda f: f["num_of_percentage"] > 5,         2, "Encodages suspects (trop de '%')"),
    ("protocol_https",     lambda f: not f["protocol_https"],             1, "Protocole non securise (HTTP)"),
]

MAX_SCORE = sum(w for _, _, w, _ in SIGNAL_WEIGHTS)


def detect_signals(feats: dict) -> tuple[list[tuple[str, int]], int]:
    triggered = []
    score = 0
    for key, condition, weight, label in SIGNAL_WEIGHTS:
        if condition(feats):
            msg = label
            if key == "number_sub_domain":
                msg = f"Sous-domaines excessifs ({feats['number_sub_domain']})"
            elif key == "length_url":
                msg = f"URL tres longue ({feats['length_url']} caracteres)"
            triggered.append((msg, weight))
            score += weight
    return triggered, score


# --------------------------------------------------------------------------
# En-tete
# --------------------------------------------------------------------------
st.title("Detection d'URLs malveillantes")
st.caption("Machine Learning (Random Forest) — analyse lexicale et classification")

tab_url, tab_pred, tab_train, tab_about = st.tabs([
    "Verifier une URL",
    "Predire par lot (CSV)",
    "Entrainer et evaluer",
    "A propos",
])

# ==========================================================================
# ONGLET 1 — VERIFICATION D'UNE URL
# ==========================================================================
with tab_url:
    st.subheader("Verifier si une URL est malveillante")
    st.markdown(
        "Saisissez une URL pour obtenir un verdict base sur l'analyse de ses caracteristiques "
        "lexicales (aucun acces reseau, aucun telechargement)."
    )

    url_input = st.text_input(
        "URL a analyser",
        placeholder="exemple : http://secure-paypal.verify.com/login?account=123",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyser = st.button("Analyser", type="primary")

    if analyser and url_input.strip():
        feats = extract_features(url_input.strip())
        signals, score = detect_signals(feats)

        # Seuils : 0 = sur / 1-3 = suspect / 4+ = malveillant
        if score == 0:
            verdict = "BENIGNE"
            verdict_class = "verdict-safe"
            verdict_detail = "Aucun signal d'alerte detecte."
        elif score <= 3:
            verdict = "SUSPECTE"
            verdict_class = "verdict-warning"
            verdict_detail = f"Score de risque : {score}/{MAX_SCORE} — quelques signaux presents."
        else:
            verdict = "MALVEILLANTE"
            verdict_class = "verdict-danger"
            verdict_detail = f"Score de risque : {score}/{MAX_SCORE} — plusieurs signaux critiques."

        st.markdown("---")
        st.markdown(
            f'<div class="{verdict_class}">'
            f'<strong>URL {verdict}</strong> &nbsp;|&nbsp; {verdict_detail}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if signals:
            st.markdown("**Signaux d'alerte detectes :**")
            for msg, weight in signals:
                stars = "+" * weight
                st.markdown(
                    f'<div class="signal-item">- [{stars}] {msg}</div>',
                    unsafe_allow_html=True,
                )
            st.caption("Legende : + faible risque, ++ risque modere, +++ risque eleve")
        else:
            st.markdown("Aucun signal d'alerte detecte.")

        with st.expander("Detail des 33 features extraites"):
            feat_df = pd.DataFrame(feats.items(), columns=["Feature", "Valeur"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    elif analyser and not url_input.strip():
        st.warning("Veuillez saisir une URL.")

# ==========================================================================
# ONGLET 2 — PREDICTION PAR LOT
# ==========================================================================
with tab_pred:
    st.subheader("Predire le type d'URLs a partir d'un fichier CSV")
    bundle = load_iscx_models()

    if bundle is None:
        st.warning("Modeles ISCX introuvables (`iscx_models.joblib`). "
                   "Utilisez l'onglet **Entrainer et evaluer** pour en creer.")
    else:
        st.success(f"Modeles charges — {len(bundle['features'])} features, "
                   f"classes : {', '.join(bundle['classes'])}.")

        mode = st.radio("Type de prediction :",
                        ["Detection binaire (benign / malicious)",
                         "Classification multi-classe (5 types)"],
                        horizontal=True)

        st.markdown("Chargez un CSV au format ISCX-URL2016 (79 colonnes de features). "
                    "La colonne `URL_Type_obf_Type` est optionnelle.")
        up = st.file_uploader("CSV a predire", type=["csv"], key="pred")

        if up is not None:
            df = pd.read_csv(up)
            has_target = TARGET in df.columns
            y_true_raw = df[TARGET].astype(str) if has_target else None
            X = df.drop(columns=[TARGET]) if has_target else df.copy()

            feats = bundle["features"]
            X = clean_features(X, bundle.get("medians"))
            for c in [c for c in feats if c not in X.columns]:
                X[c] = 0
            X = X[feats]

            binary = mode.startswith("Detection binaire")
            model = bundle["model_binary"] if binary else bundle["model_multiclass"]
            preds = model.predict(X)

            if binary:
                pred_names = ["benign" if p == 0 else "malicious" for p in preds]
            else:
                pred_names = [bundle["classes"][p] for p in preds]

            out = df.copy()
            out.insert(0, "PREDICTION", pred_names)
            n = len(out)

            st.markdown(f"**{n:,} URLs analysees**")
            counts = pd.Series(pred_names).value_counts()
            c1, c2 = st.columns([1, 1])
            with c1:
                st.bar_chart(counts)
            with c2:
                if binary:
                    mal = int((np.array(pred_names) == "malicious").sum())
                    st.metric("URLs malveillantes", f"{mal:,}", f"{mal/n*100:.1f} %")
                    st.metric("URLs benignes", f"{n-mal:,}")

            if has_target:
                if binary:
                    y_true = (y_true_raw.str.lower() != "benign").astype(int).values
                else:
                    cmap = {c: i for i, c in enumerate(bundle["classes"])}
                    y_true = y_true_raw.map(cmap).values
                acc = accuracy_score(y_true, preds) * 100
                st.info(f"Accuracy sur ce fichier (verite terrain presente) : **{acc:.2f} %**")

            st.markdown("**Resultats detailles (200 premieres lignes)**")
            st.dataframe(out.head(200), use_container_width=True, height=350)
            st.download_button("Telecharger les predictions (CSV)",
                               out.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

# ==========================================================================
# ONGLET 3 — ENTRAINEMENT & EVALUATION
# ==========================================================================
with tab_train:
    st.subheader("Entrainer et evaluer sur votre dataset")
    st.markdown("Chargez le CSV ISCX-URL2016 (`All.csv`) ou tout CSV contenant "
                "les features numeriques et la colonne cible `URL_Type_obf_Type`.")

    up2 = st.file_uploader("CSV d'entrainement", type=["csv"], key="train")

    colA, colB, colC = st.columns(3)
    with colA:
        scenario = st.selectbox("Scenario",
                                ["Binaire (benign/malicious)", "Multi-classe (5 types)"])
    with colB:
        algo = st.selectbox("Algorithme",
                            ["Random Forest", "XGBoost"] if HAS_XGB else ["Random Forest"])
    with colC:
        test_size = st.slider("Part du test (%)", 10, 40, 20) / 100

    if up2 is not None and st.button("Lancer l'entrainement", type="primary"):
        df = pd.read_csv(up2).drop_duplicates().reset_index(drop=True)
        if TARGET not in df.columns:
            st.error(f"Colonne cible '{TARGET}' introuvable dans le CSV.")
        else:
            y_raw = df[TARGET].astype(str)
            X = clean_features(df.drop(columns=[TARGET]))
            binary = scenario.startswith("Binaire")

            if binary:
                y = (y_raw.str.lower() != "benign").astype(int).values
                labels = ["benign", "malicious"]
            else:
                classes = sorted(y_raw.unique())
                cmap = {c: i for i, c in enumerate(classes)}
                y = y_raw.map(cmap).values
                labels = classes

            st.write("**Distribution des classes :**")
            st.bar_chart(y_raw.value_counts())

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, shuffle=True,
                random_state=RANDOM_STATE, stratify=y)

            with st.spinner(f"Entrainement ({algo})..."):
                t0 = time.time()
                if algo == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                 random_state=RANDOM_STATE)
                else:
                    clf = XGBClassifier(
                        n_estimators=300, max_depth=10, learning_rate=0.2,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss" if binary else "mlogloss",
                        n_jobs=-1, random_state=RANDOM_STATE)
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_te)
                dt = time.time() - t0

            cm = confusion_matrix(y_te, pred)
            avg = "binary" if binary else "macro"
            acc = accuracy_score(y_te, pred) * 100
            prec = precision_score(y_te, pred, average=avg, zero_division=0) * 100
            rec = recall_score(y_te, pred, average=avg, zero_division=0) * 100
            f1 = f1_score(y_te, pred, average=avg, zero_division=0) * 100
            fpr = fpr_from_cm(cm) * 100

            st.markdown(f"**Resultats — {algo} · {scenario}** *(entraine en {dt:.1f}s)*")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc:.2f} %")
            m2.metric("Precision", f"{prec:.2f} %")
            m3.metric("Recall", f"{rec:.2f} %")
            m4.metric("F1-score", f"{f1:.2f} %")
            m5.metric("FPR", f"{fpr:.2f} %")

            colcm, colimp = st.columns(2)
            with colcm:
                st.markdown("**Matrice de confusion**")
                st.pyplot(plot_cm(cm, labels, f"{algo} — {scenario}"))
            with colimp:
                if hasattr(clf, "feature_importances_"):
                    st.markdown("**Importance des features (top 12)**")
                    imp = pd.Series(clf.feature_importances_,
                                    index=X.columns).sort_values(ascending=False).head(12)
                    st.bar_chart(imp)

            st.markdown("**Rapport de classification detaille**")
            rep = classification_report(y_te, pred, target_names=labels,
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ==========================================================================
# ONGLET 4 — A PROPOS
# ==========================================================================
with tab_about:
    st.subheader("A propos")
    st.markdown("""
Cette application applique le **Machine Learning** a la detection d'URLs
malveillantes, sur le dataset **ISCX-URL2016** (Universite du Nouveau-Brunswick).

**Deux modes de fonctionnement :**

- **Verification d'une URL** : saisissez n'importe quelle URL et obtenez un
  verdict immediat. Le modele extrait 33 features lexicales (longueurs, ratios,
  mots suspects, presence d'IP, etc.) sans aucune requete reseau.

- **Prediction par lot** : chargez un CSV au format ISCX-URL2016 (79 features
  pre-calculees) pour analyser des milliers d'URLs d'un coup.

**Approche** inspiree de :

> Nana et al. (2024), *Characterization of Malicious URLs Using Machine
> Learning and Feature Engineering.*

**Scenarios :**
- Binaire : benign vs malicious.
- Multi-classe : benign / Defacement / phishing / malware / spam.

**Resultats de reference sur ISCX-URL2016 :**
- Binaire : XGBoost ~99,3 % accuracy, FPR ~1,0 %.
- Multi-classe : XGBoost ~97,5 % accuracy, FPR ~0,7 %.
""")
    st.markdown("---")
    st.markdown("Fichiers : `app.py`, `url_features.py`, `rf_pretrained.joblib`, `iscx_models.joblib`.")
    if not HAS_XGB:
        st.info("XGBoost non installe : `pip install xgboost` pour l'activer.")
