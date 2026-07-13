# Detection d'URLs malveillantes — Interface Streamlit

Application web de detection et classification d'URLs malveillantes par
Machine Learning (Random Forest / XGBoost), inspiree de Nana et al. (2024).

## Contenu

| Fichier | Role |
|---|---|
| `app.py` | L'application Streamlit (3 onglets) |
| `url_features.py` | Extraction des 33 features lexicales |
| `rf_pretrained.joblib` | Modele Random Forest pre-entraine (~93 % accuracy) |
| `requirements.txt` | Dependances Python |

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur (http://localhost:8501).

## Les 3 onglets

1. **🔍 Analyser une URL** — colle une URL, obtiens un verdict (benigne /
   malveillante) avec la probabilite, les signaux d'alerte detectes et le
   detail des 33 features. Fonctionne immediatement grace au modele
   pre-entraine.

2. **📊 Entrainer & evaluer** — charge ton propre CSV (colonnes `url` +
   `label`/`type`/`class`), choisis l'algorithme, et obtiens les metriques
   (Accuracy, Precision, Recall, F1, FPR), la matrice de confusion et
   l'importance des features. Compatible avec le dataset **ISCX-URL2016 /
   Kaggle** (binaire ou multi-classe, detection automatique).

3. **ℹ️ A propos** — explication de l'approche et des metriques.

## Note sur le dataset

Le modele pre-entraine a ete construit sur des URLs **incluant leur chemin**
(ex. `site.com/page.html`). Pour de meilleurs resultats dans l'onglet 1,
saisis l'URL complete avec son chemin plutot qu'un domaine racine seul.
