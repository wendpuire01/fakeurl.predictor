# Detection & classification d'URLs malveillantes — ISCX-URL2016

Projet de Securite Informatique — Machine Learning (Random Forest & XGBoost)
sur le dataset de reference **ISCX-URL2016** (79 features), d'apres Nana et al. (2024).

## Resultats obtenus

| Scenario | Meilleur modele | Accuracy | FPR |
|---|---|---|---|
| Detection binaire | XGBoost | 99,26 % | 1,00 % |
| Multi-classe (5 types) | XGBoost | 97,50 % | 0,66 % |

## Contenu

| Fichier | Role |
|---|---|
| `Rapport_Detection_URLs_ISCX.docx` | Rapport complet (methodo, resultats, figures, comparaison) |
| `URL_Detection_ISCX_Notebook.ipynb` | Notebook commente et **deja execute** |
| `app.py` | Application Streamlit interactive |
| `iscx_models.joblib` | Modeles pre-entraines (binaire + multi-classe) |
| `train_iscx.py` | Script d'entrainement autonome |
| `requirements.txt` | Dependances |

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Garde `app.py` et `iscx_models.joblib` dans le meme dossier.

**Onglets :**
1. **🔮 Predire** — charge un CSV au format ISCX, obtiens les predictions (binaire ou 5 classes) + export CSV.
2. **📊 Entrainer & evaluer** — reentraine RF/XGBoost sur ton CSV et affiche metriques, matrice de confusion, importance des features.
3. **ℹ️ A propos** — explication de l'approche.

## Relancer l'entrainement complet

```bash
python train_iscx.py    # attend le fichier All.csv dans /mnt/user-data/uploads
```
