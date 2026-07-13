

<h1 align="center"> Détection & classification d'URLs malveillantes par Machine Learning</h1>

<p align="center">
  Détection et classification d'URLs malveillantes avec <b>Random Forest</b> et <b>XGBoost</b>,
  sur le dataset de référence <b>ISCX-URL2016</b> (79 features).<br>
  Approche inspirée de <i>Nana et al. (2024), « Characterization of Malicious URLs Using Machine Learning and Feature Engineering »</i>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-006ACC">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Licence-MIT-green">
</p>

---

##  Table des matières

- [Aperçu](#-aperçu)
- [Résultats](#-résultats)
- [Captures d'écran](#-captures-décran)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Méthodologie](#-méthodologie)
- [Perspectives](#-perspectives)
- [Références](#-références)

---

##  Aperçu

Les URLs malveillantes sont l'un des principaux vecteurs d'attaque sur le web : phishing, diffusion de malware, défacement de sites, spam. Les listes noires classiques deviennent vite obsolètes et ne détectent pas les nouvelles URLs. Ce projet utilise le **Machine Learning** pour détecter et classifier automatiquement les URLs à partir de leurs caractéristiques structurelles.

Le projet couvre **deux scénarios** :

| Scénario | Description | Classes |
|----------|-------------|---------|
| **Détection binaire** | Une URL est-elle légitime ou dangereuse ? | `benign` / `malicious` |
| **Classification multi-classe** | Quel type de menace ? | `benign` / `Defacement` / `phishing` / `malware` / `spam` |

Deux algorithmes sont comparés : **Random Forest** et **XGBoost**. Au-delà de l'accuracy, on suit le **taux de faux positifs (FPR)** — métrique clé mise en avant par le papier de référence, car un bon détecteur doit générer le moins possible de fausses alertes sur des URLs légitimes.

---

##  Résultats

Résultats obtenus sur l'ensemble de test (20 % du dataset, 26 953 URLs après déduplication) :

### Détection binaire

| Modèle | Accuracy | Precision | Recall | F1-score | **FPR** |
|--------|:--------:|:---------:|:------:|:--------:|:-------:|
| Random Forest | 99.04 % | 99.43 % | 99.23 % | 99.33 % | 1.47 % |
| **XGBoost**  | **99.26 %** | **99.61 %** | **99.36 %** | **99.49 %** | **1.00 %** |

### Classification multi-classe (5 types)

| Modèle | Accuracy | Precision | Recall | F1-score | **FPR** |
|--------|:--------:|:---------:|:------:|:--------:|:-------:|
| Random Forest | 96.44 % | 96.53 % | 94.64 % | 95.49 % | 0.94 % |
| **XGBoost**  | **97.50 %** | **97.31 %** | **96.21 %** | **96.73 %** | **0.66 %** |

>  Ces résultats **dépassent** ceux du papier de référence (accuracy 97.8 %, FPR 1.13 %) et surclassent largement plusieurs travaux antérieurs dont le FPR se situe entre 8 % et 12 %.



---





---

##  Dataset

Le projet utilise **ISCX-URL2016** (Université du Nouveau-Brunswick), dataset de référence pour la détection d'URLs malveillantes, dans sa version **pré-featurisée** (`All.csv`).

- **26 953 URLs** (après suppression des doublons)
- **79 features numériques** : longueurs des composants, ratios entre parties de l'URL, comptages de symboles et de chiffres, entropies, nombre de tokens du domaine, etc.
- **5 classes** : benign, phishing, spam, Defacement, malware

| Classe | Nombre d'URLs |
|--------|:-------------:|
| benign | 7 464 |
| phishing | 7 359 |
| spam | 5 331 |
| Defacement | 5 068 |
| malware | 1 731 |

 **Téléchargement** : [ISCX-URL2016 — UNB](https://www.unb.ca/cic/datasets/url-2016.html)

> Le fichier `All.csv` n'est pas inclus dans le dépôt (taille). Place-le à la racine du projet avant de lancer l'entraînement.

---

##  Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<ton-utilisateur>/detection-urls-malveillantes.git
cd detection-urls-malveillantes

# 2. (Optionnel) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---
##  Utilisation

### Lancer l'application interactive

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`. Garde `app.py` et `iscx_models.joblib` dans le même dossier.

### Relancer l'entraînement complet

```bash
python train_iscx.py
```

Ce script charge `All.csv`, nettoie les données, entraîne RF et XGBoost sur les deux scénarios, et génère les matrices de confusion, l'importance des features et le fichier `results.json`.

### Explorer le notebook

```bash
jupyter notebook URL_Detection_ISCX_Notebook.ipynb
```

Le notebook est **déjà exécuté** : toutes les sorties, métriques et graphiques y sont intégrés.

---

##  Structure du projet

```
.
├── app.py                              # Application Streamlit (3 onglets)
├── train_iscx.py                       # Script d'entraînement autonome
├── URL_Detection_ISCX_Notebook.ipynb   # Notebook commenté et exécuté
├── iscx_models.joblib                  # Modèles pré-entraînés (binaire + multi-classe)
├── Rapport_Detection_URLs_ISCX.docx    # Rapport complet du projet
├── requirements.txt                    # Dépendances Python
├── results.json                        # Métriques exportées
├── assets/                             # Captures et figures
└── README.md
```

---

##  Méthodologie

1. **Chargement & nettoyage** — suppression des doublons, remplacement des valeurs infinies (issues de ratios) par NaN, puis imputation par la médiane de chaque colonne.
2. **Séparation train/test** — 80 % / 20 %, avec mélange (`shuffle`) et stratification sur les classes (`random_state = 5`).
3. **Entraînement** — Random Forest (200 arbres) et XGBoost (300 estimateurs, profondeur 10).
4. **Évaluation** — Accuracy, Precision, Recall, F1-score et FPR ; matrices de confusion et importance des features.

### Features les plus discriminantes

L'analyse d'importance (Random Forest) fait ressortir des indicateurs liés à la **structure du domaine** et aux **symboles**, ce qui rend le modèle explicable :


---

##  Perspectives

- Tester la robustesse sur des URLs récentes hors dataset.
- Ajouter des features *host-based* (WHOIS, âge du domaine, réputation).
- Évaluer le modèle sur des URLs issues d'un cyberspace local.
- Passer d'un IDS à un IPS (action automatique sur détection).

---

##  Références

1. **S. R. Nana, D. Bassolé, J. S. D. Ouattara, O. Sié** (2024). *Characterization of Malicious URLs Using Machine Learning and Feature Engineering.* InterSol 2023, LNICST 541, Springer.
2. **M. S. I. Mamun et al.** (2016). *Detecting Malicious URLs Using Lexical Analysis.* NSS 2016 — dataset **ISCX-URL2016**, University of New Brunswick.
3. **D. Sahoo, C. Liu, S. C. H. Hoi** (2017). *Malicious URL Detection using Machine Learning: A Survey.*

---

