# Imbalanced Levantine Arabic Dialect Identification

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-0BA5A0?logo=streamlit)](https://arabic-dialect-identification.streamlit.app/)
[![IJEECS 2026](https://img.shields.io/badge/IJEECS%202026-10.11591%2Fijeecs.v43.i1.pp259--270-2663EB)](https://doi.org/10.11591/ijeecs.v43.i1.pp259-270)
[![UGM Thesis](https://img.shields.io/badge/UGM%20ETD-M.Cs.%20Thesis%202026-0BA5A0)](https://etd.repository.ugm.ac.id/penelitian/detail/282346)
[![License: MIT](https://img.shields.io/badge/License-MIT-16A349.svg)](https://opensource.org/licenses/MIT)

Official research codebase, experimental notebooks, and evaluation artifacts for the **IJEECS 2026 journal article** (*Evaluating oversampling methods for imbalanced Arabic dialect identification*, Vol. 43, No. 1, pp. 259–270) and the **Universitas Gadjah Mada M.Cs. Thesis (2026)** (*Cluster and Density-Based Hybrid Oversampling Model for Handling Imbalanced Data in Arabic Dialect Classification*).

This project investigates **Levantine Arabic dialect classification** on the **Shami Dialect Corpus** (`N = 66,247` tweets across Syrian, Lebanese, Palestinian, and Jordanian dialects) in a sparse-dense **5,644-dimensional** feature space, combining cluster-purity synthesis (**ASTRA-SMOTE**), cosine density post-filtering (**SMOTE-RADIANT**), and sequential hybridization (**ASTRA-RADIANT**).

---

## Links & Interactive Demo

- **Live Streamlit Application**: https://arabic-dialect-identification.streamlit.app/
- **Refactored Demo Repository & Git LFS Model Checkpoints**: https://github.com/moel-ihsan/arabic-dialect-id-demo
- **Journal Publication (IJEECS, July 2026)**: https://doi.org/10.11591/ijeecs.v43.i1.pp259-270
- **UGM M.Cs. Thesis (ETD Repository)**: https://etd.repository.ugm.ac.id/penelitian/detail/282346

> **Note on Pretrained Models (`**/model/` & `**/*.pkl`)**: Serialized `.pkl` checkpoints and feature extractors are excluded from this research repository by `.gitignore` (`**/model/`, `**/*.pkl`). All 12 trained LightGBM and XGBoost checkpoints (spanning `Original`, `SMOTE`, `ASTRA-SMOTE`, `SMOTE-RADIANT`, `ASTRA-RADIANT`, and `Class Weight`) along with the `shami_feature_extractor.pkl` pipeline are hosted via Git LFS in [`arabic-dialect-id-demo`](https://github.com/moel-ihsan/arabic-dialect-id-demo). Running the notebooks locally generates per-experiment `.pkl` checkpoints under `artifacts/<composition>/model/` (including `model_lightgbm_astra_radiant.pkl` and `model_xgboost_astra_radiant.pkl`).

---

## Evaluated Imbalance Handling Scenarios (12 Setups)

All six scenarios are evaluated across leaf-wise **LightGBM** and level-wise **XGBoost** on a strictly leakage-free stratified **60 / 15 / 25** partition (`39,748` train / `9,937` validation / `16,562` held-out test tweets):

| Scenario | Notebook Key | Paradigm | Core Mechanism |
| :--- | :--- | :--- | :--- |
| **Original** | `original` | Baseline | Natural imbalanced training distribution (`57.0%` SY vs. `10.6%` JO) |
| **SMOTE** | `smote` | Unconstrained Generative | Linear `k`-NN interpolation (`k = 5`, `+29,630` synthetic samples) |
| **ASTRA-SMOTE** | `knsmote` | Preventive Cluster-Purity | Restricts synthesis to safe MiniBatchKMeans clusters (`59/128` > 50% purity) |
| **SMOTE-RADIANT** | `dbscansmote` | Corrective Density-Filtered | Purges noisy synthetic outliers via cosine DBSCAN (`5,336` dropped; `18.0%`) |
| **ASTRA-RADIANT** | `kndbsmote` | Sequential Hybrid | Chains preventive ASTRA-SMOTE with corrective RADIANT (`3,526` dropped; `11.9%`) |
| **Class Weight** | `balanced_weighted` | Cost-Sensitive | Inverse-frequency weighting (`SY=1.00`, `LB=3.49`, `PS=3.55`, `JO=5.38`) |

---

## Key Held-Out Test Results (`N = 16,562`)

| Architecture | Scenario | Accuracy | Macro-Prec | Macro-Rec | Macro-F1 | JO Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| LightGBM | Original | 0.8963 | 0.8627 | 0.8439 | 0.8526 | 0.7520 |
| LightGBM | SMOTE | 0.8961 | 0.8584 | 0.8463 | 0.8518 | 0.7503 |
| LightGBM | ASTRA-SMOTE | 0.8959 | 0.8603 | 0.8443 | 0.8517 | 0.7503 |
| **LightGBM** | **SMOTE-RADIANT** | **0.8974** | **0.8620** | **0.8469** | **0.8539** | 0.7543 |
| LightGBM | ASTRA-RADIANT | 0.8970 | 0.8625 | 0.8448 | 0.8529 | 0.7491 |
| LightGBM | Class Weight | 0.8931 | 0.8431 | 0.8578 | 0.8502 | 0.7930 |
| XGBoost | Original | 0.8960 | 0.8620 | 0.8429 | 0.8517 | 0.7469 |
| XGBoost | SMOTE | 0.8964 | 0.8602 | 0.8454 | 0.8524 | 0.7537 |
| XGBoost | ASTRA-SMOTE | 0.8956 | 0.8604 | 0.8430 | 0.8511 | 0.7486 |
| XGBoost | SMOTE-RADIANT | 0.8964 | 0.8618 | 0.8445 | 0.8525 | 0.7497 |
| **XGBoost** | **ASTRA-RADIANT** | **0.8977** | **0.8640** | **0.8464** | **0.8546** | **0.7548** |
| XGBoost | Class Weight | 0.8934 | 0.8427 | 0.8587 | 0.8504 | 0.7948 |

### Highlights
- **Architecture-Dependent Hybridization**: Sequential **ASTRA-RADIANT** reaches the highest overall score on level-wise **XGBoost** (`0.8546` Macro-F1, `89.77%` accuracy), whereas density-filtered **SMOTE-RADIANT** leads on leaf-wise **LightGBM** (`0.8539` Macro-F1, `89.74%` accuracy).
- **Spatial Intrusion Reduction**: **ASTRA-RADIANT** reduces Jordanian synthetic inter-class intrusion from `16.54%` (standard SMOTE) to `1.05%` and minimizes global cluster overlap (`0.9731`).
- **Bootstrap Validation**: `N = 1,000` paired two-sided Wilcoxon signed-rank tests confirm statistically significant gains (*p* < 0.001) over unconstrained baselines and Class Weighting.

---

## Repository Structure & Execution Order

To reproduce the leakage-free 5,644-dimensional feature matrix, resampling runs, and analysis artifacts (`artifacts/`), execute the notebooks in order:

### 1. Preprocessing, Oversampling Core & Tuning
- `0. feature_engineering.ipynb` — Arabic cleaning, normalization, `stopwords.txt` filtering, 12-block (`5,644`-dim) sparse-dense fusion, and out-of-fold LR/SVM stacking
- `0. hybrid_oversampling.ipynb` — `ASTRA-SMOTE`, `SMOTE-RADIANT`, and sequential `ASTRA-RADIANT` implementations
- `0. utils_ml.ipynb` — shared training, evaluation, and logging helpers
- `1.5. joint_search.ipynb` — joint search for target sampling volumes and hyperparameters

### 2. Model Training Notebooks (LightGBM & XGBoost)
- **Original Baseline**: `1. lightgbm_original.ipynb`, `1. xgboost_original.ipynb`
- **SMOTE**: `2. lightgbm_smote.ipynb`, `2. xgboost_smote.ipynb`
- **ASTRA-SMOTE**: `3A. lightgbm_knsmote.ipynb`, `3A. xgboost_knsmote.ipynb`
- **SMOTE-RADIANT**: `3B. lightgbm_dbscansmote.ipynb`, `3B. xgboost_dbscansmote.ipynb`
- **Class Weight**: `4. lightgbm_balanced_weighted.ipynb`, `4. xgboost_balanced_weighted.ipynb`
- **ASTRA-RADIANT**: `5. lightgbm_kndbsmote.ipynb`, `5. xgboost_kndbsmote.ipynb`

### 3. Spatial, Statistical & Error Analysis Notebooks
- `z. analysis_cluster_purity.ipynb` — safe-zone cluster purity and inter-class intrusion analysis
- `z. analysis_effect_size.ipynb` — `N = 1,000` bootstrap paired Wilcoxon tests and effect sizes
- `z. analysis_entropy_margin_conf.ipynb` — confidence, margin, and Shannon entropy boundary audits
- `z. analysis_token_exclusivity_error.ipynb` — dialect-exclusive token collapse analysis

### 4. Interactive Demo Application
The `app-streamlit/` directory preserves the initial research prototype. The redesigned, production interactive dashboard and Git LFS model artifacts are maintained in [`moel-ihsan/arabic-dialect-id-demo`](https://github.com/moel-ihsan/arabic-dialect-id-demo) and deployed at https://arabic-dialect-identification.streamlit.app/.

---

## Environment Setup & Dataset

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This study uses the public **Shami Dialect Corpus** (Kwaik et al., 2018, LREC). Due to dataset size and redistribution terms, raw corpus files are placed locally under `dataset/` prior to running `0. feature_engineering.ipynb`.

---

## Citation

```bibtex
@article{ahmad2026evaluating,
  title   = {Evaluating oversampling methods for imbalanced Arabic dialect identification},
  author  = {Ahmad, Maulana Ihsan and Musdholifah, Aina and Nurwidyantoro, Arif},
  journal = {Indonesian Journal of Electrical Engineering and Computer Science (IJEECS)},
  volume  = {43},
  number  = {1},
  pages   = {259--270},
  year    = {2026},
  month   = {July},
  issn    = {2502-4752},
  doi     = {10.11591/ijeecs.v43.i1.pp259-270}
}

@mastersthesis{ahmad2026hybrid,
  title  = {Cluster and Density-Based Hybrid Oversampling Model for Handling Imbalanced Data in Arabic Dialect Classification},
  author = {Ahmad, Maulana Ihsan},
  school = {Department of Computer Science and Electronics, Universitas Gadjah Mada},
  year   = {2026},
  url    = {https://etd.repository.ugm.ac.id/penelitian/detail/282346}
}
```

---

## Authors & Affiliation

- **Maulana Ihsan Ahmad** — Lead Researcher & M.Cs. Student, Department of Computer Science and Electronics, Universitas Gadjah Mada (`maulanaihsanahmad@mail.ugm.ac.id`)
- **Aina Musdholifah, Ph.D.** — Co-Author & Thesis Supervisor, Intelligent Systems Laboratory, Universitas Gadjah Mada (`aina_m@ugm.ac.id`)
- **Arif Nurwidyantoro, Ph.D.** — Corresponding Author & Co-Supervisor, Software and Data Engineering Laboratory, Universitas Gadjah Mada (`arifn@ugm.ac.id`)
