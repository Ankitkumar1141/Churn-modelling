# 🔄 Customer Churn Prediction: End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiment%20Registry-00A396.svg)](https://dagshub.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Champion%20Model-EB5424.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-4172B8.svg)](https://optuna.org/)
[![Status](https://img.shields.io/badge/Status-Ongoing%20%2F%20In%20Active%20Development-orange.svg)](#)

---

## 📌 Project Overview

Customer churn poses a significant challenge in customer retention and lifetime value optimization. This repository houses an **end-to-end Machine Learning pipeline** designed to predict customer churn with high recall and precision on realistic, imbalanced enterprise data.

The project transitions from exploratory multi-table relational analysis and experimental hypothesis testing to Bayesian hyperparameter tuning, model ensembling, MLflow experiment tracking, and production-ready inference pipeline serialization.

> **Project Status:** Ongoing / Active Development — Model development and experimental benchmarking are complete; modular `src/` pipeline, FastAPI serving, and deployment layers are under active implementation.

---

## 🚀 Key Highlights & Architectural Flow

- **Multi-Table Relational Ingestion:** Integrates across Demographic, Account, and Geography lookup sheets for 10,000 customer records.
- **Advanced Missing Data Strategy:** Comprehensive analysis of missingness mechanisms (36.17% missing in customer account balances) evaluating drop vs. statistical and KNN imputation strategies.
- **Class Imbalance Mitigation:** Systematically benchmarked 7+ balancing strategies (Random UnderSampling, SMOTE, ADASYN, SMOTE+Tomek, Class Weights) to solve the 3.91:1 imbalance ratio and boost minority class recall to **>75%**.
- **Systematic Model Benchmarking:** Cross-validated evaluation across Logistic Regression, Random Forest, LightGBM, XGBoost, and Stacking Classifiers.
- **Bayesian Optimization via Optuna:** Automated fine-grained hyperparameter search spaces across tree-based algorithms and ensembled architectures.
- **MLflow & DagsHub Experiment Tracking:** Centralized logging of metrics (Recall, ROC-AUC, F1, PR-AUC), parameter configurations, and model artifacts.
- **Production Artifact Serialization:** Leak-free `ImbPipeline` and `ColumnTransformer` bundling KNN imputation, scalers, encoders, and the champion model into `models/inference_pipeline.joblib`.

---

## 📂 Repository Structure

```
Churn-modelling/
│
├── data/
│   ├── raw/                        # Multi-sheet raw dataset (raw_data.xlsx)
│   ├── interim/                    # Joined and intermediate datasets
│   ├── cleaned/                    # Cleaned and schema-aligned data
│   └── processed/                  # Feature-engineered & train/test split data
│
├── notebooks/                      # Sequential exploratory & experimental notebooks
│   ├── Data_overview.ipynb         # 01: Multi-sheet schema inspection & data overview
│   ├── Data_Cleaning.ipynb         # 02: Relational joins, deduplication & type formatting
│   ├── eda_fe.ipynb                # 03: Exploratory data analysis & feature engineering
│   ├── drop_vs_impute.ipynb        # 04: Experiment 1: Drop NaNs vs. Median/KNN imputation
│   ├── Missing_indicator.ipynb     # 05: Experiment 2: Value indicator flag evaluation
│   ├── class_imbalance.ipynb       # 06: Imbalance mitigation benchmarking (7+ strategies)
│   ├── model_selection.ipynb       # 07: Algorithmic cross-validated baseline benchmarking
│   ├── RF_HP_Tuning.ipynb          # 08: Optuna hyperparameter tuning for Random Forest
│   ├── XGB_HP_Tuning.ipynb         # 09: Optuna hyperparameter tuning for XGBoost
│   ├── LGBM_HP_Tuning.ipynb        # 10: Optuna hyperparameter tuning for LightGBM
│   ├── Stacking_Classifier_HP_Tuning.ipynb  # 11: Stacking ensemble hyperparameter tuning
│   └── Final_Estimator.ipynb       # 12: Champion model evaluation, MLflow run & serialization
│
├── models/                         # Serialized models and preprocessing transformers
│   ├── inference_pipeline.joblib   # Complete end-to-end inference pipeline
│   ├── model.joblib                # Champion XGBoost model weights
│   ├── preprocessor.joblib         # Fitted ColumnTransformer (MinMax + OneHot)
│   ├── knn_imputer.joblib          # Fitted KNN Imputer for missing features
│   └── resampler.joblib            # Trained RandomUnderSampler configuration
│
├── src/                            # Modular production pipeline code (In Progress)
│   ├── data/                       # Ingestion & cleaning modules
│   ├── features/                   # Preprocessing & transformation pipelines
│   ├── models/                     # Train, evaluation & registry scripts
│   └── visualization/              # Diagnostic & performance plotting utilities
│
├── scripts/                        # Utility scripts & model promotion workflows
│   ├── data_clean_utils.py         # Helper utilities for data cleaning
│   └── promote_model_to_prod.py    # DagsHub / MLflow registry promotion logic
│
├── deploy/                         # Deployment configs and automation
├── app.py                          # FastAPI inference service endpoint (In Progress)
├── Dockerfile                      # Container definition for reproducible serving
├── appspec.yml                     # AWS CodeDeploy deployment specification
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## 🔬 Experimental Workflow & Notebook Sequence

| # | Notebook | Objective | Key Findings / Outcome |
|---|---|---|---|
| **01** | `Data_overview.ipynb` | Multi-sheet data profiling | Inspected 3 relational sheets (`Demographic`, `Location`, `Account`) across 10,000 customers. |
| **02** | `Data_Cleaning.ipynb` | Merge & clean relational tables | Joined tables on keys; dropped redundant IDs (`CustomerId`, `AccountId`, `Name`, `LocationId`). |
| **03** | `eda_fe.ipynb` | EDA & feature engineering | Discovered churn patterns across age, balance, activity, and geographic segments. |
| **04** | `drop_vs_impute.ipynb` | Missing value strategy | Evaluated 3,617 missing balances; established KNN Imputation as superior to dropping data. |
| **05** | `Missing_indicator.ipynb` | Missing indicator feature test | Assessed whether missingness itself carries predictive information for churn likelihood. |
| **06** | `class_imbalance.ipynb` | Imbalance handling benchmark | Benchmarked 7 strategies; Random UnderSampling maximized Churn Recall to **75.7%** (ROC-AUC **0.840**). |
| **07** | `model_selection.ipynb` | Baseline algorithmic bake-off | Tested LR, RF, XGBoost, LightGBM, KNN, and SVM; tree ensembles showed clear superiority. |
| **08** | `RF_HP_Tuning.ipynb` | Random Forest Optuna tuning | Tuned `n_estimators`, `max_depth`, `min_samples_split`, and `criterion`. |
| **09** | `XGB_HP_Tuning.ipynb` | XGBoost Optuna tuning | Optimized `learning_rate`, `colsample_bytree`, `subsample`, and regularization parameters. |
| **10** | `LGBM_HP_Tuning.ipynb` | LightGBM Optuna tuning | Explored leaf-wise tree parameters (`num_leaves`, `max_depth`, `reg_alpha`). |
| **11** | `Stacking_Classifier_HP_Tuning.ipynb` | Ensembling | Tuned a stacked model combining Random Forest, LightGBM, and XGBoost with a meta-learner. |
| **12** | `Final_Estimator.ipynb` | Champion model pipeline & export | Packaged champion XGBoost architecture with preprocessing and serialized deployment artifacts. |

---

## 🏆 Key Experimental Results

### Class Imbalance Benchmark (Target: Churned = 20.37%)
| Strategy | Recall (Churn) | F1-Score | Precision | ROC-AUC | Business Fit |
|---|:---:|:---:|:---:|:---:|---|
| **Random UnderSampling** 🥇 | **0.757** | **0.573** | 0.461 | **0.840** | **Highest churn detection (Retention focus)** |
| **ADASYN** | 0.550 | 0.568 | 0.586 | 0.836 | Balanced trade-off |
| **SMOTE + Tomek** | 0.533 | 0.564 | **0.598** | 0.831 | Precision focus (Lower false positive rate) |
| **Baseline (No Sampling)** | 0.445 | 0.560 | 0.204 | 0.838 | Misses majority of churners |

### Champion Model Architecture
- **Algorithm:** XGBoost Classifier (`XGBClassifier`)
- **Imbalance Treatment:** Random UnderSampling (`imblearn`)
- **Missing Value Handling:** KNN Imputation (`n_neighbors=5`)
- **Scaling & Encoding:** `MinMaxScaler` for continuous features (`Age`, `Salary`, `Tenure`, `Balance`) + `OneHotEncoder` for categoricals (`Gender`, `Geography`)
- **Artifact:** Serialized into `models/inference_pipeline.joblib`

---

## 🛠️ Tech Stack

- **Core & Data Processing:** Python 3.10+, Pandas, NumPy, Scipy, OpenPyXL
- **Machine Learning & Preprocessing:** Scikit-Learn, Imbalanced-Learn, XGBoost, LightGBM
- **Hyperparameter Optimization:** Optuna
- **Experiment Tracking & MLOps:** MLflow, DagsHub, DVC
- **Serving & Deployment:** FastAPI, Uvicorn, Pydantic, Docker
- **Visualization:** Matplotlib, Seaborn, Plotly, Missingno

---

## ⚡ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Ankitkumar1141/Churn-modelling.git
cd Churn-modelling
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv churn_venv
.\churn_venv\Scripts\activate

# macOS / Linux
python3 -m venv churn_venv
source churn_venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory to enable MLflow and DagsHub tracking:
```env
DAGSHUB_USER_TOKEN=your_dagshub_token_here
MLFLOW_TRACKING_URI=https://dagshub.com/Ankitkumar1141/Churn-modelling.mlflow
```

### 5. Run the Notebooks
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter lab
```
Execute the notebooks in sequence from `notebooks/Data_overview.ipynb` to `notebooks/Final_Estimator.ipynb`.

---

## 🔮 Production Roadmap

- [x] Comprehensive EDA, data cleaning, and feature engineering
- [x] Missing value imputation benchmarking (KNN vs. drop)
- [x] Class imbalance treatment evaluation & recall maximization
- [x] Multi-algorithm baseline comparison and Optuna hyperparameter tuning
- [x] Final model pipeline creation and artifact persistence
- [ ] Refactoring notebook workflows into modular `src/` modules (`train.py`, `data_cleaning.py`)
- [ ] DVC data and pipeline stage orchestration (`dvc.yaml`)
- [ ] REST API development via FastAPI (`app.py`) for real-time and batch predictions
- [ ] Containerized deployment with Docker and automated CI/CD pipeline via AWS CodeDeploy

---

## 👤 Author

- **Ankit Kumar**

