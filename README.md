# HELOC Credit Risk — Machine Learning Pipeline

Machine learning pipeline for predicting HELOC (Home Equity Line of Credit) applicant risk using the FICO HELOC dataset (10,459 applicants, 23 numeric credit-bureau predictors). The model classifies each applicant's `RiskPerformance` as **Good** or **Bad**: applicants predicted Good are routed to loan officers for review; applicants predicted Bad are denied with an explanation of the top contributing factors.

The final model — a random forest with a tuned decision threshold — is served by a companion Streamlit application (separate repository). This repository contains the full modeling pipeline that produced it.

## Business framing

A false positive (auto-rejecting a good applicant, ~$1,500 in lost lending value) costs roughly twice as much as a false negative (sending a bad applicant to manual review, ~$780 expected). The operating metric is therefore **F-beta with β = 0.72**, weighting precision on the Bad class about 1.9× recall, with Bad-class precision used as a validation check against metric gaming. Full cost analysis is in `report_condensed.docx`, Section I.

## Repository contents

| File | Description |
|---|---|
| `Project-Final.ipynb` | Main pipeline: EDA, preprocessing, model selection (high-performance track), SHAP analysis, and export of the final model artifacts. Reads `heloc_dataset_v1.csv`. |
| `FRL_optuna_uptodate.py` | Falling Rule List track: from-scratch FRL implementation (after Chen & Rudin 2018) with a 200-trial Optuna TPE hyperparameter search, 5-fold CV on AUC, fold-parallel. **Runtime up to ~20 minutes — not recommended to run casually.** |
| `FRL_threshold_tuning_2.py` | Refits the FRL with the optimal hyperparameters from the TPE search (manually inserted) and tunes the decision threshold on F₀.₇₂. |
| `heloc_dataset_v1.csv` | Raw input data (10,459 rows). |
| `df_clean_v3.csv` | Post-preprocessing dataframe as prepared for the models. |
| `df_clean_v3_for_FRL.csv` | Slightly transformed version of the cleaned data used as input by the two FRL scripts. |
| `final_model_files/` | Deployed artifacts consumed by the Streamlit app: trained random forest, scaler, feature order, model config (hyperparameters + threshold), and preprocessing info (imputation medians, missing-value codes, special-imputation rules). |
| `report_condensed.docx` | Full project report: cost analysis, EDA, model selection, FRL assessment, robustness and monitoring plan. |
| `ML_HELOC_Presentation.pptx` | Project presentation. |

## Pipeline overview

### Preprocessing (20 final features)

The dataset encodes missingness with sentinel codes: `-9` (no bureau record), `-8` (no usable trades/inquiries), and `-7` (condition not met — informative, not truly missing).

1. Drop the 588 rows (5.6%) where every predictor is sentinel-coded → 9,871 rows.
2. Convert remaining `-9`/`-8` to NaN and impute with column medians.
3. Impute `-7` in `MSinceMostRecentDelq` and `MSinceMostRecentInqexcl7days` as **column max + 1 standard deviation**, encoding "event never happened" as a very long time ago and preserving monotonic ordering for both linear and tree models.
4. Drop three multicollinear columns (|corr| > 0.8): `NumInqLast6Mexcl7days`, `NumTrades90Ever2DerogPubRec`, `NumTotalTrades`.

### Model selection (high-performance track)

70/30 stratified train/test split (seed 42); 5-fold cross-validated AUC within the training set for model comparison and grid-search tuning; threshold tuning on the held-out test set using F₀.₇₂.

- Baselines: logistic regression 0.7979 CV AUC, random forest 0.7951, XGBoost 0.7775, decision tree 0.6869.
- After grid search, logistic regression (0.7981) and random forest (0.7976) were nearly tied on CV AUC. The tie was broken on test-set threshold performance: LR at threshold 0.45 gave F₀.₇₂ = 0.7404 with Bad-precision 0.7115; **RF at threshold 0.55 gave F₀.₇₂ = 0.7423 with Bad-precision 0.7517**. Following the precision-first policy, the random forest was selected.

Note: the notebook's automated "Best Overall Model" printout refers only to the CV-AUC comparison (where LR narrowly led); the final selection of the random forest on precision grounds is documented in the report, Section III.B.

**Final model:** random forest, `n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=2`, rejection threshold **0.55**. Per-applicant explanations via SHAP (TreeExplainer), reporting each feature's contribution to the predicted probability of a Bad rating.

### Falling Rule List track (interpretable alternative)

Conducted separately from the rest of the models. The FRL algorithm was recreated from Chen & Rudin (2018), "An Optimization Approach to Learning Falling Rule Lists," extended to allow up to 3 conditions per rule and tuned with a Tree-structured Parzen Estimator instead of grid search. It deliberately uses an 80/20 split (vs. 70/30 for the main track) to give the rule list more training data.

Best hyperparameters: `max_rules=28, min_support_frac=0.0362, n_thresholds=20, max_coverage_frac=0.6239, prob_gap=0.0042`. At the selected threshold of 0.5: F₀.₇₂ = 0.7236, precision 0.72 — close to the random forest (0.7423 / 0.7517) but ultimately rejected because SHAP provides comparable per-applicant explanations without the FRL's rule-instability problem across refits. See report, Sections III.C–D.

## Reproducing

1. `Project-Final.ipynb` end-to-end reproduces preprocessing, model selection, SHAP, and regenerates `final_model_files/`. Requires `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`.
2. `python FRL_optuna_uptodate.py` reruns the FRL hyperparameter search (~20 min, multi-core; requires `optuna`, `joblib`, `tabulate`). `python FRL_threshold_tuning_2.py` refits the best FRL and tunes its threshold (results print to stdout; the FRL scripts write no files).

## Team

Ethan Pochna (falling rule list, cost analysis, F-beta + precision evaluation plan, preprocessing contributions), Haris (EDA and preprocessing), Elma (model training and evaluation), James (Streamlit application). Generative AI usage is disclosed in the report, Section V.
