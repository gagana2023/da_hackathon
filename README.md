NAME:GAGANA S
ROLL: DA25C008
DA5401 Kaggle Challenge: LLM Response Evaluation Scoring
Overview
This repository contains the code, experiments, and report for the DA5401 Kaggle competition on LLM response evaluation scoring. The challenge required robust prediction of integer scores (0–10) for LLM responses evaluated under a variety of semantic metrics. The solution features advanced data engineering, ordinal regression with metric-aware priors, leakage-safe validation, and calibrated neural modeling.

Problem Statement
Objective: Predict human-like evaluation scores for LLM responses across dozens of metrics.

Inputs: Each sample incorporates metric name, system prompt, prompt, and expected response.

Outputs: A single integer score (0–10) per sample.

Data Engineering & EDA
Cleaned integer scores, rounded any non-standard values (9.5 → 10).

Analyzed class imbalance: 91% of scores are 9–10.

Created hash-based duplicate groups and quantified train/test overlap.

Constructed leakage-safe cross-validation splits using GroupKFold.

Feature Engineering
Gemma-2B-generated metric embeddings (768-dim).

One-hot encoding for top 40 metric names + “other” category.

Per-metric Bayesian priors (empirical class distributions, Laplace+shrinkage).

System prompt length as a proxy feature.

Model Architecture
Baseline: Logistic regression on metric embeddings + priors.

Primary Model: Ordinal Cumulative Neural Network

Dense layers (input: 810 features).

Monotonic cumulative thresholding for probability outputs.

Inverse frequency class weighting per fold.

Label smoothing (ε=0.02) for stability.

Hacks & Calibration
Duplicate fill: deterministic prediction for 15% of test set via combo_hash.

Temperature scaling: region-based temperatures optimize confidence calibration.

Probability tilt: shifts small probability mass from max class (10) to 8/9.

Ridge regressor: final residual correction using expected value and entropy of predicted probabilities.

Training & Validation
5-fold leakage-safe GroupKFold CV:

Fold	Valid MAE
1	0.7250
2	1.1140
3	1.6910
4	1.1180
5	1.3990
Mean	1.2094
OOF QWK: -0.0864

Submission & Results
Final OOF MAE: 1.2094

QWK: -0.0864

Test prediction distribution: majority in scores 6, 9, and 10.

15% of test predictions filled directly via train/test duplicates.

Reproducibility
All experiments run with fixed random seeds.

Training/validation code protected from data leakage by hash-based group splits.

Hyperparameters and blending coefficients tuned via grid search and cross-validation.

How to Run
Install required Python packages from requirements.txt.

Place dataset files in the expected locations.

Run solution.ipynb notebook to reproduce all results and generate submission files.

Submission files: submission_calibrated_shrunk_residual_safe.csv and submission_final.csv.
