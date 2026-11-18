# Predicting company bankruptcy

Can a company's financial ratios tell us whether it will go bankrupt? These scripts train and
compare classic machine-learning models on 43,405 records of Polish companies, each described by
64 financial ratios, from a first logistic regression up to gradient boosting — plus clustering
(K-means), KNN and a neural network on the same question.

The catch: only about 5 in 100 records are bankruptcies. A model that always answers "still
operating" is 95% accurate and useless. So the scripts judge models by how many bankruptcies they
catch, how many false alarms they raise, and how well they rank risky companies first (average
precision).

## Scripts

Run them in order. Each one prints its results and saves its figures to `figures/<script name>/`.

| Script | What it does |
| --- | --- |
| `01_explore_data.py` | Class balance by forecast horizon, missing values, skewed ratios, and what each ratio means |
| `02_regression.py` | Warm-up regression: predict return on assets from three ratios |
| `03_logistic_regression.py` | Logistic regression and the 95% accuracy trap; class weights |
| `04_first_model_comparison.py` | The first comparison of five models, plus a per-horizon check (keeps duplicates and uses `year`, so not comparable with 05-07) |
| `05_decision_tree.py` | Decision trees on two ratios: one question, then depths 1-5 chosen on validation data |
| `06_random_forest.py` | Ordinary vs class-weighted random forest, with an alert threshold chosen on validation data |
| `07_compare_models.py` | The final comparison, one change at a time: logistic regression, class weights, gaps and extremes, random forest, gradient boosting |
| `08_kmeans.py` | Clustering (Aula 08): group companies without the label; elbow and silhouette for k = 2-7, two scaling settings, stability across initializations |
| `09_knn.py` | KNN (Aula 12): a grid of 40 configurations (scale, K, weights, metric) chosen on validation, cutoff tuned on validation |
| `10_mlp.py` | Neural network (Aula 13): five MLP configurations trained to 1,200 epochs with snapshots at 1, 10, 50, 100, 200, 400, 800 and 1,200; checkpoint and cutoff chosen on validation |

`common.py` holds the shared loading, splitting and plotting helpers. `columns.py` lists the 64
ratios with their formulas.

## Run it

Python 3.13. Scripts 06, 07 and 09 take about two minutes each; 10 about twenty-five.

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe 07_compare_models.py
```

macOS or Linux:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python 07_compare_models.py
```

## Headline result

From `07_compare_models.py`. Every model faces the same final test: 8,601 records, including 417
bankruptcies.

| Model | Bankruptcies caught | Missed | False alarms | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Always "operating" | 0 | 417 | 0 | 95.2% |
| First logistic regression | 6 | 411 | 10 | 95.1% |
| Gradient boosting, cutoff 0.56 | 305 | 112 | 368 | 94.4% |

Gradient boosting ranked bankruptcies best on three training folds (average precision 0.720,
against 0.115 for the first logistic regression). Its alert cutoff was chosen on validation data
before the final test was opened. Accuracy barely moves, while caught bankruptcies go from 6 to
305. Its scores still need calibration before they can be read as real probabilities.

## New models, same final test (scripts 08-10)

The last three scripts follow the classroom lessons on methods the project had not touched yet.
Every configuration is chosen on validation data before the 8,601-record final test is opened.

- **Clustering (08).** K-means groups companies using the 64 ratios only; the bankruptcy label is
  read afterwards to describe the groups, never to build them. With plain standardization the
  extreme ratios dominate the distances: every k from 2 to 7 isolates one- to nine-record outlier
  groups, so those solutions are discarded. Shrinking the extremes first fixes it: k = 3 wins
  (silhouette 0.375) and the groups are stable across initializations (adjusted Rand index ~1.0).
  The group with liabilities above total assets goes bankrupt 12.9% of the time, against 3.4%
  overall — clustering finds risky profiles without any labels.
- **KNN (09).** Of the 40 configurations (2 scales x 5 values of K x 2 weights x 2 metrics), the
  standardized ones rank bankruptcies far better than the raw ratios (best validation average
  precision 0.152 against 0.090): the unit of measurement must not decide the neighbors. The
  winner (K = 9, uniform weights, Manhattan) catches 274 of 417 bankruptcies with 2,107 false
  alarms — slightly ahead of the weighted logistic regression (263 caught) — but every prediction
  costs a search through all training records, and the vote shares are not calibrated chances.
- **Neural network (10).** Following Aula 13, training is watched at checkpoints (epochs 1, 10,
  50, 100, 200, 400, 800 and 1,200): every snapshot of the network is scored on validation data,
  and the best checkpoint (fewest epochs on ties) is the one that faces the final test. The
  lesson stops at 400 epochs; the extra checkpoints show that the classroom cap was premature
  here — every configuration was still improving at 1,200. The MLP has no class weights, so the
  bankrupt records are repeated ten times in the training data (validation and test stay
  natural). Of the five configurations (20/64 units, one or two hidden layers, sigmoid or ReLU,
  strong or weak L2), ReLU with two hidden layers 20-10 wins (validation average precision 0.288
  at epoch 1,200), while the sigmoid setup of the lesson ranks last (0.194). The chosen network
  catches 285 of 417 bankruptcies with 1,181 false alarms (test average precision 0.243, balanced
  accuracy 0.770, ROC AUC 0.828, Brier 0.091) — clearly ahead of the weighted logistic
  regression in every metric, but still below the tree ensembles. Against the 400-epoch run,
  catching more bankruptcies (285 against 241) also raised false alarms (1,181 against 1,072);
  precision still improved slightly (0.184 to 0.194). As in script 07, the scores are softmax
  outputs, not measured probabilities.

## All models on the same final test

Every model below was built on the same deduplicated data and faces the same final test: 8,601
records, including 417 bankruptcies. Alert cutoffs were tuned on validation data; average
precision and the confusion counts do not depend on the cutoff. Clustering (08) does not predict,
so it is not listed.

| Model (script, cutoff) | Accuracy | Balanced accuracy | Precision | Recall | F1 | F2 | MCC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Always "operating" | 95.2% | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.048 |
| Decision tree, depth 1 (05) | 95.2% | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| First logistic regression (07) | 95.1% | 0.507 | 0.375 | 0.014 | 0.028 | 0.018 | 0.066 | 0.109 |
| Weighted logistic regression (09, 10) | 72.9% | 0.683 | 0.108 | 0.631 | 0.184 | 0.320 | 0.174 | 0.124 |
| KNN, K = 9 (09, cutoff 0.01) | 73.8% | 0.700 | 0.115 | 0.657 | 0.196 | 0.338 | 0.192 | 0.126 |
| MLP, two layers 20-10 (10, epoch 1,200, cutoff 0.42) | 84.7% | 0.770 | 0.194 | 0.683 | 0.303 | 0.455 | 0.308 | 0.243 |
| Random forest (06, cutoff 0.12) | 91.4% | 0.812 | 0.323 | 0.698 | 0.441 | 0.566 | 0.437 | 0.563 |
| Gradient boosting (07, cutoff 0.56) | 94.4% | 0.843 | 0.453 | 0.731 | 0.560 | 0.651 | 0.549 | 0.681 |

Accuracy alone would pick the useless model: the always-"operating" baseline wins on accuracy
and loses on every other metric. Balanced accuracy and MCC expose that, F2 rewards catching
bankruptcies, and average precision measures the ranking used to triage companies. The ranking
mirrors the lesson order: simple models with raw scores stay near the 95% accuracy trap, the
neural network lands in between, and the tree ensembles catch the most bankruptcies with the
fewest false alarms. Gradient boosting stays the best of all ten scripts.

### Confusion matrix counts

TP = caught bankruptcies, FP = false alarms, TN = quiet operating companies, FN = missed
bankruptcies.

| Model | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: |
| Always "operating" | 0 | 0 | 8,184 | 417 |
| Decision tree, depth 1 (05) | 0 | 0 | 8,184 | 417 |
| First logistic regression (07) | 6 | 10 | 8,174 | 411 |
| Weighted logistic regression (09, 10) | 263 | 2,173 | 6,011 | 154 |
| KNN, K = 9 (09) | 274 | 2,107 | 6,077 | 143 |
| MLP, two layers 20-10 (10) | 285 | 1,181 | 7,003 | 132 |
| Random forest (06) | 291 | 611 | 7,573 | 126 |
| Gradient boosting (07) | 305 | 368 | 7,816 | 112 |

## About the data

- 43,405 records: 64 financial ratios, `year` and `class` (1 = went bankrupt, 0 = still
  operating). The column names describe the published formulas; `columns.py` maps each one to
  the original `Attr1` to `Attr64`.
- `year` is not a calendar year. Groups 1 to 5 hold forecasts made 5 down to 1 years ahead.
- 401 records are exact duplicates. Scripts 05-07 remove them.
- There are no company IDs, so one company can appear in several horizon groups. Splits are by
  record, not by company.

## Source and license

Polish Companies Bankruptcy, [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
(DOI [10.24432/C5F600](https://doi.org/10.24432/C5F600)), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Only the column headers were renamed;
every value is unchanged.

Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features
generation in application to bankruptcy prediction. *Expert Systems with Applications*, 58, 93-101.
