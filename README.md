# Predicting company bankruptcy

Can a company's financial ratios tell us whether it will go bankrupt? These scripts train and
compare classic machine-learning models on 43,405 records of Polish companies, each described by
64 financial ratios, from a first logistic regression up to gradient boosting.

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

`common.py` holds the shared loading, splitting and plotting helpers. `columns.py` lists the 64
ratios with their formulas.

## Run it

Python 3.13. Scripts 06 and 07 take about two minutes each.

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
