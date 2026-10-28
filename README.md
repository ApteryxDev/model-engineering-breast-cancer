# Breast Cancer Diagnosis Classification

This repository contains the Python analysis for the DLBDSME01 Model Engineering case study. It compares a dummy baseline, logistic regression, a depth-3 decision tree, and a random forest on the Wisconsin Diagnostic Breast Cancer dataset. The target is benign (B) versus malignant (M). The patient identifier is retained only for error review and is excluded from the model features.

## Data

Place the supplied CSV at `data/raw/Data for Task 1.csv`. The raw file is excluded from Git; do not commit it unless the course and dataset terms explicitly permit redistribution.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab
```

## Run the analysis

Run the audit and analysis scripts from the repository root:

```bash
python src/audit_data.py
python src/explore_data.py
python src/compare_models.py
python src/evaluate_final.py
python src/feature_importance.py
```

The notebook at `notebooks/generate_report_figures.ipynb` is the canonical source for the report figures. Open it from the repository root, restart the kernel, and run all cells from top to bottom. It saves the final figures in `reports/figures/`. Some scripts also save exploratory figures with overlapping filenames; run the notebook last to refresh the report versions.

The comparison uses a stratified 80/20 train-test split with random seed 42 and shuffled, stratified five-fold cross-validation with random seed 42. Logistic-regression scaling is fitted inside the pipeline. The random forest uses 300 trees. The held-out test set is reserved for the final evaluation; training-fold out-of-fold probabilities are used for threshold analysis.

## Tests

Run the repository housekeeping test with:

```bash
python -m unittest discover -s tests -v
```

The tests check that the notebook and comparison script use the same CV splitter and forest size. The notebook also exports metrics, CV scores, and test errors under `results/`.
