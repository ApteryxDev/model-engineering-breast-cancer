from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Data for Task 1.csv"

df = pd.read_csv(CSV_PATH)

# Ignore the blank trailing column in the supplied CSV.
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

# Exclude the record ID. Treat malignant (M) as the positive class.
X = df.drop(columns=["id", "diagnosis"])
y = (df["diagnosis"] == "M").astype(int)

# Keep a stratified test set untouched for the final evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

models = {
    "Dummy baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=42),
    ),
    "Shallow decision tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    ),
    "Random forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}

print(f"Training records: {len(X_train)}")
print(f"Test records held back for final evaluation: {len(X_test)}")
print("\nFive-fold cross-validation results on training data:")

for name, model in models.items():
    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
    )

    print(
        f"{name:22} "
        f"F1: {results['test_f1'].mean():.3f} "
        f"(± {results['test_f1'].std():.3f}) | "
        f"Precision: {results['test_precision'].mean():.3f} | "
        f"Recall: {results['test_recall'].mean():.3f}"
    )
