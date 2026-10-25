from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Data for Task 1.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

df = pd.read_csv(CSV_PATH)
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

X = df.drop(columns=["id", "diagnosis"])
y = (df["diagnosis"] == "M").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Logistic regression: standardized coefficients show direction and strength.
logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42),
)
logistic.fit(X_train, y_train)

predictions = pd.Series(
    logistic.predict(X_test),
    index=X_test.index,
)
probabilities = pd.Series(
    logistic.predict_proba(X_test)[:, 1],
    index=X_test.index,
)

scaler = logistic.named_steps["standardscaler"]
model_step = logistic.named_steps["logisticregression"]

scaled_test = scaler.transform(X_test)
contributions = pd.DataFrame(
    scaled_test * model_step.coef_[0],
    index=X_test.index,
    columns=X.columns,
)

misclassified_indices = X_test.index[predictions != y_test]

for index in misclassified_indices:
    actual = "M" if y_test.loc[index] == 1 else "B"
    predicted = "M" if predictions.loc[index] == 1 else "B"
    case_id = df.loc[index, "id"]

    row = contributions.loc[index]

    print(
        f"\nID {case_id}: actual={actual}, predicted={predicted}, "
        f"malignant probability={probabilities.loc[index]:.3f}"
    )
    print("Largest contributions toward malignant:")
    print(row.nlargest(3).to_string(float_format=lambda value: f"{value:.3f}"))
    print("Largest contributions toward benign:")
    print(row.nsmallest(3).to_string(float_format=lambda value: f"{value:.3f}"))

coefficients = pd.Series(
    logistic.named_steps["logisticregression"].coef_[0],
    index=X.columns,
)
top_coefficients = coefficients.loc[
    coefficients.abs().nlargest(10).index
].sort_values()

print("Logistic regression: 10 largest standardized coefficients")
print(top_coefficients.to_string(float_format=lambda value: f"{value:.3f}"))

# Random forest: permutation importance measures how much test F1 drops
# when one feature's values are shuffled.
forest = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)
forest.fit(X_train, y_train)

permutation = permutation_importance(
    forest,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=30,
    random_state=42,
)

forest_importance = pd.Series(
    permutation.importances_mean,
    index=X.columns,
).sort_values(ascending=False)

print("\nRandom forest: 10 highest permutation importances")
print(forest_importance.head(10).to_string(
    float_format=lambda value: f"{value:.4f}"
))

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Plot signed logistic coefficients.
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#e76f51" if value > 0 else "#258f83" for value in top_coefficients]
ax.barh(top_coefficients.index, top_coefficients.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic regression: largest standardized coefficients")
ax.set_xlabel("Coefficient (positive values raise predicted malignant odds)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "logistic_coefficients.png", dpi=160)
plt.close(fig)

# Plot random forest permutation importance.
top_forest = forest_importance.head(10).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_forest.index, top_forest.values, color="#457b9d")
ax.set_title("Random forest: permutation importance")
ax.set_xlabel("Mean decrease in test F1 when feature is shuffled")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "random_forest_permutation_importance.png", dpi=160)
plt.close(fig)

print(f"\nSaved importance charts to: {FIGURE_DIR}")

