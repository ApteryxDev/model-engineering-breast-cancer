from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Data for Task 1.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

df = pd.read_csv(CSV_PATH)
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

X = df.drop(columns=["id", "diagnosis"])
y = (df["diagnosis"] == "M").astype(int)  # 1 = malignant

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42),
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

test_results = df.loc[X_test.index, ["id", "diagnosis"]].copy()
test_results["predicted"] = pd.Series(
    predictions, index=X_test.index
).map({0: "B", 1: "M"})
test_results["p_malignant"] = model.predict_proba(X_test)[:, 1]

errors = test_results[test_results["diagnosis"] != test_results["predicted"]]

print("\nMisclassified test cases:")
print(errors.to_string(index=False))
print(f"Test F1:        {f1_score(y_test, predictions):.3f}")
print(f"Test precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
print(f"Test recall:    {recall_score(y_test, predictions):.3f}")

# Rows are actual classes; columns are predicted classes.
# Order: benign (0), then malignant (1).
matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
print("\nConfusion matrix (rows = actual, columns = predicted):")
print("                 Predicted B   Predicted M")
print(f"Actual B            {matrix[0, 0]:>4}          {matrix[0, 1]:>4}")
print(f"Actual M            {matrix[1, 0]:>4}          {matrix[1, 1]:>4}")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 4))
image = ax.imshow(matrix, cmap="Blues")
ax.set_xticks([0, 1], ["Benign", "Malignant"])
ax.set_yticks([0, 1], ["Benign", "Malignant"])
ax.set_xlabel("Predicted diagnosis")
ax.set_ylabel("Actual diagnosis")
ax.set_title("Logistic regression: test-set confusion matrix")

for row in range(2):
    for column in range(2):
        ax.text(column, row, matrix[row, column], ha="center", va="center")

fig.colorbar(image, ax=ax)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "logistic_regression_confusion_matrix.png", dpi=160)
plt.close(fig)

print(f"\nSaved confusion matrix to {FIGURE_DIR}")

focus_features = [
    "concave points_worst",
    "perimeter_worst",
    "concave points_mean",
    "radius_worst",
    "area_worst",
]

error_details = df.loc[
    df["id"].isin(errors["id"]),
    ["id", "diagnosis", *focus_features],
].merge(
    errors[["id", "predicted", "p_malignant"]],
    on="id",
)

print("\nSelected feature values for misclassified cases:")
print(error_details.to_string(index=False))

training_data = df.loc[X_train.index]
print("\nTraining-set median values by actual diagnosis:")
print(
    training_data.groupby("diagnosis")[focus_features]
    .median()
    .round(3)
    .to_string()
)
