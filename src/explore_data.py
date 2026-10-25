from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Data for Task 1.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# The supplied CSV has a blank final header, read by pandas as "Unnamed: ...".
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

required_columns = {"id", "diagnosis", "radius_mean", "concave points_mean", "area_worst"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
colors = {"B": "#258f83", "M": "#e76f51"}

# 1. Number of cases in each diagnosis group
counts = df["diagnosis"].value_counts().reindex(["B", "M"], fill_value=0)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    ["Benign", "Malignant"],
    [counts["B"], counts["M"]],
    color=[colors["B"], colors["M"]],
)
ax.bar_label(bars)
ax.set_title("Cases by diagnosis")
ax.set_ylabel("Number of cases")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "diagnosis_counts.png", dpi=160)
plt.close(fig)

# 2. Compare selected feature values between groups
selected_features = ["radius_mean", "concave points_mean", "area_worst"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, feature in zip(axes, selected_features):
    values = [
        df.loc[df["diagnosis"] == label, feature].dropna()
        for label in ["B", "M"]
    ]
    ax.boxplot(values)
    ax.set_xticks([1, 2], ["Benign", "Malignant"])
    ax.set_title(feature.replace("_", " "))
    ax.set_ylabel("Feature value")

fig.suptitle("Selected feature values by diagnosis")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "feature_comparisons.png", dpi=160)
plt.close(fig)

# 3. Show the relationship between two features
fig, ax = plt.subplots(figsize=(7, 5))
for label, name in [("B", "Benign"), ("M", "Malignant")]:
    group = df[df["diagnosis"] == label]
    ax.scatter(
        group["radius_mean"],
        group["concave points_mean"],
        label=name,
        color=colors[label],
        alpha=0.7,
    )

ax.set_title("Radius and concave points by diagnosis")
ax.set_xlabel("Mean radius")
ax.set_ylabel("Mean concave points")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "radius_vs_concave_points.png", dpi=160)
plt.close(fig)

# Rank individual features by their correlation with malignant diagnosis.
# This is descriptive; correlation does not show that a feature causes cancer.
feature_columns = [
    column for column in df.columns
    if column not in {"id", "diagnosis"}
]
malignant = (df["diagnosis"] == "M").astype(int)
correlations = (
    df[feature_columns]
    .corrwith(malignant)
    .sort_values(key=lambda values: values.abs(), ascending=False)
)

print("Saved charts to:", FIGURE_DIR)
print("\nTop 10 features associated with malignant diagnosis:")
print(correlations.head(10).to_string(float_format=lambda value: f"{value:.3f}"))
