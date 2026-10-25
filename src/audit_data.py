import csv
from collections import Counter
from pathlib import Path

# Find the project folder from this script's location.
project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "data" / "raw" / "Data for Task 1.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")

with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    original_columns = reader.fieldnames

    if not original_columns:
        raise ValueError("The CSV appears to have no header row.")

    blank_headers = [
        column for column in original_columns
        if not column or not column.strip()
    ]
    columns = [
        column for column in original_columns
        if column and column.strip()
    ]

    # Ignore the blank trailing header in the supplied CSV.
    reader.fieldnames = columns
    rows = list(reader)

if blank_headers:
    print(f"Ignored blank header columns: {len(blank_headers)}")

print(f"Rows: {len(rows)}")
print(f"Columns: {len(columns)}")
print(f"Column names: {columns}")

missing = {
    column: sum(
        row.get(column) is None or row[column].strip() == ""
        for row in rows
    )
    for column in columns
}
print("\nMissing values per column:")
for column, count in missing.items():
    if count:
        print(f"  {column}: {count}")
if not any(missing.values()):
    print("  None")

diagnoses = Counter(
    row["diagnosis"].strip()
    for row in rows
    if row.get("diagnosis")
)
print(f"\nDiagnosis counts: {dict(diagnoses)}")

ids = [row["id"].strip() for row in rows if row.get("id")]
duplicate_ids = sum(count - 1 for count in Counter(ids).values() if count > 1)
print(f"Duplicate IDs: {duplicate_ids}")

normalized_rows = [
    tuple((row.get(column) or "").strip() for column in columns)
    for row in rows
]
duplicate_rows = len(normalized_rows) - len(set(normalized_rows))
print(f"Duplicate rows: {duplicate_rows}")

feature_columns = [
    column for column in columns
    if column not in {"id", "diagnosis"}
]
print(f"\nModel features: {len(feature_columns)}")
print("ID included as a feature: No")
print(f"Diagnosis labels found: {sorted(diagnoses)}")
