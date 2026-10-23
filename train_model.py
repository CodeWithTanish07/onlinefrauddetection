
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
CONFUSION_PATH = MODEL_DIR / "confusion_matrix.csv"

print("Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset nahi mili: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if "Class" not in df.columns:
    raise ValueError("Dataset mein 'Class' column nahi mila.")

if df.isnull().any().any():
    raise ValueError("Dataset mein missing values hain. Pehle unhe inspect karo.")

X = df.drop(columns=["Class"])
y = df["Class"]

if set(y.unique()) != {0, 1}:
    raise ValueError("Class column mein 0 aur 1 labels expected hain.")

print(f"Dataset loaded: {df.shape}")
print("\nClass distribution:")
print(y.value_counts().sort_index())

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
print("Training complete!")

# ---------------- EVALUATION ----------------
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, predictions, labels=[0, 1])

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    output_dict=True,
    zero_division=0,
)

# ---------------- CURVE DATA ----------------
fpr, tpr, _ = roc_curve(y_test, probabilities)

pr_precision, pr_recall, _ = precision_recall_curve(
    y_test, probabilities
)

metrics = {
    "model_name": "Random Forest",
    "dataset_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "fraud_cases": int((y == 1).sum()),
    "legitimate_cases": int((y == 0).sum()),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "fraud_precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "fraud_recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "fraud_f1_score": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "average_precision": float(
        average_precision_score(y_test, probabilities)
    ),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "feature_columns": list(X.columns),

    # Actual held-out test-set data for dashboard curves
    "roc_curve": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    },
    "precision_recall_curve": {
        "precision": pr_precision.tolist(),
        "recall": pr_recall.tolist(),
    },
}

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

print(f"Average Precision: {metrics['average_precision']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

print("\nConfusion Matrix:")
print(cm)
print("Matrix order: [[TN, FP], [FN, TP]]")

# ---------------- SAVE ARTIFACTS ----------------
joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

pd.DataFrame(
    cm,
    index=["Actual Legitimate", "Actual Fraud"],
    columns=["Predicted Legitimate", "Predicted Fraud"],
).to_csv(CONFUSION_PATH)

print("\nAll files saved successfully!")
print(f"Model: {MODEL_PATH}")
print(f"Metrics: {METRICS_PATH}")
print(f"Confusion matrix: {CONFUSION_PATH}")
