#task1_iris.py#
import kagglehub, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import joblib

path = kagglehub.dataset_download("saurabh00007/iriscsv")
print("Downloaded to:", path)
csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
if not csvs:
    raise FileNotFoundError("No CSV found in the downloaded folder.")
csv_path = csvs[0]
print("Using CSV:", csv_path)
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]
print(df.head())

target_candidates = [c for c in df.columns if c in ["species","class","variety","target"]]
if not target_candidates:
    nonnum = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    target_candidates = nonnum[:1]
if not target_candidates:
    raise ValueError("Couldn't infer target column. Please check your CSV headers.")
target = target_candidates[0]
print("Target column:", target)
X = df.drop(columns=[target])
y = df[target]
X = X.apply(pd.to_numeric, errors="coerce")
df = pd.concat([X, y], axis=1).dropna()
X, y = df.drop(columns=[target]), df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)
preds = clf.predict(X_test_s)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(classification_report(y_test, preds))
ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.title("Iris — Confusion Matrix")
plt.tight_layout(); plt.show()
joblib.dump({"scaler": scaler, "model": clf, "feature_names": X.columns.tolist()}, "iris_logreg.joblib")
print("Saved model to iris_logreg.joblib")
example_scaled = X_test_s[:1]
print("Demo prediction:", clf.predict(example_scaled)[0])
