# Import necessary models and libraries here

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Load the dataset

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# Split the dataset into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# Create a pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

pipe.fit(X_train, y_train)
print("Model training complete.")

y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix(Test Set)")
plt.show()

# Ensure models folder exists
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Save model pipeline
model_path = MODELS_DIR / "model.joblib"
joblib.dump(pipe, model_path)
print(f"Model saved to {model_path}")