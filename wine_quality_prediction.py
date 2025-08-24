import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("WineQuality_Prediction.csv")

print("Dataset shape:", data.shape)
print(data.info())
print("\nMissing values:\n", data.isnull().sum())

# Quick stats
print("\nStatistical Summary:\n", data.describe())

# Heatmap correlation
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="mako", fmt=".1f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Split features & target
X = data.drop("quality", axis=1)
y = data["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

print("Train size:", X_train.shape, " Test size:", X_test.shape)

# Model training
model = RandomForestClassifier(random_state=5)
model.fit(X_train, y_train)

# Evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\nTraining Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
