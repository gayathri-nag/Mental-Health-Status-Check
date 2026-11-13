#import pandas as pd
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#import joblib

# Generate synthetic data
#np.random.seed(42)
#data_size = 500

#X = np.random.randint(1, 6, size=(data_size, 6))  # 6 input features (scores from 1-5)
#y = np.random.choice([0, 1, 2], size=data_size)   # Mental health status: 0 (Low), 1 (Medium), 2 (High)

# Convert to DataFrame for handling missing values
#df = pd.DataFrame(X, columns=["anxiety", "stress", "anger", "sleep_issues", "mood_swings", "depression"])
#df["target"] = y

# Fill missing values (if any) with the median
#df.fillna(df.median(), inplace=True)

# Separate features and target after handling missing values
#X = df.drop("target", axis=1)
#y = df["target"]

# Train-test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
#model = RandomForestClassifier(n_estimators=100, random_state=42)
#model.fit(X_train, y_train)

# Save the model
#joblib.dump(model, "mental_health_model.pkl")
#print("✅ Model trained and saved as mental_health_model.pkl")

#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the existing dataset
df = pd.read_csv("mental_health_data.csv")

# Define input features
feature_columns = [
    "Anxiety",
    "Stress",
    "Anger",
    "Sleep Issues",
    "Mood Swings",
    "Depression"
]

target_column = "Mental_Health_Status"

# Separate features and target
X = df[feature_columns]
y = df[target_column]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Initialize Random Forest classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low Risk", "Medium Risk", "High Risk"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

# Save trained model
joblib.dump(model, "mental_health_model.pkl")

print("\nModel trained and saved successfully!")