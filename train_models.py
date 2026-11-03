"""
Assignment 2 - M.Tech AIML (BITS Pilani)
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset

5 classification models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier (k=5)
4. Naive Bayes Classifier (GaussianNB)
5. Support Vector Machine (SVC with probability=True)

Used Random state =42 for all models and train_test_split to have the standard set of data for all runs.

Metric to evaluate model:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Saves trained models, scaler, metrics, and held-out test_data.csv.
"""

#imports

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)


# Function to load and preprocess the data
def load_and_preprocess_data(filepath="data.csv"):
    df = pd.read_csv(filepath)
    print(f"Loaded raw dataset shape: {df.shape}")

    # Drop non-predictive ID column and empty trailing column if present
    drop_cols = [col for col in ['id', 'Unnamed: 32'] if col in df.columns]
    df = df.drop(columns=drop_cols)

    # Separate features and target
    target_col = 'diagnosis'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Target: M (Malignant) -> 1, B (Benign) -> 0
    y_raw = df[target_col]
    y = y_raw.map({'M': 1, 'B': 0}).astype(int)

    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]

    print(f"Features count: {len(feature_cols)}, Samples: {len(df)}")
    print(f"Class distribution: Benign (0)={sum(y==0)}, Malignant (1)={sum(y==1)}")

    return df, X, y, feature_cols


def main():
    # 1. Load data
    df, X, y, feature_cols = load_and_preprocess_data("data.csv")

    # 2. Stratified train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Save test_data.csv for submission and testing
    # Keeping the original diagnosis representation ('M' and 'B') so test file is realistic
    test_df = X_test.copy()
    test_df['diagnosis'] = y_test.map({1: 'M', 0: 'B'})
    test_df.to_csv("test_data.csv", index=False)
    print("Saved 'test_data.csv' successfully.")

    # 4. Standard Scaler for feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Define Models
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "use_scaled": True
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
            "use_scaled": False
        },
        "KNN": {
            "model": KNeighborsClassifier(n_neighbors=5, weights='distance'),
            "use_scaled": True
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "use_scaled": True
        },
        "Support Vector Machine": {
            "model": SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
            "use_scaled": True
        }
    }

    # Create model output directory
    os.makedirs("model", exist_ok=True)

    results = []

    print()
    print("MODEL TRAINING & EVALUATION ON TEST SET")
    print()

    trained_models = {}

    for name, config in models.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        clf.fit(X_tr, y_train)
        trained_models[name] = clf

        # Predictions
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "ML Model Name": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

    # Display Comparison Table
    results_df = pd.DataFrame(results)
    print("\nComparison Table:")
    print(results_df.to_string(index=False))

    # 6. Save artifacts
    joblib.dump(scaler, "model/scaler.joblib")
    with open("model/features.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open("model/metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save individual models
    model_filenames = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "KNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "Support Vector Machine": "model/svm.joblib"
    }

    for name, filename in model_filenames.items():
        joblib.dump(trained_models[name], filename)
        print(f"Saved {name} -> {filename}")


    print("\nAll model artifacts saved successfully in 'model/' directory!")


if __name__ == "__main__":
    main()
