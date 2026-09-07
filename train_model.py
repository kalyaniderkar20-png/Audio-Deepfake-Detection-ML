import os
import sys
import pandas as pd
import numpy as np
import joblib

# Fix Windows terminal UTF-8 encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def train_pipeline():
    csv_path = "dataset/features.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found. Please run create_dataset.py first.")
        return False

    print("[DATA] Loading preprocessed feature dataset...")
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop(columns=["Label", "File"])
    y = df["Label"]

    feature_names = list(X.columns)

    # Train / Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    print(f"[SPLIT] Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build Multi-Ensemble Classifier (Random Forest + Extra Trees + Gradient Boosting)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )

    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.85,
        random_state=42
    )

    ensemble_model = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('gb', gb)],
        voting='soft'
    )

    print("[TRAIN] Training Multi-Ensemble ML Model (RF + ExtraTrees + GradientBoosting)...")
    ensemble_model.fit(X_train_scaled, y_train)

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_scaled_all = scaler.transform(X)
    cv_scores = cross_val_score(ensemble_model, X_scaled_all, y, cv=skf, scoring='accuracy')

    # Predictions & Evaluation
    y_pred = ensemble_model.predict(X_test_scaled)
    y_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 1.0
    conf_mat = confusion_matrix(y_test, y_pred)

    print("\n========================================================")
    print(f"[METRIC] Model Accuracy  : {acc * 100:.2f}%")
    print(f"[CV 5-Fold] Mean Score   : {np.mean(cv_scores) * 100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")
    print(f"[METRIC] Precision       : {prec * 100:.2f}%")
    print(f"[METRIC] Recall          : {rec * 100:.2f}%")
    print(f"[METRIC] F1 Score        : {f1 * 100:.2f}%")
    print(f"[METRIC] ROC-AUC Score   : {auc * 100:.2f}%")
    print("========================================================\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real Voice", "Deepfake Voice"] if len(np.unique(y)) > 1 else None))

    print("Confusion Matrix:")
    print(conf_mat)

    # Extract Random Forest feature importances for UI visualization
    rf.fit(X_train_scaled, y_train)
    importances = rf.feature_importances_

    # Save model artifacts
    os.makedirs("model", exist_ok=True)
    model_path = "model/audio_deepfake_model.pkl"
    scaler_path = "model/scaler.pkl"
    metrics_path = "model/metrics.pkl"

    joblib.dump(ensemble_model, model_path)
    joblib.dump(scaler, scaler_path)

    metrics_data = {
        "accuracy": acc,
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": conf_mat.tolist(),
        "feature_names": feature_names,
        "feature_importances": importances.tolist(),
        "train_count": len(X_train),
        "test_count": len(X_test)
    }

    joblib.dump(metrics_data, metrics_path)

    print(f"\n[SAVE] Model saved to: {model_path}")
    print(f"[SAVE] Scaler saved to: {scaler_path}")
    print(f"[SAVE] Metrics saved to: {metrics_path}")
    print("[SUCCESS] Training pipeline complete!")
    return True


if __name__ == "__main__":
    train_pipeline()
