"""
This file runs an xgboost run/pass classifier on the same features as train_model.py.
It also compares the results against the logistic regression baseline in train_model.py.
"""


import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix,
)
 
from data_loader import load_pbp_data
from features import filter_plays, season_split, build_features
 
YEARS = list(range(2015, 2024))
TEST_SEASON_START = 2022
MODEL_PATH = "model_xgb.joblib"
PRED_PATH = "test_predictions_xgb.parquet"
BASELINE_MODEL_PATH = "model.joblib"  # saved by model_train.py, used for comparison
 
 
def evaluate(y_test, y_pred, y_proba, label):
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    print(f"\n--- {label} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print(classification_report(y_test, y_pred, target_names=["run", "pass"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return acc, ll
 
 
def main():
    pbp = load_pbp_data(YEARS)
    df = filter_plays(pbp)
 
    train_df, test_df = season_split(df, TEST_SEASON_START)
    X_train, X_test, y_train, y_test, scaler, feature_columns = build_features(
        train_df, test_df
    )
 
    # XGBoost handles raw (unscaled) numeric features fine, and benefits
    # from an explicit class-imbalance weight since run plays are the
    # minority class (~41% in training data).
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
 
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
 
    xgb_acc, xgb_ll = evaluate(y_test, y_pred, y_proba, "XGBoost")
 
    # Feature importance (gain-based — how much each feature improves
    # splits on average, more informative than raw split counts)
    importances = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nTop 15 most important features:")
    print(importances.head(15).to_string(index=False))
 
    # Compare against the logistic regression baseline, if it was trained first
    try:
        baseline_bundle = joblib.load(BASELINE_MODEL_PATH)
        baseline_model = baseline_bundle["model"]
        baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
        baseline_pred = baseline_model.predict(X_test)
        lr_acc, lr_ll = evaluate(y_test, baseline_pred, baseline_proba, "Logistic Regression (baseline)")
 
        print("\n=== Comparison ===")
        print(f"{'Model':<25}{'Accuracy':<12}{'Log Loss':<12}")
        print(f"{'Logistic Regression':<25}{lr_acc:<12.4f}{lr_ll:<12.4f}")
        print(f"{'XGBoost':<25}{xgb_acc:<12.4f}{xgb_ll:<12.4f}")
    except FileNotFoundError:
        print(f"\n(No baseline found at {BASELINE_MODEL_PATH} — run train.py first for a comparison.)")
 
    joblib.dump(
        {"model": model, "scaler": scaler, "feature_columns": feature_columns},
        MODEL_PATH,
    )
    print(f"\nSaved model to {MODEL_PATH}")
 
    test_with_probs = test_df.loc[X_test.index].copy()
    test_with_probs["pred_proba_pass"] = y_proba
    test_with_probs.to_parquet(PRED_PATH)
    print(f"Saved test set + predicted probabilities to {PRED_PATH}")
 
 
if __name__ == "__main__":
    main()