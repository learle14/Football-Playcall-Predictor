"""
This file will train a baseline logistic regression model for pass/run classification
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)

from data_loader import load_pbp_data
from features import filter_plays, season_split, build_features

YEARS = list(range(2015, 2024))
TEST_SEASONS_START = 2022
MODEL_PATH = "model.joblib"

def main():
    pbp = load_pbp_data(YEARS)
    df = filter_plays(pbp)
    train_df, test_df = season_split(df, TEST_SEASONS_START)
    x_train, x_test, y_train, y_test, scaler, feature_columns = build_features(train_df, test_df)
    print("\nClass balance (train):")
    print(y_train.value_counts(normalize=True))

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["run","pass"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    coefs = pd.DataFrame({"feature": feature_columns, "coefficient": model.coef_[0]}).sort_values("coefficient", ascending=False)

    print("\nTop 10 features pushing toward PASS:")
    print(coefs.head(10).to_string(index=False))
    print("\nTop 10 features pushing toward RUN:")
    print(coefs.tail(10).to_string(index=False))

    # Save data to run re-run inference
    joblib.dump(
        {"model": model, "scaler": scaler, "feature_columns": feature_columns}, MODEL_PATH
    )
    print(f"\nSaved model + scaler + feature columns to {MODEL_PATH}")


    test_with_probs = test_df.loc[x_test.index].copy()
    test_with_probs["pred_proba_pass"] = y_proba
    test_with_probs.to_parquet("test_predictions.parquet")
    print("Saved test set + predicted probabilities to test_predictions.parquet")

if __name__ == "__main__":
    main()
