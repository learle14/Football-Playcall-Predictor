"""
This file will analyze which teams had the most/least predictable offenses
using both logistical regresssion and xgboost.

"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
 
MIN_PLAYS = 100  # filter out teams/situations with too few plays to trust
 
 
def binary_entropy(p):
    """
    Shannon entropy of a Bernoulli(p) distribution, in bits.
    Max value is 1.0 at p=0.5 (most uncertain), drops to 0 at p=0 or p=1.
    """
    p = np.clip(p, 1e-9, 1 - 1e-9)  # avoid log(0)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
 
 
def compute_team_predictability(df, team_col="posteam", proba_col="pred_proba_pass"):
    """
    Returns a DataFrame with one row per team, ranked by unpredictability.
    """
    df = df.copy()
    df["true_label"] = (df["play_type"] == "pass").astype(int)
    df["entropy"] = binary_entropy(df[proba_col])
 
    rows = []
    for team, group in df.groupby(team_col):
        if len(group) < MIN_PLAYS:
            continue
 
        mean_entropy = group["entropy"].mean()
        team_log_loss = log_loss(
            group["true_label"], group[proba_col], labels=[0, 1]
        )
        team_accuracy = accuracy_score(
            group["true_label"], (group[proba_col] >= 0.5).astype(int)
        )
        pass_rate = group["true_label"].mean()
 
        rows.append(
            {
                "team": team,
                "n_plays": len(group),
                "pass_rate": pass_rate,
                "mean_entropy": mean_entropy,
                "log_loss": team_log_loss,
                "accuracy": team_accuracy,
            }
        )
 
    result = pd.DataFrame(rows)
 
    # Combine both signals into one ranking. Both are min-max normalized
    # so they're on the same scale before averaging.
    result["entropy_rank_score"] = (
        result["mean_entropy"] - result["mean_entropy"].min()
    ) / (result["mean_entropy"].max() - result["mean_entropy"].min())
    result["log_loss_rank_score"] = (
        result["log_loss"] - result["log_loss"].min()
    ) / (result["log_loss"].max() - result["log_loss"].min())
 
    result["unpredictability_score"] = (
        result["entropy_rank_score"] + result["log_loss_rank_score"]
    ) / 2
 
    return result.sort_values("unpredictability_score", ascending=False).reset_index(
        drop=True
    )
 
 
def main():
    # Usage: python predictability.py [path_to_predictions.parquet] [output_csv_name]
    # Defaults to the logistic regression output; pass test_predictions_xgb.parquet
    # (and optionally an output name) to score the XGBoost model instead.
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "test_predictions.parquet"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "team_unpredictability.csv"
 
    df = pd.read_parquet(pred_path)
    result = compute_team_predictability(df)
 
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
 
    print(f"\nTeams ranked by unpredictability (n={len(result)}, min {MIN_PLAYS} plays):\n")
    print(
        result[
            [
                "team",
                "n_plays",
                "pass_rate",
                "mean_entropy",
                "log_loss",
                "accuracy",
                "unpredictability_score",
            ]
        ].to_string(index=False)
    )
 
    print("\nMost unpredictable 5 teams:")
    print(result.head(5)["team"].tolist())
 
    print("\nMost predictable 5 teams:")
    print(result.tail(5)["team"].tolist())
 
    result.to_csv(output_path, index=False)
    print(f"\nSaved full results to {output_path}")
 
 
if __name__ == "__main__":
    main()