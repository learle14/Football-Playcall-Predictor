"""
This file will display the data gathered by teh logistical regression and xgboost
"""



import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just save files
import matplotlib.pyplot as plt
 
FIGURES_DIR = "figures"
LR_RESULTS_PATH = "team_unpredictability.csv"
XGB_RESULTS_PATH = "team_unpredictability_xgb.csv"  # optional
 
 
def plot_unpredictability_bar(result, title, out_path):
    result_sorted = result.sort_values("unpredictability_score", ascending=True)
 
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = plt.cm.RdYlGn_r(
        (result_sorted["unpredictability_score"] - result_sorted["unpredictability_score"].min())
        / (result_sorted["unpredictability_score"].max() - result_sorted["unpredictability_score"].min())
    )
    ax.barh(result_sorted["team"], result_sorted["unpredictability_score"], color=colors)
    ax.set_xlabel("Unpredictability Score (0 = most predictable, 1 = least)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
 
 
def plot_entropy_vs_logloss(result, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(result["mean_entropy"], result["log_loss"], s=60, alpha=0.7)
 
    for _, row in result.iterrows():
        ax.annotate(
            row["team"],
            (row["mean_entropy"], row["log_loss"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
 
    ax.set_xlabel("Mean Predictive Entropy (higher = more situationally ambiguous)")
    ax.set_ylabel("Log Loss (higher = model more often confidently wrong)")
    ax.set_title(title)
    ax.axvline(result["mean_entropy"].median(), color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(result["log_loss"].median(), color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
 
 
def plot_model_comparison(lr_metrics, xgb_metrics, out_path):
    labels = ["Accuracy", "Log Loss"]
    lr_vals = [lr_metrics["accuracy"], lr_metrics["log_loss"]]
    xgb_vals = [xgb_metrics["accuracy"], xgb_metrics["log_loss"]]
 
    x = range(len(labels))
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([i - width / 2 for i in x], lr_vals, width, label="Logistic Regression")
    ax.bar([i + width / 2 for i in x], xgb_vals, width, label="XGBoost")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
 
 
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
 
    if not os.path.exists(LR_RESULTS_PATH):
        print(f"{LR_RESULTS_PATH} not found — run predictability.py first.")
        return
 
    lr_result = pd.read_csv(LR_RESULTS_PATH)
    plot_unpredictability_bar(
        lr_result,
        "Team Unpredictability (Logistic Regression)",
        os.path.join(FIGURES_DIR, "unpredictability_bar_lr.png"),
    )
    plot_entropy_vs_logloss(
        lr_result,
        "Entropy vs. Log Loss by Team (Logistic Regression)",
        os.path.join(FIGURES_DIR, "entropy_vs_logloss_lr.png"),
    )
 
    if os.path.exists(XGB_RESULTS_PATH):
        xgb_result = pd.read_csv(XGB_RESULTS_PATH)
        plot_unpredictability_bar(
            xgb_result,
            "Team Unpredictability (XGBoost)",
            os.path.join(FIGURES_DIR, "unpredictability_bar_xgb.png"),
        )
        plot_entropy_vs_logloss(
            xgb_result,
            "Entropy vs. Log Loss by Team (XGBoost)",
            os.path.join(FIGURES_DIR, "entropy_vs_logloss_xgb.png"),
        )
    else:
        print(f"({XGB_RESULTS_PATH} not found — skipping XGBoost team charts.)")
 
 
if __name__ == "__main__":
    main()