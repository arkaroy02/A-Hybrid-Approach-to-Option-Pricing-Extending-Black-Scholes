import matplotlib.pyplot as plt
import numpy as np  


def plot_dashboard(df):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # 1. Actual vs Hybrid
    axes[0,0].scatter(df["Market_Price"], df["hybrid_price"], alpha=0.3)
    axes[0,0].plot([df["Market_Price"].min(), df["Market_Price"].max()],
                   [df["Market_Price"].min(), df["Market_Price"].max()])
    axes[0,0].set_title("Actual vs Hybrid")

    # 2. BS vs Hybrid
    axes[0,1].scatter(df["Market_Price"], df["bs_price"], alpha=0.3, label="BS")
    axes[0,1].scatter(df["Market_Price"], df["hybrid_price"], alpha=0.3, label="Hybrid")
    axes[0,1].legend()
    axes[0,1].set_title("BS vs Hybrid")

    # 3. Error Distribution
    axes[0,2].hist(df["bs_error"], bins=50, alpha=0.5, label="BS")
    axes[0,2].hist(df["hybrid_error"], bins=50, alpha=0.5, label="Hybrid")
    axes[0,2].legend()
    axes[0,2].set_title("Error Distribution")

    # 4. Error Asymmetry
    pos = df["bs_error"] > 0
    neg = df["bs_error"] < 0
    axes[1,0].scatter(df["bs_error"][pos], df["hybrid_error"][pos], alpha=0.3)
    axes[1,0].scatter(df["bs_error"][neg], df["hybrid_error"][neg], alpha=0.3)
    axes[1,0].axhline(0)
    axes[1,0].axvline(0)
    axes[1,0].set_title("Error Asymmetry")

    # 5. Error vs Moneyness
    axes[1,1].scatter(df["log_moneyness"], df["bs_error"], alpha=0.3, label="BS")
    axes[1,1].scatter(df["log_moneyness"], df["hybrid_error"], alpha=0.3, label="Hybrid")
    axes[1,1].legend()
    axes[1,1].set_title("Error vs Moneyness")

    # 6. Alpha Distribution
    axes[1,2].hist(df["alpha"], bins=50)
    axes[1,2].set_title("Alpha Distribution")

    # 7. Alpha vs Moneyness
    axes[2,0].scatter(df["log_moneyness"], df["alpha"], alpha=0.3)
    axes[2,0].set_title("Alpha vs Moneyness")

    # 8. ML Correction vs Error
    correction = df["ml_price"] / df["bs_price"]
    axes[2,1].scatter(correction, df["hybrid_error"], alpha=0.3)
    axes[2,1].axhline(0)
    axes[2,1].set_title("Correction vs Error")

    # 9. Absolute Error vs Moneyness
    abs_error = abs(df["hybrid_price"] - df["Market_Price"])
    axes[2,2].scatter(df["log_moneyness"], abs_error, alpha=0.3)
    axes[2,2].set_title("Abs Error vs Moneyness")

    plt.tight_layout()
    plt.show()