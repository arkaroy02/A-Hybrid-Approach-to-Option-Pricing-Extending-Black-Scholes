
import numpy as np

from utils.features import recompute_features
import torch
from utils.bs import bs_call


def get_predictions(model, df, features, scaler):

    model.eval()
    df = recompute_features(df.copy())

    # scale
    X = scaler.transform(df[features])
    X = torch.tensor(X, dtype=torch.float32)

    # ✅ vectorize your scalar BS function
    bs_vec = np.vectorize(bs_call)

    bs_price = bs_vec(
        df["S"].values,
        df["K"].values,
        df["T"].values,
        df["r"].values,
        df["sigma"].values,
        0.01
    )

    bs_tensor = torch.tensor(bs_price, dtype=torch.float32)

    with torch.no_grad():
        hybrid, _, _ = model(X, bs_tensor)

    hybrid = hybrid.detach().cpu().numpy()

    return hybrid, bs_price

def stress_scenarios(model, df, features, scaler):
    scenarios = {
        "base": df.copy(),

        "low_vol": df.assign(sigma=df["sigma"] * 0.5),
        "high_vol": df.assign(sigma=df["sigma"] * 2),

        "deep_ITM": df.assign(K=df["K"] * 0.7),
        "deep_OTM": df.assign(K=df["K"] * 1.3),

        "near_expiry": df.assign(T=np.maximum(df["T"] * 0.1, 1e-4)),

        "market_crash": df.assign(
            sigma=df["sigma"] * 2,
            S=df["S"] * 0.8,
            T=np.maximum(df["T"] * 0.5, 1e-4)
        )
    }

    results = {}

    for name, df_s in scenarios.items():
        hybrid, bs = get_predictions(model, df_s, features, scaler)

        results[name] = {
            "hybrid_mean": np.mean(hybrid),
            "hybrid_std": np.std(hybrid),
            "bs_mean": np.mean(bs),
            "bs_std": np.std(bs)
        }

    return results


def regime_test(model, df, features, scaler):
    median_vix = df["vix"].median()

    low_regime = df[df["vix"] < median_vix]
    high_regime = df[df["vix"] >= median_vix]

    results = {}

    for name, subset in [("low_vix", low_regime), ("high_vix", high_regime)]:
        hybrid, bs = get_predictions(model, subset, features, scaler)

        market = subset["Market_Price"].values

        rmse_h = np.sqrt(np.mean((hybrid - market)**2))
        rmse_bs = np.sqrt(np.mean((bs - market)**2))

        results[name] = {
            "hybrid_rmse": rmse_h,
            "bs_rmse": rmse_bs
        }

    return results

def tail_risk_analysis(model, df, features, scaler):
    hybrid, bs = get_predictions(model, df, features, scaler)
    market = df["Market_Price"].values

    # relative error
    hybrid_err = (hybrid - market) / (market + 1e-6)
    bs_err = (bs - market) / (market + 1e-6)

    results = {
        "hybrid_95": np.percentile(np.abs(hybrid_err), 95),
        "hybrid_99": np.percentile(np.abs(hybrid_err), 99),
        "bs_95": np.percentile(np.abs(bs_err), 95),
        "bs_99": np.percentile(np.abs(bs_err), 99),
    }

    return results