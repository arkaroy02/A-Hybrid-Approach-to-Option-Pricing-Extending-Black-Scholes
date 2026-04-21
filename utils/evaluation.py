
import numpy as np
import pandas as pd

from utils.features import recompute_features
import torch
from utils.bs import bs_call_vectorized


def representative_sample(
    df,
    max_rows=25000,
    random_state=42,
    stratify_cols=("T", "log_moneyness", "vix"),
    n_bins=5,
):
    if max_rows is None or len(df) <= max_rows:
        return df.copy()

    sampled = df.copy()
    strat_cols = []

    for col in stratify_cols:
        if col not in sampled.columns:
            continue
        try:
            binned = pd.qcut(sampled[col], q=n_bins, duplicates="drop")
            strat_name = f"{col}_bin"
            sampled[strat_name] = binned.astype(str)
            strat_cols.append(strat_name)
        except ValueError:
            continue

    if not strat_cols:
        return sampled.sample(n=max_rows, random_state=random_state).sort_index()

    sampled["_strata"] = sampled[strat_cols].agg("|".join, axis=1)
    group_sizes = sampled["_strata"].value_counts()
    proportions = group_sizes / len(sampled)

    sampled_parts = []
    for strata, size in group_sizes.items():
        group = sampled[sampled["_strata"] == strata]
        target = max(1, int(round(proportions[strata] * max_rows)))
        target = min(target, len(group))
        sampled_parts.append(group.sample(n=target, random_state=random_state))

    result = pd.concat(sampled_parts, axis=0)

    if len(result) > max_rows:
        result = result.sample(n=max_rows, random_state=random_state)
    elif len(result) < max_rows:
        remaining = sampled.drop(index=result.index, errors="ignore")
        if not remaining.empty:
            extra_n = min(max_rows - len(result), len(remaining))
            extra = remaining.sample(n=extra_n, random_state=random_state)
            result = pd.concat([result, extra], axis=0)

    sort_col = "Date" if "Date" in result.columns else None
    if sort_col:
        return result.sort_values(sort_col).reset_index(drop=True)
    return result.reset_index(drop=True)


def get_predictions(model, df, features, scaler):
    model.eval()
    df = recompute_features(df.copy())

    # scale
    X = scaler.transform(df[features])
    X = torch.tensor(X, dtype=torch.float32)

    bs_price = bs_call_vectorized(
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


def stress_scenarios(model, df, features, scaler, max_rows=None, random_state=42):
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
        if max_rows is not None:
            df_s = representative_sample(df_s, max_rows=max_rows, random_state=random_state)
        hybrid, bs = get_predictions(model, df_s, features, scaler)

        results[name] = {
            "n_rows": len(df_s),
            "hybrid_mean": np.mean(hybrid),
            "hybrid_std": np.std(hybrid),
            "bs_mean": np.mean(bs),
            "bs_std": np.std(bs)
        }

    return results


def regime_test(model, df, features, scaler, max_rows=None, random_state=42):
    median_vix = df["vix"].median()

    low_regime = df[df["vix"] < median_vix]
    high_regime = df[df["vix"] >= median_vix]

    results = {}

    for name, subset in [("low_vix", low_regime), ("high_vix", high_regime)]:
        if max_rows is not None:
            subset = representative_sample(subset, max_rows=max_rows, random_state=random_state)
        hybrid, bs = get_predictions(model, subset, features, scaler)

        market = subset["Market_Price"].values

        rmse_h = np.sqrt(np.mean((hybrid - market)**2))
        rmse_bs = np.sqrt(np.mean((bs - market)**2))

        results[name] = {
            "n_rows": len(subset),
            "hybrid_rmse": rmse_h,
            "bs_rmse": rmse_bs
        }

    return results


def tail_risk_analysis(model, df, features, scaler, max_rows=None, random_state=42):
    if max_rows is not None:
        df = representative_sample(df, max_rows=max_rows, random_state=random_state)
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
