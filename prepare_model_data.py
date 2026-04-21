from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.bs import bs_call_vectorized
from utils.features import compute_greeks


FEATURE_COLUMNS = [
    "log_moneyness",
    "T",
    "sqrt_T",
    "sigma",
    "sigma_sqrt_T",
    "vix",
    "vol_ma",
    "delta",
    "gamma",
    "vega",
    "theta",
    "vix_ratio",
    "time_vol",
    "vega_scaled",
]

REQUIRED_OPTION_COLUMNS = [
    "trade_date",
    "expiry",
    "strike",
    "settle_price",
]

REQUIRED_SPOT_COLUMNS = [
    "Date",
    "close",
    "volume",
]

REQUIRED_VIX_COLUMNS = [
    "Date",
    "vix",
]


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _load_options_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    _require_columns(df, REQUIRED_OPTION_COLUMNS, "options data")
    return df


def _load_spot_data(path: str | Path) -> pd.DataFrame:
    spot = pd.read_csv(path)
    spot.columns = [col.strip() for col in spot.columns]
    _require_columns(spot, REQUIRED_SPOT_COLUMNS, "spot data")

    spot["Date"] = pd.to_datetime(
        spot["Date"], dayfirst=True, format="mixed", errors="coerce"
    )
    spot["close"] = pd.to_numeric(spot["close"], errors="coerce")
    spot["volume"] = pd.to_numeric(spot["volume"], errors="coerce")
    spot = spot.sort_values("Date").reset_index(drop=True)

    spot["returns"] = np.log(spot["close"] / spot["close"].shift(1))
    spot["sigma"] = spot["returns"].rolling(window=20).std() * np.sqrt(252)
    spot["vol_ma"] = spot["volume"].rolling(window=5).mean()

    return spot


def _load_vix_data(path: str | Path) -> pd.DataFrame:
    vix = pd.read_csv(path)
    vix.columns = [col.strip() for col in vix.columns]

    unnamed_cols = [col for col in vix.columns if col.lower().startswith("unnamed") or col == "H1"]
    if unnamed_cols:
        vix = vix.drop(columns=unnamed_cols)

    _require_columns(vix, REQUIRED_VIX_COLUMNS, "vix data")

    vix["Date"] = pd.to_datetime(vix["Date"], errors="coerce").dt.tz_localize(None)
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")

    return vix


def build_model_ready_dataset(
    options_path: str | Path,
    spot_path: str | Path,
    vix_path: str | Path,
    risk_free_rate: float = 0.067,
    dividend_yield: float = 0.01,
    min_price: float = 5.0,
    min_t: float = 0.05,
    max_t: float = 0.5,
    min_moneyness: float = -0.3,
    max_moneyness: float = 0.3,
) -> pd.DataFrame:
    options_df = _load_options_data(options_path)
    spot = _load_spot_data(spot_path)
    vix = _load_vix_data(vix_path)

    options_df["trade_date"] = pd.to_datetime(
        options_df["trade_date"], dayfirst=True, format="mixed", errors="coerce"
    )
    options_df["expiry"] = pd.to_datetime(
        options_df["expiry"], dayfirst=True, format="mixed", errors="coerce"
    )
    options_df["strike"] = pd.to_numeric(options_df["strike"], errors="coerce")
    options_df["settle_price"] = pd.to_numeric(
        options_df["settle_price"], errors="coerce"
    )

    sigma_map = spot.set_index("Date")["sigma"]
    spot_price_map = spot.set_index("Date")["close"]
    vol_ma_map = spot.set_index("Date")["vol_ma"]

    options_df["sigma"] = options_df["trade_date"].map(sigma_map)
    options_df["spot_price"] = options_df["trade_date"].map(spot_price_map)
    options_df["vol_ma"] = options_df["trade_date"].map(vol_ma_map)

    options_df = options_df.dropna(
        subset=["trade_date", "expiry", "strike", "settle_price", "sigma", "spot_price"]
    ).copy()

    options_df["T"] = (options_df["expiry"] - options_df["trade_date"]).dt.days / 365.0

    final_dataset = pd.DataFrame(
        {
            "Date": options_df["trade_date"],
            "S": options_df["spot_price"],
            "K": options_df["strike"],
            "T": options_df["T"],
            "r": risk_free_rate,
            "sigma": options_df["sigma"],
            "Market_Price": options_df["settle_price"],
            "vol_ma": options_df["vol_ma"],
        }
    )

    final_dataset["Date"] = pd.to_datetime(final_dataset["Date"]).dt.tz_localize(None)
    final_dataset = final_dataset.merge(vix, on="Date", how="left")
    final_dataset["vix"] = final_dataset["vix"].ffill()
    final_dataset["vol_ma"] = final_dataset["vol_ma"].ffill()

    numeric_cols = ["S", "K", "T", "r", "sigma", "Market_Price", "vix", "vol_ma"]
    for col in numeric_cols:
        final_dataset[col] = pd.to_numeric(final_dataset[col], errors="coerce")

    final_dataset["bs_price"] = bs_call_vectorized(
        final_dataset["S"].values,
        final_dataset["K"].values,
        final_dataset["T"].values,
        final_dataset["r"].values,
        final_dataset["sigma"].values,
        dividend_yield,
    )
    final_dataset["error"] = final_dataset["Market_Price"] - final_dataset["bs_price"]

    final_dataset["log_moneyness"] = np.log(final_dataset["S"] / final_dataset["K"])
    final_dataset["sqrt_T"] = np.sqrt(final_dataset["T"])
    final_dataset["sigma_sqrt_T"] = final_dataset["sigma"] * final_dataset["sqrt_T"]

    final_dataset = final_dataset[
        (final_dataset["Market_Price"] > min_price)
        & (final_dataset["bs_price"] > min_price)
        & (final_dataset["T"] > min_t)
        & (final_dataset["T"] < max_t)
        & (final_dataset["log_moneyness"] > min_moneyness)
        & (final_dataset["log_moneyness"] < max_moneyness)
    ].copy()

    final_dataset = compute_greeks(final_dataset)
    eps = 1e-6
    final_dataset["vix_ratio"] = final_dataset["sigma"] / (final_dataset["vix"] + eps)
    final_dataset["time_vol"] = final_dataset["sigma"] * np.sqrt(final_dataset["T"])
    final_dataset["vega_scaled"] = final_dataset["vega"] / (
        final_dataset["Market_Price"] + eps
    )

    final_dataset = (
        final_dataset.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    ordered_columns = [
        "Date",
        "S",
        "K",
        "T",
        "r",
        "sigma",
        "Market_Price",
        "vix",
        "vol_ma",
        "bs_price",
        "error",
        *FEATURE_COLUMNS,
    ]

    unique_ordered_columns = list(dict.fromkeys(ordered_columns))

    return final_dataset[unique_ordered_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a model-ready dataset from options_ohlcv-style raw options data."
    )
    parser.add_argument("--options", required=True, help="Path to raw options CSV.")
    parser.add_argument("--spot", required=True, help="Path to spot price CSV.")
    parser.add_argument("--vix", required=True, help="Path to VIX CSV.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for the model-ready dataset.",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.067)
    parser.add_argument("--dividend-yield", type=float, default=0.01)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-t", type=float, default=0.05)
    parser.add_argument("--max-t", type=float, default=0.5)
    parser.add_argument("--min-moneyness", type=float, default=-0.3)
    parser.add_argument("--max-moneyness", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_dataset = build_model_ready_dataset(
        options_path=args.options,
        spot_path=args.spot,
        vix_path=args.vix,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
        min_price=args.min_price,
        min_t=args.min_t,
        max_t=args.max_t,
        min_moneyness=args.min_moneyness,
        max_moneyness=args.max_moneyness,
    )

    final_dataset.to_csv(output_path, index=False)

    print(f"Saved model-ready dataset to: {output_path}")
    print(f"Rows: {len(final_dataset):,}")
    print(f"Columns: {len(final_dataset.columns)}")
    print("Feature columns:")
    print(", ".join(FEATURE_COLUMNS))


if __name__ == "__main__":
    main()
