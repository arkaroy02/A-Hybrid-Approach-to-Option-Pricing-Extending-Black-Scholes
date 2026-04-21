Hybrid Option Pricing Model

This project combines Black-Scholes pricing with learned corrections from historical market data to improve real-world option price estimation.

## Main Workflow

- `testing.ipynb`: main experiment notebook for modeling and evaluation
- `preprocess.ipynb`: data cleaning and feature engineering notebook
- `prepare_model_data.py`: reusable preprocessing pipeline for model-ready data
- `stress_testing.ipynb`: standalone stress/regime/tail evaluation notebook using saved artifacts
- `models/`: hybrid model definitions
- `utils/`: feature, evaluation, plotting, and pricing helpers

## Project Layout

```text
.
|-- testing.ipynb
|-- preprocess.ipynb
|-- models/
|-- utils/
|-- artifacts/
|-- data/
|   |-- raw/
|   |   |-- market/
|   |   `-- options/
|   |-- interim/
|   `-- processed/
|-- notebooks/
|   `-- archive/
|-- junk/
`-- outputs/
```

## Notes

- Keep new experiments in `testing.ipynb`.
- Run `preprocess.ipynb` when raw data changes or you want to rebuild processed datasets.
- Save trained artifacts from `testing.ipynb` before running `stress_testing.ipynb`.
- Older exploratory notebooks are kept in `notebooks/archive/` so the root stays focused.
- Retired notebooks and side datasets can be parked in `junk/` before permanent deletion.

## Setup

Install dependencies with:

```powershell
pip install -r requirements.txt
```

## Input Data

To prepare a model-ready dataset, you need three CSV inputs:

- Options data in `options_ohlcv` style with at least: `trade_date`, `expiry`, `strike`, `settle_price`
- Spot price data with at least: `Date`, `close`, `volume`
- VIX data with at least: `Date`, `vix`

Current example paths in this project:

- `data/interim/options_ohlcv.csv`
- `data/raw/market/spot_price.csv`
- `data/raw/market/VIX_data.csv`

## Using Preprocess Notebook

Open `preprocess.ipynb` and run the main build cell. It calls the shared function from `prepare_model_data.py` and creates a model-ready dataframe.

Example notebook setup:

```python
import importlib
import prepare_model_data

importlib.reload(prepare_model_data)
build_model_ready_dataset = prepare_model_data.build_model_ready_dataset

OPTIONS_FILE = "data/interim/options_ohlcv.csv"
SPOT_FILE = "data/raw/market/spot_price.csv"
VIX_FILE = "data/raw/market/VIX_data.csv"
OUTPUT_FILE = "data/processed/AsianPaints_Model_Data_2019_24.csv"

final_dataset = build_model_ready_dataset(
    options_path=OPTIONS_FILE,
    spot_path=SPOT_FILE,
    vix_path=VIX_FILE,
)

final_dataset.to_csv(OUTPUT_FILE, index=False)
```

`final_dataset` is the dataframe you can inspect inside the notebook before saving.

## Using The Function Directly

You can also import the function in any Python file or notebook:

```python
from prepare_model_data import build_model_ready_dataset

df = build_model_ready_dataset(
    options_path="data/interim/options_ohlcv.csv",
    spot_path="data/raw/market/spot_price.csv",
    vix_path="data/raw/market/VIX_data.csv",
)
```

Optional parameters you can override:

- `risk_free_rate`
- `dividend_yield`
- `min_price`
- `min_t`
- `max_t`
- `min_moneyness`
- `max_moneyness`

## Using The CLI

You can run the preprocessing pipeline without opening a notebook:

```powershell
python prepare_model_data.py `
  --options data\interim\options_ohlcv.csv `
  --spot data\raw\market\spot_price.csv `
  --vix data\raw\market\VIX_data.csv `
  --output data\processed\AsianPaints_Model_Data_ready.csv
```

## Notebook Flow

Recommended order:

1. Run `preprocess.ipynb` to create or refresh the processed dataset.
2. Run `testing.ipynb` to train the model and save `artifacts/trained_model_checkpoint.pt` plus `artifacts/trained_scaler.pkl`.
3. Run `stress_testing.ipynb` to load those saved artifacts and evaluate stress scenarios separately from training.

## Output Schema

The prepared dataset contains model-ready columns used by `testing.ipynb`:

- `Date`
- `S`
- `K`
- `T`
- `r`
- `sigma`
- `Market_Price`
- `vix`
- `vol_ma`
- `bs_price`
- `error`
- `log_moneyness`
- `sqrt_T`
- `sigma_sqrt_T`
- `delta`
- `gamma`
- `vega`
- `theta`
- `vix_ratio`
- `time_vol`
- `vega_scaled`
