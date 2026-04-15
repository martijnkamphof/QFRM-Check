import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

START_DATE = "2016-01-01"
END_DATE   = "2026-03-31"

EQUITY_TICKERS = ("AAPL", "MSFT", "ASML.AS", "^GSPC")
RATE_TICKERS   = ("^IRX",)
FX_TICKERS     = ("EURUSD=X",)

ALL_TICKERS = EQUITY_TICKERS + RATE_TICKERS + FX_TICKERS

MAX_ROW_MISSING_SHARE = 0.40
MAX_SINGLE_GAP_TO_FFILL = 1

DATA_DIR         = Path(__file__).parent / "data"
RAW_PRICES_CSV   = DATA_DIR / "raw_prices.csv"
CLEAN_PRICES_CSV = DATA_DIR / "clean_prices.csv"
RETURNS_CSV      = DATA_DIR / "returns.csv"
PLOTS_DIR        = DATA_DIR / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def print_return_stats(returns):
    print("\nReturn summary:")
    print(returns.describe().T[["mean", "std", "min", "max"]].round(4))

    print("\nReturn skewness:")
    print(returns.skew().round(4))

    print("\nReturn kurtosis:")
    print(returns.kurtosis().round(4))


def download_data():
    print(f"Downloading from {START_DATE} to {END_DATE}")

    df = yf.download(
        ALL_TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    prices = df["Close"][list(ALL_TICKERS)]
    prices.index = pd.to_datetime(prices.index)

    print(f"Downloaded {len(prices)} rows.")
    return prices.sort_index()


def synchronise(prices):
    prices = prices.dropna(how="all").copy()

    for col in EQUITY_TICKERS + FX_TICKERS:
        prices.loc[prices[col] <= 0, col] = np.nan

    n_missing_before = int(prices.isna().sum().sum())

    prices = prices.loc[prices.isna().mean(axis=1) <= MAX_ROW_MISSING_SHARE]
    prices = prices.ffill(limit=MAX_SINGLE_GAP_TO_FFILL)
    prices = prices.dropna(how="any")

    n_missing_after = int(prices.isna().sum().sum())

    print(f"Missing values before cleaning: {n_missing_before}")
    print(f"Missing values after cleaning:  {n_missing_after}")
    print(f"Cleaned sample has {len(prices)} rows.")

    return prices


def compute_returns(prices):
    returns = pd.DataFrame(index=prices.index[1:])

    for t in EQUITY_TICKERS:
        returns[t] = np.log(prices[t] / prices[t].shift(1)).iloc[1:]

    if "EURUSD=X" in prices.columns:
        returns["EURUSD"] = np.log(
            prices["EURUSD=X"] / prices["EURUSD=X"].shift(1)
        ).iloc[1:]

    for t in RATE_TICKERS:
        returns[t] = prices[t].diff().iloc[1:]

    returns = returns.dropna()

    print(f"Returns calculated, len: {len(returns)}")
    print_return_stats(returns)

    return returns


def plot_prices(prices):
    eq = list(EQUITY_TICKERS)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    norm = 100.0 * prices[eq].div(prices[eq].iloc[0], axis=1)

    for c in norm.columns:
        ax1.plot(norm.index, norm[c], linewidth=1.6, label=c)

    ax1.set_title("Normalized equity and index prices", fontsize=12)
    ax1.set_ylabel("Index (start = 100)")
    ax1.legend(frameon=False, ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(prices.index, prices["^IRX"], linewidth=1.6, label="^IRX")
    ax2.set_title("Interest rate series", fontsize=11)
    ax2.set_ylabel("Level")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_prices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_returns(returns):
    cols = list(returns.columns)

    fig, axes = plt.subplots(
        len(cols), 1, figsize=(10.5, 2.1 * len(cols)), sharex=True
    )

    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        ax.plot(returns.index, returns[c], linewidth=1.0)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(c, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_returns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_or_build(force_rebuild=False):
    if not force_rebuild and CLEAN_PRICES_CSV.exists() and RETURNS_CSV.exists():
        prices = pd.read_csv(CLEAN_PRICES_CSV, index_col=0, parse_dates=True)
        returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)

        print_return_stats(returns)
        return prices, returns

    raw = download_data()
    raw.to_csv(RAW_PRICES_CSV)

    prices = synchronise(raw)
    returns = compute_returns(prices)

    prices.to_csv(CLEAN_PRICES_CSV)
    returns.to_csv(RETURNS_CSV)

    plot_prices(prices)
    plot_returns(returns)

    return prices, returns


if __name__ == "__main__":
    load_or_build(force_rebuild=True)