import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── date range  ────────────────────────────────────────────────
START_DATE = "2011-03-30"
END_DATE   = "2020-12-31"

# ── loan parameters  ──────────────────────────────────────────
CREDIT_SPREAD = 0.015   # s = 150 bps fixed spread
TRADING_DAYS  = 252

# ── source data files (root data/ folder) ────────────────────────────────────
DATA_DIR      = Path(__file__).resolve().parent / "data"
STOCKS_FILE   = DATA_DIR / "Daily_stocks_return_2010-2020.xlsx"
SP500_FILE    = DATA_DIR / "daily_SP500_return_2010-2025.xlsx"
EURUSD_FILE   = DATA_DIR / "Daily_EUR-USD_rate_2010-2025.csv"
TREASURY_FILE = DATA_DIR / "Daily_3M-Treasury-Rate_2010-2020.xlsx"

# ── output files ─────────────────────────────────────────────────────────────
RETURNS_CSV = DATA_DIR / "returns.csv"
PLOTS_DIR   = DATA_DIR / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STOCK_COLS = ["IBM", "MSFT", "MTOR"]


# ── data loading ─────────────────────────────────────────────────────────────

def load_stocks():
    """Daily total return series for IBM, MSFT, MTOR (decimal form).

    Source is a CRSP long-format file with columns:
      PERMNO | Header CUSIP -8 Characters | Ticker | PERMCO |
      Daily Calendar Date | Daily Total Return
    Pivoted to wide format: date × ticker.
    """
    df = pd.read_excel(STOCKS_FILE)
    df["Daily Calendar Date"] = pd.to_datetime(
        df["Daily Calendar Date"], format="%Y-%m-%d"
    )
    wide = df.pivot_table(
        index="Daily Calendar Date",
        columns="Ticker",
        values="Daily Total Return",
        aggfunc="first",
    )
    wide.index.name = "Date"
    wide.columns.name = None
    wide.columns = [c.upper().strip() for c in wide.columns]

    missing = [c for c in STOCK_COLS if c not in wide.columns]
    if missing:
        raise KeyError(
            f"Tickers {missing} not found in {STOCKS_FILE.name}. "
            f"Available: {list(wide.columns)}"
        )
    return wide[STOCK_COLS].astype(float)


def load_sp500():
    """Daily total return series for the S&P 500 (decimal form).

    Source is a CRSP file with columns:
      INDNO | YYYYMMDD key | Daily Calendar Date |
      Daily Total Return on Index | Index Base Date
    """
    df = pd.read_excel(SP500_FILE)
    df["Daily Calendar Date"] = pd.to_datetime(
        df["Daily Calendar Date"], format="%Y-%m-%d"
    )
    df = df.set_index("Daily Calendar Date")[["Daily Total Return on Index"]]
    df.index.name = "Date"
    return df.rename(columns={"Daily Total Return on Index": "SP500"}).astype(float)


def load_eurusd():
    """EUR/USD levels → simple daily returns: r_t = FX_t / FX_{t-1} - 1.

    """
    df = pd.read_csv(EURUSD_FILE)
    df["observation_date"] = pd.to_datetime(df["observation_date"], format="%Y-%m-%d")
    df = df.set_index("observation_date")[["DEXUSEU"]]
    df.index.name = "Date"

    levels = df["DEXUSEU"].astype(float)
    levels[levels <= 0] = np.nan
    ret = levels / levels.shift(1) - 1.0
    ret = ret.iloc[1:]   # drop leading NaN from differencing

    return ret.rename("EURUSD").to_frame()


def load_treasury():
    """3-month Treasury bill rate → daily loan return r_t = (y_t + s) / 252.

    Source is the 'Daily' sheet of the FRED Excel file, with columns
    'observation_date' and 'DTB3'. Rate is in annualised percentage points
    (e.g. 0.08 means 0.08 %), so we divide by 100 to get decimal form.
    """
    df = pd.read_excel(TREASURY_FILE, sheet_name="Daily")
    df["observation_date"] = pd.to_datetime(
        df["observation_date"], format="%Y-%m-%d"
    )
    df = df.set_index("observation_date")[["DTB3"]]
    df.index.name = "Date"

    rate_pct     = pd.to_numeric(df["DTB3"], errors="coerce")   # blank cells → NaN
    rate_decimal = rate_pct / 100.0
    loan_return  = (rate_decimal + CREDIT_SPREAD) / TRADING_DAYS

    return loan_return.rename("Loan").to_frame()


# ── cleaning ─────────────────────────────────────────────────────────────────

def fill_isolated_missing(df):
    """Fill NaN only when surrounded by two valid observations (local average).

    This conservative interpolation is applied column-by-column before merging.
    A gap of more than one consecutive missing day is left as NaN.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        is_na       = s.isna()
        prev_valid  = s.shift(1).notna()
        next_valid  = s.shift(-1).notna()
        isolated    = is_na & prev_valid & next_valid
        df.loc[isolated, col] = (s.shift(1) + s.shift(-1))[isolated] / 2.0
    return df


def synchronise(stocks, sp500, eurusd, treasury):
    """Merge all series, apply isolated-NaN interpolation, drop remaining NaN rows,
    and restrict to the assignment sample window."""

    n_isolated_filled = 0
    pieces = []
    for src in (stocks, sp500, eurusd, treasury):
        filled = fill_isolated_missing(src)
        n_isolated_filled += int(src.isna().sum().sum() - filled.isna().sum().sum())
        pieces.append(filled)

    df = pieces[0].join(pieces[1:], how="outer")

    n_missing_before = int(df.isna().sum().sum())
    df = df.dropna(how="any")

    # restrict to assignment sample
    df = df.loc[START_DATE:END_DATE]
    n_missing_after = int(df.isna().sum().sum())

    print(f"Isolated NaN observations filled (local average): {n_isolated_filled}")
    print(f"Missing values before dropping rows:  {n_missing_before}")
    print(f"Missing values after dropping rows:   {n_missing_after}")
    print(f"Synchronised sample: {df.index[0].date()} to {df.index[-1].date()}, "
          f"{len(df)} observations.")

    return df


# ── descriptive statistics ────────────────────────────────────────────────────
def print_return_stats(returns):
    pct = returns * 100.0   # convert to percentage points

    # Build table
    tbl = returns.copy()
    tbl = pct.describe().T[["mean", "std", "min", "max"]]

    tbl["p5"]     = pct.quantile(0.05)
    tbl["median"] = pct.quantile(0.50)
    tbl["p95"]    = pct.quantile(0.95)
    tbl["skew"]   = pct.skew()
    tbl["kurt"]   = pct.kurtosis()

    # Reorder columns exactly as requested
    tbl = tbl[["mean", "std", "min", "p5", "median", "p95", "max", "skew", "kurt"]]

    # Rename columns
    tbl.columns = ["Mean (%)", "SD (%)", "Min (%)", "P5 (%)",
                   "Median (%)", "P95 (%)", "Max (%)", "Skew.", "Kurt"]

    print("\nReturn summary:")
    print(tbl.round(3))


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_returns(returns):
    cols = list(returns.columns)

    fig, axes = plt.subplots(
        len(cols), 1, figsize=(10.5, 2.2 * len(cols)), sharex=True
    )
    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        ax.plot(returns.index, returns[c] * 100, linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(c, fontsize=10)
        ax.set_ylabel("Return (%)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_returns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved: 01_returns.png")


def plot_cumulative_returns(returns):
    cum = (1 + returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(10.5, 5))
    for c in cum.columns:
        ax.plot(cum.index, cum[c] * 100, linewidth=1.4, label=c)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Cumulative returns (rebased to 0)", fontsize=12)
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_cumulative_returns.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved: 02_cumulative_returns.png")


# ── main entry point ──────────────────────────────────────────────────────────

def load_or_build(force_rebuild=False):
    """Return a clean returns DataFrame.  Uses cached CSV when available."""

    if not force_rebuild and RETURNS_CSV.exists():
        returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)
        print(f"Loaded returns from cache ({len(returns)} rows).")
        print_return_stats(returns)
        return returns

    print(f"Building returns from source files ({START_DATE} to {END_DATE}).")
    stocks   = load_stocks()
    sp500    = load_sp500()
    eurusd   = load_eurusd()
    treasury = load_treasury()

    returns = synchronise(eurusd, treasury, stocks, sp500)

    returns.to_csv(RETURNS_CSV)
    print(f"Returns saved to {RETURNS_CSV.name}.")

    print_return_stats(returns)
    plot_returns(returns)
    plot_cumulative_returns(returns)

    return returns


if __name__ == "__main__":
    load_or_build(force_rebuild=True)
