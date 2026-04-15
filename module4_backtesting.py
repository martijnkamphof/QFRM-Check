import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path

from module1_data import load_or_build
from module2_portfolio import WEIGHTS
from module3_var_es import OOS_FORECASTS_CSV, ALPHA, METHODS

DATA_DIR  = Path(__file__).resolve().parent / "data"
PLOTS_DIR = DATA_DIR / "plots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = DATA_DIR / "backtest_results.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def _portfolio_loss_pct(returns):
    """Realised daily portfolio loss series in % ( L_t = −r_t^p )."""
    w = np.array([WEIGHTS[c] for c in returns.columns], dtype=float)
    return pd.Series(
        -(returns.values @ w) * 100.0,
        index=returns.index,
        name="Loss (%)",
    )


def load_oos_data(force_rebuild=False):
    """Return (var_forecasts, realized_losses) both in %, aligned on dates.

    var_forecasts : DataFrame, columns = METHODS, index = forecast dates
    realized_losses : Series, same index
    """
    var_df = pd.read_csv(OOS_FORECASTS_CSV, index_col=0, parse_dates=True)

    returns  = load_or_build(force_rebuild)
    loss_all = _portfolio_loss_pct(returns)

    # keep only dates present in both
    common = var_df.index.intersection(loss_all.index)
    return var_df.loc[common], loss_all.loc[common]


# ── Table 4: violations per year ──────────────────────────────────────────────

def violations_table(var_df, losses):
    """Build Table 4: VaR violation counts per calendar year per method.

    A violation on day t is: L_t > VaR_t  (both in %).
    Expected violations ≈ number of OOS days in that year × (1 − α).
    """
    hit = pd.DataFrame(index=var_df.index, dtype=bool)
    for method in METHODS:
        hit[method] = losses > var_df[method]

    # number of OOS trading days per year (for the 'Expected' column)
    days_per_year = hit.resample("YE").size()

    tbl = pd.DataFrame({"Expected": (days_per_year * (1.0 - ALPHA)).round(1)})
    tbl.index = tbl.index.year

    for method in METHODS:
        tbl[method] = hit[method].resample("YE").sum().values

    tbl.index.name = "Year"
    return tbl, hit


def print_violations_table(tbl):
    print(f"\nTable 4 — VaR violations per year (99% confidence level)")
    print(tbl.to_string())


# ── violation plots (Figure 2) ────────────────────────────────────────────────

def plot_violations(var_df, losses, hit, filename="07_violations.png"):
    """One panel per method: portfolio loss (grey), rolling VaR (blue),
    violation days (red dots).  Matches PDF Figure 2."""
    n = len(METHODS)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n), sharex=True)

    for ax, method in zip(axes, METHODS):
        viol_idx = hit.index[hit[method]]

        ax.plot(losses.index, losses, color="grey", linewidth=0.7, label="Loss")
        ax.plot(var_df.index, var_df[method], color="steelblue",
                linewidth=1.1, label="VaR")
        ax.scatter(viol_idx, losses.loc[viol_idx], color="red",
                   s=10, zorder=5, label="Violation")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"VaR Violations — {method}", fontsize=10)
        ax.set_ylabel("Loss (%)")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {filename}")


# ── ACF of violations (Figure 3) ─────────────────────────────────────────────

def plot_acf_violations(hit, max_lag=20, filename="08_acf_violations.png"):
    """Autocorrelation of violation indicators for Normal, Student-t, Historical."""
    subset = ["Normal", "Student-t", "Historical"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, method in zip(axes, subset):
        x = hit[method].astype(float).values
        x -= x.mean()
        n = len(x)
        acf = np.array([
            np.corrcoef(x[:n - lag], x[lag:])[0, 1] if lag > 0 else 1.0
            for lag in range(max_lag + 1)
        ])
        conf = 1.96 / np.sqrt(n)

        ax.bar(range(max_lag + 1), acf, width=0.6, color="steelblue", alpha=0.8)
        ax.axhline(conf,  color="blue", linestyle="--", linewidth=0.9)
        ax.axhline(-conf, color="blue", linestyle="--", linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(method, fontsize=10)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_xlim(-0.5, max_lag + 0.5)
        ax.grid(True, alpha=0.25)

    fig.suptitle("ACF of violation indicators (dashed = 95% bands)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {filename}")


# ── main entry point ──────────────────────────────────────────────────────────

def run_backtest(force_rebuild=False):
    """Compute Table 4 and all violation plots.

    Returns
    -------
    tbl : DataFrame — violations per year × method (Table 4)
    hit : DataFrame — daily boolean violation indicators
    """
    var_df, losses = load_oos_data(force_rebuild)

    tbl, hit = violations_table(var_df, losses)
    print_violations_table(tbl)
    tbl.to_csv(RESULTS_CSV)

    plot_violations(var_df, losses, hit)
    plot_acf_violations(hit)

    return tbl, hit


if __name__ == "__main__":
    tbl, hit = run_backtest()
