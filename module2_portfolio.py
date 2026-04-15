import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path

from module1_data import load_or_build

DATA_DIR  = Path(__file__).resolve().parent / "data"
PLOTS_DIR = DATA_DIR / "plots"
LOSSES_CSV = DATA_DIR / "portfolio_losses.csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── portfolio specification (PDF section 2) ───────────────────────────────────
PORTFOLIO_VALUE = 1_000_000.0

WEIGHTS = {
    "IBM":   0.20,
    "MSFT":  0.20,
    "MTOR":  0.10,
    "SP500": 0.20,
    "EURUSD": 0.15,
    "Loan":  0.15,
}

# sanity check
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-10, "Portfolio weights must sum to 1."


# ── P&L computation ───────────────────────────────────────────────────────────

def compute_component_pnl(returns):
    """Dollar P&L for each portfolio component.

    All return series are already in decimal form:
      - IBM, MSFT, MTOR, SP500: daily total returns (from CRSP data)
      - EURUSD:  simple daily returns (FX_t / FX_{t-1} - 1)
      - Loan:    daily floating-rate return ((y_t + s) / 252)

    P&L_i = w_i * V * r_i
    """
    pnl = pd.DataFrame(index=returns.index)

    for asset in WEIGHTS:
        pnl[asset] = PORTFOLIO_VALUE * WEIGHTS[asset] * returns[asset]

    return pnl.dropna()


# ── descriptive statistics ────────────────────────────────────────────────────
def print_pnl_stats(pnl, losses):
    print("\nComponent P&L summary (USD):")

    comp = pnl.describe().T[["mean", "std", "min", "max"]]
    comp["p5"]     = pnl.quantile(0.05)
    comp["median"] = pnl.quantile(0.50)
    comp["p95"]    = pnl.quantile(0.95)
    comp["skew"]   = pnl.skew()
    comp["kurt"]   = pnl.kurtosis()

    comp = comp[["mean", "std", "min", "p5", "median", "p95", "max", "skew", "kurt"]]
    comp.columns = ["Mean", "SD", "Min", "P5", "Median", "P95", "Max", "Skew.", "Kurt"]

    print(comp.round(2))

    print("\nPortfolio loss summary (USD):")

    port = losses.describe()[["mean", "std", "min", "max"]].to_frame().T
    port["p5"]     = losses.quantile(0.05)
    port["median"] = losses.quantile(0.50)
    port["p95"]    = losses.quantile(0.95)
    port["skew"]   = losses.skew()
    port["kurt"]   = losses.kurtosis()

    port = port[["mean", "std", "min", "p5", "median", "p95", "max", "skew", "kurt"]]
    port.columns = ["Mean", "SD", "Min", "P5", "Median", "P95", "Max", "Skew.", "Kurt"]

    print(port.round(2))


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_pnl(pnl, portfolio_pnl):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    for c in pnl.columns:
        ax1.plot(pnl.index, pnl[c], linewidth=0.9, label=c)

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Component P&L", fontsize=12)
    ax1.set_ylabel("USD")
    ax1.legend(frameon=False, ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(portfolio_pnl.index, portfolio_pnl, linewidth=1.1,
             label="Portfolio P&L")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Total portfolio P&L", fontsize=11)
    ax2.set_ylabel("USD")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_portfolio_pnl.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved: 03_portfolio_pnl.png")


def plot_loss_distribution(losses, portfolio_pnl):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    mu    = losses.mean()
    sigma = losses.std(ddof=1)

    # time series of losses
    axes[0, 0].plot(losses.index, losses, linewidth=1.0)
    axes[0, 0].set_title("Daily losses (USD)", fontsize=11)
    axes[0, 0].set_ylabel("USD")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # loss histogram with normal overlay
    x = np.linspace(losses.min(), losses.max(), 300)
    axes[0, 1].hist(losses, bins=50, density=True, alpha=0.85)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), linewidth=1.3)
    axes[0, 1].set_title("Loss histogram + normal fit", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # QQ plot against normal
    std_loss = (losses - mu) / sigma
    emp_q    = np.sort(std_loss.values)
    probs    = (np.arange(1, len(emp_q) + 1) - 0.5) / len(emp_q)
    norm_q   = stats.norm.ppf(probs)

    lo = min(norm_q.min(), emp_q.min())
    hi = max(norm_q.max(), emp_q.max())
    axes[1, 0].scatter(norm_q, emp_q, s=5, alpha=0.7)
    axes[1, 0].plot([lo, hi], [lo, hi], linewidth=1.1)
    axes[1, 0].set_title("QQ plot (vs. normal)", fontsize=11)
    axes[1, 0].set_xlabel("Normal quantiles")
    axes[1, 0].set_ylabel("Empirical quantiles")
    axes[1, 0].grid(True, alpha=0.3)

    # cumulative P&L
    cum_pnl = portfolio_pnl.cumsum()
    axes[1, 1].plot(cum_pnl.index, cum_pnl, linewidth=1.0)
    axes[1, 1].set_title("Cumulative P&L (USD)", fontsize=11)
    axes[1, 1].set_ylabel("USD")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_loss_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved: 04_loss_distribution.png")


# ── main entry point ──────────────────────────────────────────────────────────

def build_portfolio(force_rebuild=False):
    """Return (pnl, portfolio_pnl, losses) DataFrames and save to CSV."""
    returns = load_or_build(force_rebuild)

    pnl           = compute_component_pnl(returns)
    portfolio_pnl = pnl.sum(axis=1).rename("Portfolio_PnL")
    losses        = (-portfolio_pnl).rename("Loss")

    out = pnl.copy()
    out["Portfolio_PnL"] = portfolio_pnl
    out["Loss"]          = losses
    out.to_csv(LOSSES_CSV)
    print(f"Portfolio losses saved to {LOSSES_CSV.name}.")

    print_pnl_stats(pnl, losses)
    plot_pnl(pnl, portfolio_pnl)
    plot_loss_distribution(losses, portfolio_pnl)

    return pnl, portfolio_pnl, losses


if __name__ == "__main__":
    build_portfolio(force_rebuild=False)
