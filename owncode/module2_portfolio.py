import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path

from module1_data import load_or_build

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = DATA_DIR / "plots"
LOSSES_CSV = DATA_DIR / "portfolio_losses.csv"

PORTFOLIO_VALUE = 1_000_000.0
TRADING_DAYS = 250
CREDIT_SPREAD = 0.015
LOAN_DURATION = 0.25

WEIGHTS = {
    "AAPL": 0.25,
    "MSFT": 0.25,
    "ASML.AS": 0.20,
    "^GSPC": 0.20,
    "^IRX": 0.10,
}

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def print_pnl_stats(pnl, losses):
    print("\nComponent PnL summary:")
    print(pnl.describe().T[["mean", "std", "min", "max"]].round(4))

    print("\nComponent skewness:")
    print(pnl.skew().round(4))

    print("\nComponent kurtosis:")
    print(pnl.kurtosis().round(4))

    print("\nPortfolio loss summary:")
    print(losses.describe()[["mean", "std", "min", "max"]].round(4))
    print("skew     ", round(losses.skew(), 4))
    print("kurtosis ", round(losses.kurtosis(), 4))


def compute_component_pnl(prices, returns):
    pnl = pd.DataFrame(index=returns.index)

    pnl["AAPL"] = PORTFOLIO_VALUE * WEIGHTS["AAPL"] * (np.exp(returns["AAPL"]) - 1.0)
    pnl["MSFT"] = PORTFOLIO_VALUE * WEIGHTS["MSFT"] * (np.exp(returns["MSFT"]) - 1.0)
    pnl["^GSPC"] = PORTFOLIO_VALUE * WEIGHTS["^GSPC"] * (np.exp(returns["^GSPC"]) - 1.0)
    asml_usd_simple_return = np.exp(returns["ASML.AS"] + returns["EURUSD"]) - 1.0
    pnl["ASML.AS"] = PORTFOLIO_VALUE * WEIGHTS["ASML.AS"] * asml_usd_simple_return
    loan_notional = PORTFOLIO_VALUE * WEIGHTS["^IRX"]

    lagged_rate = prices["^IRX"].shift(1).reindex(returns.index) / 100.0
    carry_pnl = loan_notional * (lagged_rate + CREDIT_SPREAD) / TRADING_DAYS
    mtm_pnl = -loan_notional * LOAN_DURATION * (returns["^IRX"] / 100.0)

    pnl["^IRX"] = carry_pnl + mtm_pnl
    pnl = pnl.dropna()

    return pnl


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

    ax2.plot(portfolio_pnl.index, portfolio_pnl, linewidth=1.1, label="Portfolio P&L")
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


def plot_loss_distribution(losses, portfolio_pnl):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    mu = losses.mean()
    sigma = losses.std(ddof=1)

    axes[0, 0].plot(losses.index, losses, linewidth=1.0)
    axes[0, 0].set_title("Daily losses", fontsize=11)
    axes[0, 0].set_ylabel("USD")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[0, 1].hist(losses, bins=50, density=True, alpha=0.85)
    x = np.linspace(losses.min(), losses.max(), 300)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), linewidth=1.3)
    axes[0, 1].set_title("Loss histogram + normal fit", fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    std_loss = (losses - mu) / sigma
    emp_q = np.sort(std_loss.values)
    probs = (np.arange(1, len(emp_q) + 1) - 0.5) / len(emp_q)
    norm_q = stats.norm.ppf(probs)

    axes[1, 0].scatter(norm_q, emp_q, s=5, alpha=0.7)
    lo = min(norm_q.min(), emp_q.min())
    hi = max(norm_q.max(), emp_q.max())
    axes[1, 0].plot([lo, hi], [lo, hi], linewidth=1.1)
    axes[1, 0].set_title("QQ plot", fontsize=11)
    axes[1, 0].set_xlabel("Normal quantiles")
    axes[1, 0].set_ylabel("Empirical quantiles")
    axes[1, 0].grid(True, alpha=0.3)

    cum_pnl = portfolio_pnl.cumsum()
    axes[1, 1].plot(cum_pnl.index, cum_pnl, linewidth=1.0)
    axes[1, 1].set_title("Cumulative P&L", fontsize=11)
    axes[1, 1].set_ylabel("USD")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_loss_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_portfolio(force_rebuild=False):
    prices, returns = load_or_build(force_rebuild)

    pnl = compute_component_pnl(prices, returns)
    portfolio_pnl = pnl.sum(axis=1).rename("Portfolio_PnL")
    losses = (-portfolio_pnl).rename("Loss")

    out = pnl.copy()
    out["Portfolio_PnL"] = portfolio_pnl
    out["Loss"] = losses
    out.to_csv(LOSSES_CSV)

    print_pnl_stats(pnl, losses)

    plot_pnl(pnl, portfolio_pnl)
    plot_loss_distribution(losses, portfolio_pnl)

    return pnl, portfolio_pnl, losses


if __name__ == "__main__":
    build_portfolio(force_rebuild=False)