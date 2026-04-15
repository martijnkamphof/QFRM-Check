import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from module2_portfolio import build_portfolio

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = DATA_DIR / "plots"
OUTPUT_CSV = DATA_DIR / "multiday_var_results.csv"
PORTFOLIO_TABLE_CSV = DATA_DIR / "multiday_var_portfolio_table.csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.95, 0.99]
HORIZONS = [1, 5, 10]


def historical_var(losses, alpha):
    losses = np.sort(np.asarray(losses, dtype=float))
    losses = losses[np.isfinite(losses)]

    n = len(losses)

    k = int(np.floor(alpha * n))
    k = min(max(k, 1), n)
    return float(losses[k - 1])


def build_nonoverlapping_losses(daily_losses, h):
    daily_losses = np.asarray(daily_losses, dtype=float)
    daily_losses = daily_losses[np.isfinite(daily_losses)]

    n_blocks = len(daily_losses) // h
    if n_blocks == 0:
        return np.array([])

    blocks = daily_losses[:n_blocks * h].reshape(n_blocks, h)
    return blocks.sum(axis=1)


def compute_multiday_var(pnl, losses):
    entities = list(pnl.columns) + ["Portfolio"]

    daily_loss = {}
    for col in pnl.columns:
        series = -pnl[col].values.astype(float)
        daily_loss[col] = series[~np.isnan(series)]

    portfolio_loss = losses.values.astype(float)
    daily_loss["Portfolio"] = portfolio_loss[~np.isnan(portfolio_loss)]

    rows = []

    for alpha in ALPHAS:
        var_1day = {
            entity: historical_var(daily_loss[entity], alpha)
            for entity in entities
        }

        for h in HORIZONS:
            for entity in entities:
                h_losses = build_nonoverlapping_losses(daily_loss[entity], h)
                hs_var = historical_var(h_losses, alpha)
                sqrt_var = var_1day[entity] * np.sqrt(h)

                rows.append({
                    "Alpha": alpha,
                    "Confidence": f"{int(alpha * 100)}%",
                    "Horizon": h,
                    "Entity": entity,
                    "N_obs": len(h_losses),
                    "HS_VaR": hs_var,
                    "SQRT_VaR": sqrt_var,
                    "Ratio_HS_over_SQRT": hs_var / sqrt_var if sqrt_var != 0 else np.nan,
                })

    return pd.DataFrame(rows)


def build_portfolio_table(results):
    port = results[results["Entity"] == "Portfolio"].copy()
    port = port[[
        "Confidence", "Horizon", "N_obs",
        "HS_VaR", "SQRT_VaR", "Ratio_HS_over_SQRT"
    ]]
    return port.sort_values(["Confidence", "Horizon"]).reset_index(drop=True)


def plot_portfolio_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, alpha in zip(axes, ALPHAS):
        port = results[
            (results["Entity"] == "Portfolio") & (results["Alpha"] == alpha)
        ].set_index("Horizon")

        x = np.arange(len(HORIZONS))
        width = 0.34

        hs_vals = [port.loc[h, "HS_VaR"] for h in HORIZONS]
        sqrt_vals = [port.loc[h, "SQRT_VaR"] for h in HORIZONS]

        bars1 = ax.bar(x - width / 2, hs_vals, width, label="Direct historical VaR")
        bars2 = ax.bar(x + width / 2, sqrt_vals, width, label="Square-root rule")

        ax.set_title(f"{int(alpha * 100)}% VaR")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}-day" for h in HORIZONS])
        ax.set_ylabel("Loss")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, axis="y", alpha=0.25)

        ymax = max(max(hs_vals), max(sqrt_vals))
        ax.set_ylim(0, ymax * 1.15)

        for bars in (bars1, bars2):
            for bar in bars:
                val = bar.get_height()
                ax.annotate(
                    f"{val:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

    axes[0].legend(frameon=True)
    fig.suptitle("Portfolio multi-day VaR: historical simulation vs square-root-of-time")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_multiday_portfolio_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_distributions(results, daily_losses):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, alpha in enumerate(ALPHAS):
        port = results[
            (results["Entity"] == "Portfolio") & (results["Alpha"] == alpha)
        ].set_index("Horizon")

        for col, h in enumerate(HORIZONS):
            ax = axes[row, col]
            h_losses = build_nonoverlapping_losses(daily_losses, h)

            ax.hist(h_losses, bins=30, density=True, alpha=0.6, edgecolor="white", linewidth=0.4)

            hs_var = port.loc[h, "HS_VaR"]
            sqrt_var = port.loc[h, "SQRT_VaR"]

            ax.axvline(hs_var, linestyle="--", linewidth=2, label="Direct HS VaR")
            ax.axvline(sqrt_var, linestyle="-.", linewidth=2, label="SQRT rule")

            ax.set_title(f"{int(alpha * 100)}% VaR, {h}-day")
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            ax.tick_params(axis="x", rotation=20)
            ax.grid(True, alpha=0.2)

            if row == 1:
                ax.set_xlabel("Loss")
            if col == 0:
                ax.set_ylabel("Density")
            if row == 0 and col == 0:
                ax.legend(frameon=True, fontsize=8)

    fig.suptitle("Portfolio loss distributions and multi-day VaR thresholds")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "09_multiday_loss_distributions_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, alpha in zip(axes, ALPHAS):
        port = results[
            (results["Entity"] == "Portfolio") & (results["Alpha"] == alpha)
        ].set_index("Horizon")

        h_grid = np.linspace(1, 10, 100)
        var1 = port.loc[1, "HS_VaR"]

        hs_vals = [port.loc[h, "HS_VaR"] for h in HORIZONS]
        sqrt_vals = [port.loc[h, "SQRT_VaR"] for h in HORIZONS]

        ax.plot(HORIZONS, hs_vals, marker="o", linewidth=2, label="Direct historical VaR")
        ax.plot(HORIZONS, sqrt_vals, marker="s", linestyle="--", linewidth=2, label="Square-root rule")
        ax.plot(h_grid, var1 * np.sqrt(h_grid), linewidth=1, alpha=0.5)

        ax.set_title(f"{int(alpha * 100)}% VaR scaling")
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("Loss")
        ax.set_xticks(HORIZONS)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=True)
    fig.suptitle("Portfolio VaR across horizons")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "10_multiday_scaling_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(force_rebuild=False):
    pnl, _, losses = build_portfolio(force_rebuild=force_rebuild)

    results = compute_multiday_var(pnl, losses)
    results.to_csv(OUTPUT_CSV, index=False)

    portfolio_table = build_portfolio_table(results)
    portfolio_table.to_csv(PORTFOLIO_TABLE_CSV, index=False)

    plot_portfolio_comparison(results)
    plot_loss_distributions(results, losses.values)
    plot_scaling_curves(results)

    print("\nPortfolio multi-day VaR results:")
    print(portfolio_table.round(4).to_string(index=False))

    return results, portfolio_table


if __name__ == "__main__":
    run()