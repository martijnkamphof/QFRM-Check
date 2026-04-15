import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_data import load_or_build
from module2_portfolio import WEIGHTS
from module3_var_es import ALPHA

DATA_DIR   = Path(__file__).resolve().parent / "data"
PLOTS_DIR  = DATA_DIR / "plots"
OUTPUT_CSV = DATA_DIR / "multiday_var.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 10]


# ── helpers ───────────────────────────────────────────────────────────────────

def _port_ret_pct(returns):
    """Daily portfolio simple returns in %."""
    w = np.array([WEIGHTS[c] for c in returns.columns], dtype=float)
    return (returns.values @ w) * 100.0


def _nonoverlapping_losses(port_ret_pct, h):
    """Non-overlapping h-day compounded portfolio losses in %.

    r^(h) = ∏(1 + r_{t+j−1}) − 1  for j = 1 … h  (PDF Section 5)
    L^(h) = −r^(h)
    """
    r = port_ret_pct / 100.0                          # decimal form
    r = r[np.isfinite(r)]
    n_blocks = len(r) // h
    if n_blocks == 0:
        return np.array([])
    blocks  = r[: n_blocks * h].reshape(n_blocks, h)
    ret_h   = np.prod(1.0 + blocks, axis=1) - 1.0    # compounded h-day return
    return -ret_h * 100.0                             # loss in %


def _hist_var(losses_pct, alpha=ALPHA):
    x = np.sort(losses_pct[np.isfinite(losses_pct)])
    k = max(int(np.floor(alpha * len(x))), 1)
    return float(x[k - 1])


# ── Table 6 ───────────────────────────────────────────────────────────────────

def compute_multiday_var(returns):
    """Build Table 6: historical vs square-root-of-time VaR for h ∈ {1, 5, 10}."""
    port_ret = _port_ret_pct(returns)

    # 1-day VaR is the base for the square-root scaling
    var_1day = _hist_var(_nonoverlapping_losses(port_ret, 1))

    rows = []
    for h in HORIZONS:
        losses_h = _nonoverlapping_losses(port_ret, h)
        var_hs   = _hist_var(losses_h)
        var_sqrt = var_1day * np.sqrt(h)
        rows.append({
            "Horizon":           f"{h}-day",
            "HS VaR (%)":        round(var_hs,   2),
            "Sqrt-of-time (%)":  round(var_sqrt, 2),
            "N blocks":          len(losses_h),
        })

    return pd.DataFrame(rows).set_index("Horizon")


def print_table(tbl):
    print(f"\nTable 6 — Historical multi-day VaR at {int(ALPHA*100)}% vs square-root-of-time")
    print(tbl[["HS VaR (%)", "Sqrt-of-time (%)"]].to_string())


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_multiday_var(tbl, filename="09_multiday_var.png"):
    h_labels = tbl.index.tolist()
    x = np.arange(len(h_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, tbl["HS VaR (%)"],       w, label="Historical Simulation", color="steelblue")
    ax.bar(x + w / 2, tbl["Sqrt-of-time (%)"], w, label="Square-root-of-time",  color="coral")

    for i, (hs, sq) in enumerate(zip(tbl["HS VaR (%)"], tbl["Sqrt-of-time (%)"])):
        ax.text(i - w / 2, hs + 0.05, f"{hs:.2f}%", ha="center", fontsize=8)
        ax.text(i + w / 2, sq + 0.05, f"{sq:.2f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(h_labels)
    ax.set_ylabel("VaR (%)")
    ax.set_title(f"Multi-day portfolio VaR at {int(ALPHA*100)}% confidence level", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {filename}")


# ── main entry point ──────────────────────────────────────────────────────────

def run(force_rebuild=False):
    """Compute Table 6 and return the results DataFrame."""
    returns = load_or_build(force_rebuild)
    tbl     = compute_multiday_var(returns)

    tbl.to_csv(OUTPUT_CSV)
    print_table(tbl)
    plot_multiday_var(tbl)

    return tbl


if __name__ == "__main__":
    run()
