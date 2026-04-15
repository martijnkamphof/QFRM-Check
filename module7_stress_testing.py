import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module2_portfolio import WEIGHTS
from module3_var_es import PORTFOLIO_CSV, compute_portfolio_var_es
from module1_data import load_or_build

DATA_DIR    = Path(__file__).resolve().parent / "data"
PLOTS_DIR   = DATA_DIR / "plots"
RESULTS_CSV = DATA_DIR / "stress_test_results.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

# ── portfolio asset groups (PDF Section 6) ────────────────────────────────────
EQUITY_ASSETS = ["IBM", "MSFT", "MTOR", "SP500"]   # combined equity weight = 70%
FX_ASSET      = "EURUSD"                             # weight = 15%
LOAN_ASSET    = "Loan"                               # weight = 15%

# ── stress scenarios ─────────────────────────────────────────────────────────
# Each entry: (type, shock_value)
#   equity : all EQUITY_ASSETS move by shock_value simultaneously
#   fx     : EUR/USD moves by shock_value
#   rate   : 3-month rate shifts by shock_value (in decimal; 200bp = 0.02)
SCENARIOS = [
    ("Equity −40%",  "equity", -0.40),
    ("Equity −20%",  "equity", -0.20),
    ("Equity +20%",  "equity", +0.20),
    ("Equity +40%",  "equity", +0.40),
    ("FX −20%",      "fx",     -0.20),
    ("FX −10%",      "fx",     -0.10),
    ("FX +10%",      "fx",     +0.10),
    ("FX +20%",      "fx",     +0.20),
    ("Rate +200bp",  "rate",   +0.020),
    ("Rate −200bp",  "rate",   -0.020),
    ("Rate +300bp",  "rate",   +0.030),
    ("Rate −300bp",  "rate",   -0.030),
]


# ── shock loss formula ────────────────────────────────────────────────────────

def _scenario_loss_pct(shock_type, shock_value):
    """Instantaneous portfolio loss in % under one shock.

    General formula (PDF Section 6.1):
        L_stress = −Σ wᵢ · Δrᵢ^shock

    Equity: all four equity assets shocked simultaneously → combined weight 70%.
    FX:     EUR/USD only → weight 15%.
    Rate:   Floating-rate loan; a Δy shift changes the daily return by Δy/252.
            Treated as a liability: rate rise increases cost → positive loss.
    """
    if shock_type == "equity":
        combined_w = sum(WEIGHTS[a] for a in EQUITY_ASSETS)
        return -combined_w * shock_value * 100.0

    elif shock_type == "fx":
        return -WEIGHTS[FX_ASSET] * shock_value * 100.0

    elif shock_type == "rate":
        # Δy in decimal (e.g. 200bp = 0.02); daily return shifts by Δy/252
        return WEIGHTS[LOAN_ASSET] * shock_value / TRADING_DAYS * 100.0

    raise ValueError(f"Unknown shock_type: {shock_type}")


# ── Table 7 ───────────────────────────────────────────────────────────────────

def build_stress_table(var_n_pct, var_hs_pct):
    """Build Table 7: scenario losses vs VaR benchmarks.

    Parameters
    ----------
    var_n_pct  : float  Normal Variance-Covariance VaR at 99% (%)
    var_hs_pct : float  Historical Simulation VaR at 99% (%)
    """
    rows = []
    for label, shock_type, shock_value in SCENARIOS:
        loss = _scenario_loss_pct(shock_type, shock_value)
        ratio = f"{loss / var_n_pct:.2f}×" if loss > 0 else "—"
        rows.append({
            "Scenario":        label,
            "Loss (%)":        round(loss, 3),
            "Ratio to VaR_N":  ratio,
            ">VaR_N":          "Yes" if loss > var_n_pct  else "No",
            ">VaR_HS":         "Yes" if loss > var_hs_pct else "No",
        })

    return pd.DataFrame(rows).set_index("Scenario")


def _load_portfolio_var():
    """Load Normal and HS VaR from the module3 portfolio CSV (recompute if absent)."""
    if PORTFOLIO_CSV.exists():
        port = pd.read_csv(PORTFOLIO_CSV, index_col=0)
    else:
        print("  Portfolio VaR CSV not found — recomputing ...")
        returns = load_or_build()
        port    = compute_portfolio_var_es(returns)

    var_n  = float(port.loc["Normal Variance-Covariance", "VaR (%)"])
    var_hs = float(port.loc["Historical Simulation",      "VaR (%)"])
    return var_n, var_hs


def print_table(tbl, var_n, var_hs):
    print(f"\nTable 7 — Stress test results (VaR_N = {var_n:.2f}%, VaR_HS = {var_hs:.2f}%)")
    print(tbl.to_string())


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_stress_results(tbl, var_n_pct, var_hs_pct, filename="10_stress_test.png"):
    losses = tbl["Loss (%)"].values.astype(float)
    labels = tbl.index.tolist()
    x      = np.arange(len(labels))

    colors = ["tomato" if l > 0 else "steelblue" for l in losses]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, losses, color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(var_n_pct,  color="black",  linestyle="--",
               linewidth=1.2, label=f"VaR_N  {var_n_pct:.2f}%")
    ax.axhline(var_hs_pct, color="dimgrey", linestyle="-.",
               linewidth=1.2, label=f"VaR_HS {var_hs_pct:.2f}%")
    ax.axhline(0, color="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Portfolio loss (%)")
    ax.set_title("Stress test: portfolio loss under instantaneous shocks", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {filename}")


# ── main entry point ──────────────────────────────────────────────────────────

def main():
    """Compute Table 7 and return results DataFrame."""
    var_n, var_hs = _load_portfolio_var()

    tbl = build_stress_table(var_n, var_hs)
    tbl.to_csv(RESULTS_CSV)

    print_table(tbl, var_n, var_hs)
    plot_stress_results(tbl, var_n, var_hs)

    return tbl


if __name__ == "__main__":
    main()
