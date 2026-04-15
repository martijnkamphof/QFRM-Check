import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
from pathlib import Path

from module1_data import load_or_build
from module2_portfolio import WEIGHTS, PORTFOLIO_VALUE

DATA_DIR  = Path(__file__).resolve().parent / "data"
PLOTS_DIR = DATA_DIR / "plots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

UNIVARIATE_CSV    = DATA_DIR / "var_es_univariate.csv"
PORTFOLIO_CSV     = DATA_DIR / "var_es_portfolio.csv"
OOS_FORECASTS_CSV = DATA_DIR / "var_es_oos_forecasts.csv"

ALPHA          = 0.99
NU             = 5              # Student-t df selected by QQ-plot (PDF section 3.2)
NU_CANDIDATES  = [3, 4, 5, 6]
EWMA_LAMBDA    = 0.94
ROLLING_WINDOW = 250
METHODS        = ["Normal", "Student-t", "Historical", "GARCH-CCC", "FHS-EWMA"]


# ── weight helper ─────────────────────────────────────────────────────────────

def _weights(returns):
    """Weight vector aligned to the column order of `returns`."""
    return np.array([WEIGHTS[c] for c in returns.columns], dtype=float)


# ── VaR / ES primitives (all inputs/outputs in %) ────────────────────────────

def normal_var_es(mu_loss_pct, sigma_pct, alpha):
    """Normal VaR and ES given loss-distribution mean and std in %.

    Loss = -return, so mu_loss = -E[r] and sigma = Std[r].
    VaR_α = mu_loss + z_α · σ
    ES_α  = mu_loss + σ · φ(z_α) / (1 − α)
    """
    z   = stats.norm.ppf(alpha)
    var = mu_loss_pct + z * sigma_pct
    es  = mu_loss_pct + sigma_pct * stats.norm.pdf(z) / (1.0 - alpha)
    return float(var), float(es)


def student_var_es(mu_loss_pct, sigma_pct, alpha, nu):
    """Student-t VaR and ES.  Scale adjusted so variance matches σ².

    c = σ / √(ν/(ν-2))  so that  c · t(ν)  has variance σ².
    """
    q     = stats.t.ppf(alpha, df=nu)
    c     = sigma_pct * np.sqrt((nu - 2.0) / nu)
    es_c  = stats.t.pdf(q, df=nu) / (1.0 - alpha) * (nu + q ** 2) / (nu - 1.0)
    var   = mu_loss_pct + q * c
    es    = mu_loss_pct + es_c * c
    return float(var), float(es)


def hist_var_es(losses_pct, alpha):
    """Historical VaR and ES: empirical quantile and tail mean."""
    x = np.sort(np.asarray(losses_pct, dtype=float))
    x = x[np.isfinite(x)]
    k = max(int(np.floor(alpha * len(x))), 1)
    return float(x[k - 1]), float(x[k - 1:].mean())


# ── volatility models ─────────────────────────────────────────────────────────

def _fit_garch(r_pct):
    """Fit GARCH(1,1) to a return series in %.

    Returns (sigma_next_pct, standardised_residuals).
    """
    r = np.asarray(r_pct, dtype=float)
    r = r[np.isfinite(r)]
    mu = float(np.mean(r))
    s  = float(np.std(r, ddof=1))
    if s <= 1e-12:
        return 0.0, np.zeros(max(len(r), 2))

    # rescale to unit std for numerical stability
    x   = (r - mu) / s
    res = arch_model(x, mean="Zero", vol="Garch", p=1, q=1,
                     dist="normal", rescale=False).fit(
        disp="off", show_warning=False, options={"maxiter": 1000}
    )
    var_f      = float(np.asarray(
        res.forecast(horizon=1, reindex=False).variance
    ).flat[-1])
    sigma_next = s * np.sqrt(max(var_f, 0.0))   # back to original units

    z = np.asarray(res.std_resid, dtype=float)
    return sigma_next, z[np.isfinite(z)]


def _ewma_vol(r_pct, lam=EWMA_LAMBDA):
    """EWMA variance filter.  Returns (sigma_series, sigma_next), same units."""
    x = np.asarray(r_pct, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([1e-12]), 1e-12
    var    = np.empty(len(x))
    var[0] = max(x[0] ** 2, 1e-12)
    for t in range(1, len(x)):
        var[t] = lam * var[t - 1] + (1.0 - lam) * x[t - 1] ** 2
        var[t] = max(var[t], 1e-12)
    next_var = lam * var[-1] + (1.0 - lam) * x[-1] ** 2
    return np.sqrt(var), float(np.sqrt(max(next_var, 1e-12)))


# ── portfolio return ──────────────────────────────────────────────────────────

def _port_ret_pct(returns):
    """Weighted portfolio return series in %."""
    return (returns.values @ _weights(returns)) * 100.0


# ── Table 2: univariate VaR / ES per asset ───────────────────────────────────

def compute_univariate_var_es(returns, alpha=ALPHA, nu=NU):
    """Normal, Student-t and Historical VaR/ES per asset in %.

    Matches PDF Table 2 structure.
    """
    rows = {}
    for col in returns.columns:
        r_pct      = returns[col].dropna().values * 100.0
        mu_loss    = -float(np.nanmean(r_pct))
        sigma      = float(np.nanstd(r_pct, ddof=1))
        losses_pct = -r_pct

        var_n, es_n = normal_var_es(mu_loss, sigma, alpha)
        var_t, es_t = student_var_es(mu_loss, sigma, alpha, nu)
        var_h, es_h = hist_var_es(losses_pct, alpha)

        rows[col] = {
            "VaR_N (%)":  max(var_n, 0.0),
            "ES_N (%)":   max(es_n,  0.0),
            "VaR_t (%)":  max(var_t, 0.0),
            "ES_t (%)":   max(es_t,  0.0),
            "VaR_HS (%)": max(var_h, 0.0),
            "ES_HS (%)":  max(es_h,  0.0),
        }

    return pd.DataFrame(rows).T


# ── Table 3: portfolio VaR / ES (all five methods) ───────────────────────────

def _port_normal(returns, alpha, nu=None):
    w      = _weights(returns)
    mu     = returns.mean().values * 100.0          # % units
    cov    = returns.cov().values * 100.0 ** 2      # %² units
    mu_p   = float(-w @ mu)
    sig_p  = float(np.sqrt(max(w @ cov @ w, 0.0)))
    if nu is None:
        return normal_var_es(mu_p, sig_p, alpha)
    return student_var_es(mu_p, sig_p, alpha, nu)


def _port_historical(returns, alpha):
    return hist_var_es(-_port_ret_pct(returns), alpha)


def _port_garch_ccc(returns, alpha):
    """GARCH(1,1) + Constant Conditional Correlation portfolio VaR/ES."""
    w      = _weights(returns)
    mus    = returns.mean().values * 100.0

    sigma_vec = np.zeros(len(returns.columns))
    z_list    = []
    for i, col in enumerate(returns.columns):
        s_next, z = _fit_garch(returns[col].values * 100.0)
        sigma_vec[i] = s_next
        z_list.append(z)

    min_len = min(len(z) for z in z_list)
    Z = np.column_stack([z[-min_len:] for z in z_list])
    R = np.corrcoef(Z, rowvar=False)
    if not np.all(np.isfinite(R)):
        R = np.eye(len(returns.columns))

    H     = np.diag(sigma_vec) @ R @ np.diag(sigma_vec)   # conditional cov in %²
    mu_p  = float(-w @ mus)
    sig_p = float(np.sqrt(max(w @ H @ w, 0.0)))
    return normal_var_es(mu_p, sig_p, alpha)


def _port_fhs_ewma(returns, alpha):
    """Filtered Historical Simulation with EWMA volatility."""
    w    = _weights(returns)
    mus  = returns.mean().values * 100.0

    sigma_next = np.zeros(len(returns.columns))
    z_list     = []
    for i, col in enumerate(returns.columns):
        r_pct = returns[col].values * 100.0
        eps   = r_pct - mus[i]
        sig_h, s_next = _ewma_vol(eps)
        z = eps[np.isfinite(eps)] / sig_h
        z = z[np.isfinite(z)]
        sigma_next[i] = s_next
        z_list.append(z)

    min_len      = min(len(z) for z in z_list)
    Z            = np.column_stack([z[-min_len:] for z in z_list])
    sim_ret      = mus + Z * sigma_next          # shape (min_len, n_assets) in %
    sim_port_loss = -(sim_ret @ w)
    return hist_var_es(sim_port_loss, alpha)


def compute_portfolio_var_es(returns, alpha=ALPHA, nu=NU):
    """All five methods for portfolio VaR and ES in %.

    Matches PDF Table 3 structure.
    """
    rows = {}

    var, es = _port_normal(returns, alpha)
    rows["Normal Variance-Covariance"] = (max(var, 0.0), max(es, 0.0))

    var, es = _port_normal(returns, alpha, nu=nu)
    rows[f"Student-t (ν={nu})"] = (max(var, 0.0), max(es, 0.0))

    var, es = _port_historical(returns, alpha)
    rows["Historical Simulation"] = (max(var, 0.0), max(es, 0.0))

    var, es = _port_garch_ccc(returns, alpha)
    rows["GARCH(1,1) + CCC"] = (max(var, 0.0), max(es, 0.0))

    var, es = _port_fhs_ewma(returns, alpha)
    rows["FHS with EWMA"] = (max(var, 0.0), max(es, 0.0))

    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["VaR (%)", "ES (%)"]
    )


# ── QQ plots: Student-t degree selection ─────────────────────────────────────

def plot_qq_student_t(returns, filename="05_qq_student_t.png"):
    """QQ plots of standardised portfolio losses vs Student-t(ν), ν ∈ {3,4,5,6}."""
    port_loss = -_port_ret_pct(returns)
    mu, sigma = port_loss.mean(), port_loss.std(ddof=1)
    std_loss  = (port_loss - mu) / sigma

    emp = np.sort(std_loss[np.isfinite(std_loss)])
    n   = len(emp)
    p   = (np.arange(1, n + 1) - 0.5) / n

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, nu in zip(axes.flat, NU_CANDIDATES):
        th = stats.t.ppf(p, df=nu)
        lo, hi = min(th.min(), emp.min()), max(th.max(), emp.max())
        ax.scatter(th, emp, s=2, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], color="red", linewidth=1.2)
        ax.set_title(f"QQ-plot Student-t, df = {nu}", fontsize=10)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {filename}")


# ── comparison bar chart ──────────────────────────────────────────────────────

def plot_portfolio_comparison(port_tbl, filename="06_portfolio_var_es.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    port_tbl["VaR (%)"].plot.bar(ax=ax1, color="steelblue", edgecolor="white")
    ax1.set_title(f"Portfolio VaR at {int(ALPHA*100)}%", fontsize=11)
    ax1.set_ylabel("%")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(True, axis="y", alpha=0.3)

    port_tbl["ES (%)"].plot.bar(ax=ax2, color="coral", edgecolor="white")
    ax2.set_title(f"Portfolio ES at {int(ALPHA*100)}%", fontsize=11)
    ax2.set_ylabel("%")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {filename}")


# ── rolling OOS VaR forecasts (for backtesting in module4) ───────────────────

def compute_oos_rolling(returns, window=ROLLING_WINDOW, alpha=ALPHA, nu=NU):
    """Rolling one-step-ahead portfolio VaR at `alpha` for each method in %.

    For each day t ≥ window the VaR is estimated from the preceding `window`
    observations.  Output is aligned so row t holds the prediction for day t
    (made from data [t-window, t-1]).
    """
    n   = len(returns)
    idx = returns.index[window:]
    out = pd.DataFrame(index=idx, columns=METHODS, dtype=float)

    print(f"  Rolling OOS: {len(idx)} forecasts, window = {window} days ...")
    for i, dt in enumerate(idx):
        win = returns.iloc[i: i + window]

        w   = _weights(win)
        mu  = win.mean().values * 100.0
        cov = win.cov().values * 1e4
        mu_p  = float(-w @ mu)
        sig_p = float(np.sqrt(max(w @ cov @ w, 0.0)))

        out.loc[dt, "Normal"]     = max(normal_var_es(mu_p, sig_p, alpha)[0], 0.0)
        out.loc[dt, "Student-t"]  = max(student_var_es(mu_p, sig_p, alpha, nu)[0], 0.0)
        out.loc[dt, "Historical"] = max(_port_historical(win, alpha)[0], 0.0)
        out.loc[dt, "GARCH-CCC"]  = max(_port_garch_ccc(win, alpha)[0], 0.0)
        out.loc[dt, "FHS-EWMA"]   = max(_port_fhs_ewma(win, alpha)[0], 0.0)

        if (i + 1) % 250 == 0:
            print(f"    {i + 1}/{len(idx)} done")

    return out


# ── main entry point ──────────────────────────────────────────────────────────

def estimate_var_es(force=False):
    """Compute full-sample univariate and portfolio VaR/ES, plus OOS forecasts.

    Returns
    -------
    uni_tbl   : DataFrame  — Table 2: univariate VaR/ES per asset (%)
    port_tbl  : DataFrame  — Table 3: portfolio VaR/ES by method (%)
    oos_tbl   : DataFrame  — rolling OOS VaR predictions (%) for backtesting
    """
    returns = load_or_build(force)

    # QQ plots to motivate ν = 5 choice
    print("\nGenerating Student-t QQ plots ...")
    plot_qq_student_t(returns)

    # Table 2 — univariate
    print("\nComputing univariate VaR / ES (Table 2) ...")
    uni_tbl = compute_univariate_var_es(returns)
    uni_tbl.to_csv(UNIVARIATE_CSV)
    print(f"\nUnivariate VaR / ES at {int(ALPHA*100)}% confidence level:")
    print(uni_tbl.round(2).to_string())

    # Table 3 — portfolio
    print("\nComputing portfolio VaR / ES (Table 3) ...")
    port_tbl = compute_portfolio_var_es(returns)
    port_tbl.to_csv(PORTFOLIO_CSV)
    print(f"\nPortfolio VaR / ES at {int(ALPHA*100)}% confidence level:")
    print(port_tbl.round(2).to_string())

    plot_portfolio_comparison(port_tbl)

    # OOS rolling forecasts for module4 backtesting
    print("\nComputing OOS rolling VaR forecasts ...")
    oos_tbl = compute_oos_rolling(returns)
    oos_tbl.to_csv(OOS_FORECASTS_CSV)
    print(f"OOS forecasts saved → {OOS_FORECASTS_CSV.name}")

    return uni_tbl, port_tbl, oos_tbl


if __name__ == "__main__":
    uni_tbl, port_tbl, oos_tbl = estimate_var_es(force=False)
