import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
from pathlib import Path

from module1_data import load_or_build
from module2_portfolio import WEIGHTS, PORTFOLIO_VALUE, CREDIT_SPREAD, TRADING_DAYS

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = DATA_DIR / "plots"

FULL_SAMPLE_CSV = DATA_DIR / "var_es_full_sample.csv"
INSAMPLE_CSV = DATA_DIR / "var_es_insample_eval.csv"
OOS_FORECASTS_CSV = DATA_DIR / "var_es_oos_forecasts.csv"
REALIZED_LOSSES_OOS_CSV = DATA_DIR / "realized_losses_oos.csv"
OOS_STRESS_CSV = DATA_DIR / "var_es_oos_stress_forecasts.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.95, 0.99]
NU_CANDIDATES = [3, 4, 5, 6]
EWMA_LAMBDA = 0.94

SPLIT_DATE = "2025-01-01"
ROLLING_WINDOW = 250

STRESS_PERIODS = {
    "Shock 2025/4": ("2025-04-01", "2025-12-31"),
    "Shock 2026/3": ("2026-02-01", "2026-03-31"),
}

LOAN_DURATION = 0.25


def get_notionals():
    cols = ["AAPL", "MSFT", "ASML.AS", "^GSPC", "^IRX"]
    vals = [PORTFOLIO_VALUE * WEIGHTS[c] for c in cols]
    return pd.Series(vals, index=cols, dtype=float)


def build_position_returns(prices, returns):
    pos_ret = pd.DataFrame(index=returns.index)

    pos_ret["AAPL"] = np.exp(returns["AAPL"]) - 1.0
    pos_ret["MSFT"] = np.exp(returns["MSFT"]) - 1.0
    pos_ret["^GSPC"] = np.exp(returns["^GSPC"]) - 1.0

    pos_ret["ASML.AS"] = np.exp(returns["ASML.AS"] + returns["EURUSD"]) - 1.0

    lagged_rate = prices["^IRX"].shift(1).reindex(returns.index) / 100.0
    carry = (lagged_rate + CREDIT_SPREAD) / TRADING_DAYS
    rate_mtm = -LOAN_DURATION * (returns["^IRX"] / 100.0)
    pos_ret["^IRX"] = carry + rate_mtm

    return pos_ret.dropna()


def compute_component_pnl(prices, returns):
    pos_ret = build_position_returns(prices, returns)
    notionals = get_notionals()

    pnl = pd.DataFrame(index=pos_ret.index)
    for c in pos_ret.columns:
        pnl[c] = notionals[c] * pos_ret[c]

    return pnl.dropna()


def make_result_table(index_names):
    cols = []
    for alpha in ALPHAS:
        level = int(alpha * 100)
        cols += [f"VaR_{level}", f"ES_{level}"]
    return pd.DataFrame(index=index_names, columns=cols, dtype=float)


def losses_from_pnl(pnl):
    return (-pnl.sum(axis=1)).rename("Loss")


def historical_var_es(losses, alpha):
    x = np.sort(pd.Series(losses).dropna().values)
    n = len(x)

    k = int(np.floor(alpha * n))
    k = min(max(k, 1), n)

    var = float(x[k - 1])
    es = float(x[k - 1:].mean())
    return var, es

def normal_var_es(mu_loss, sigma_loss, alpha):
    z = stats.norm.ppf(alpha)
    var = mu_loss + z * sigma_loss
    es = mu_loss + sigma_loss * stats.norm.pdf(z) / (1.0 - alpha)
    return float(var), float(es)


def student_var_es(mu_loss, sigma_loss, alpha, nu):
    q = stats.t.ppf(alpha, df=nu)
    scale = sigma_loss * np.sqrt((nu - 2.0) / nu)
    es_coeff = (
        stats.t.pdf(q, df=nu) / (1.0 - alpha) * (nu + q**2) / (nu - 1.0)
    )
    var = mu_loss + q * scale
    es = mu_loss + es_coeff * scale
    return float(var), float(es)


def normal_vc(position_returns):
    tickers = list(position_returns.columns)
    notionals = get_notionals().reindex(tickers).values

    mu = position_returns.mean().values
    cov = position_returns.cov().values

    out = make_result_table(tickers + ["Portfolio"])

    for i, t in enumerate(tickers):
        mu_loss = -notionals[i] * mu[i]
        sigma_loss = abs(notionals[i]) * np.sqrt(max(cov[i, i], 0.0))
        for alpha in ALPHAS:
            level = int(alpha * 100)
            var, es = normal_var_es(mu_loss, sigma_loss, alpha)
            out.loc[t, f"VaR_{level}"] = max(var, 0.0)
            out.loc[t, f"ES_{level}"] = max(es, 0.0)

    mu_loss_p = float(-(notionals @ mu))
    sigma_loss_p = float(np.sqrt(max(notionals @ cov @ notionals, 0.0)))

    for alpha in ALPHAS:
        level = int(alpha * 100)
        var, es = normal_var_es(mu_loss_p, sigma_loss_p, alpha)
        out.loc["Portfolio", f"VaR_{level}"] = max(var, 0.0)
        out.loc["Portfolio", f"ES_{level}"] = max(es, 0.0)

    return out


def select_nu(std_losses):
    vals = pd.Series(std_losses).dropna().values
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return 5

    emp = np.sort(vals)
    n = len(emp)
    p = (np.arange(1, n + 1) - 0.5) / n
    mask = (p > 0.005) & (p < 0.995)

    scores = {}
    for nu in NU_CANDIDATES:
        th = stats.t.ppf(p, df=nu)
        scores[nu] = float(np.mean(np.abs(emp[mask] - th[mask])))

    return min(scores, key=scores.get)


def plot_qq_student_t(std_losses, filename="05_qq_student_t.png"):
    vals = pd.Series(std_losses).dropna().values
    vals = vals[np.isfinite(vals)]

    emp = np.sort(vals)
    n = len(emp)
    p = (np.arange(1, n + 1) - 0.5) / n

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for ax, nu in zip(axes.flat, NU_CANDIDATES):
        th = stats.t.ppf(p, df=nu)
        lo = min(th.min(), emp.min())
        hi = max(th.max(), emp.max())
        ax.scatter(th, emp, s=2)
        ax.plot([lo, hi], [lo, hi], color="red")
        ax.set_title(f"QQ Student-t({nu})")
        ax.grid(True, alpha=0.25)

    best_nu = select_nu(vals)
    fig.suptitle(f"Best fit: Student-t({best_nu})", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return best_nu


def student_t_vc(position_returns, losses, plot_qq=True, qq_file="05_qq_student_t.png"):
    tickers = list(position_returns.columns)
    notionals = get_notionals().reindex(tickers).values

    mu = position_returns.mean().values
    cov = position_returns.cov().values

    std_losses = (losses - losses.mean()) / losses.std(ddof=1)
    nu = plot_qq_student_t(std_losses, qq_file) if plot_qq else select_nu(std_losses)

    out = make_result_table(tickers + ["Portfolio"])

    for i, t in enumerate(tickers):
        mu_loss = -notionals[i] * mu[i]
        sigma_loss = abs(notionals[i]) * np.sqrt(max(cov[i, i], 0.0))
        for alpha in ALPHAS:
            level = int(alpha * 100)
            var, es = student_var_es(mu_loss, sigma_loss, alpha, nu)
            out.loc[t, f"VaR_{level}"] = max(var, 0.0)
            out.loc[t, f"ES_{level}"] = max(es, 0.0)

    mu_loss_p = float(-(notionals @ mu))
    sigma_loss_p = float(np.sqrt(max(notionals @ cov @ notionals, 0.0)))

    for alpha in ALPHAS:
        level = int(alpha * 100)
        var, es = student_var_es(mu_loss_p, sigma_loss_p, alpha, nu)
        out.loc["Portfolio", f"VaR_{level}"] = max(var, 0.0)
        out.loc["Portfolio", f"ES_{level}"] = max(es, 0.0)

    return out


def historical_simulation(pnl, losses):
    tickers = list(pnl.columns)
    out = make_result_table(tickers + ["Portfolio"])

    for t in tickers:
        comp_loss = -pnl[t]
        for alpha in ALPHAS:
            level = int(alpha * 100)
            var, es = historical_var_es(comp_loss, alpha)
            out.loc[t, f"VaR_{level}"] = max(var, 0.0)
            out.loc[t, f"ES_{level}"] = max(es, 0.0)

    for alpha in ALPHAS:
        level = int(alpha * 100)
        var, es = historical_var_es(losses, alpha)
        out.loc["Portfolio", f"VaR_{level}"] = max(var, 0.0)
        out.loc["Portfolio", f"ES_{level}"] = max(es, 0.0)

    return out


def fit_garch(series):
    r = pd.Series(series).dropna().values
    mu = float(np.mean(r))
    s = float(np.std(r, ddof=1))
    if s <= 1e-12:
        return 0.0, np.array([0.0])

    x = 100.0 * (r - mu) / s

    res = arch_model(
        x,
        mean="Zero",
        vol="Garch",
        p=1,
        q=1,
        dist="normal",
        rescale=False
    ).fit(disp="off", show_warning=False, options={"maxiter": 1000})

    var_next_x = float(
        np.asarray(res.forecast(horizon=1, reindex=False).variance).flat[-1]
    )
    sigma_next = s * np.sqrt(var_next_x) / 100.0

    z = np.asarray(res.std_resid, dtype=float)
    z = z[np.isfinite(z)]

    return sigma_next, z


def garch_ccc(position_returns):
    tickers = list(position_returns.columns)
    notionals = get_notionals().reindex(tickers).values
    mu = position_returns.mean().values

    sigma_next = {}
    std_resid = {}

    for t in tickers:
        s_next, z = fit_garch(position_returns[t])
        sigma_next[t] = s_next
        std_resid[t] = z

    min_len = min(len(z) for z in std_resid.values())
    Z = np.column_stack([std_resid[t][-min_len:] for t in tickers])

    R = np.corrcoef(Z, rowvar=False)
    sigma_vec = np.array([sigma_next[t] for t in tickers], dtype=float)
    H_next = np.diag(sigma_vec) @ R @ np.diag(sigma_vec)

    out = make_result_table(tickers + ["Portfolio"])

    for i, t in enumerate(tickers):
        mu_loss = -notionals[i] * mu[i]
        sigma_loss = abs(notionals[i]) * sigma_vec[i]
        for alpha in ALPHAS:
            level = int(alpha * 100)
            var, es = normal_var_es(mu_loss, sigma_loss, alpha)
            out.loc[t, f"VaR_{level}"] = max(var, 0.0)
            out.loc[t, f"ES_{level}"] = max(es, 0.0)

    mu_loss_p = float(-(notionals @ mu))
    sigma_loss_p = float(np.sqrt(max(notionals @ H_next @ notionals, 0.0)))

    for alpha in ALPHAS:
        level = int(alpha * 100)
        var, es = normal_var_es(mu_loss_p, sigma_loss_p, alpha)
        out.loc["Portfolio", f"VaR_{level}"] = max(var, 0.0)
        out.loc["Portfolio", f"ES_{level}"] = max(es, 0.0)

    return out


def ewma_vol(series, lam=EWMA_LAMBDA):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([]), np.nan

    var = np.empty(len(x))
    var[0] = max(x[0] ** 2, 1e-12)

    for t in range(1, len(x)):
        var[t] = lam * var[t - 1] + (1.0 - lam) * x[t - 1] ** 2
        var[t] = max(var[t], 1e-12)

    next_var = lam * var[-1] + (1.0 - lam) * x[-1] ** 2
    return np.sqrt(var), np.sqrt(max(next_var, 1e-12))


def fhs_ewma(position_returns):
    tickers = list(position_returns.columns)
    notionals = get_notionals().reindex(tickers).values

    mus = position_returns.mean()
    z_cols = []
    sigma_next = []

    out = make_result_table(tickers + ["Portfolio"])

    for i, t in enumerate(tickers):
        r = position_returns[t].values
        mu = mus[t]
        eps = r - mu

        sigma_hist, s_next = ewma_vol(eps)
        z = eps[np.isfinite(eps)] / sigma_hist
        z = z[np.isfinite(z)]

        sim_r = mu + s_next * z
        sim_loss = -notionals[i] * sim_r

        for alpha in ALPHAS:
            level = int(alpha * 100)
            var, es = historical_var_es(sim_loss, alpha)
            out.loc[t, f"VaR_{level}"] = max(var, 0.0)
            out.loc[t, f"ES_{level}"] = max(es, 0.0)

        z_cols.append(z)
        sigma_next.append(s_next)

    min_len = min(len(z) for z in z_cols)
    Z = np.column_stack([z[-min_len:] for z in z_cols])

    mu_vec = mus.reindex(tickers).values
    sigma_vec = np.array(sigma_next, dtype=float)

    sim_ret_mat = mu_vec + Z * sigma_vec
    sim_pnl_mat = sim_ret_mat * notionals
    sim_port_loss = -sim_pnl_mat.sum(axis=1)

    for alpha in ALPHAS:
        level = int(alpha * 100)
        var, es = historical_var_es(sim_port_loss, alpha)
        out.loc["Portfolio", f"VaR_{level}"] = max(var, 0.0)
        out.loc["Portfolio", f"ES_{level}"] = max(es, 0.0)

    return out


def plot_var_es_comparison(summary):
    methods = ["Normal", "Student-t", "Historical", "GARCH-CCC", "FHS-EWMA"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("VaR_95", "Portfolio VaR 95%"),
        ("ES_95", "Portfolio ES 95%"),
        ("VaR_99", "Portfolio VaR 99%"),
        ("ES_99", "Portfolio ES 99%"),
    ]

    for ax, (measure, title) in zip(axes.flat, panels):
        vals = [summary.loc["Portfolio", (m, measure)] for m in methods]
        ax.bar(methods, vals)
        ax.set_title(title)
        ax.set_ylabel("Loss")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_var_es_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_oos_table(index, tickers):
    cols = []
    for t in tickers + ["Portfolio"]:
        cols += [(t, "Mean")]
        for alpha in ALPHAS:
            level = int(alpha * 100)
            cols += [(t, f"VaR_{level}"), (t, f"ES_{level}")]
    return pd.DataFrame(
        index=index,
        columns=pd.MultiIndex.from_tuples(cols),
        dtype=float
    )


def get_rolling_slice(df, end_idx, window):
    start_idx = max(0, end_idx - window)
    return df.iloc[start_idx:end_idx]


def normal_vc_oos_rolling(position_returns_all, split_date, window=ROLLING_WINDOW):
    tickers = list(position_returns_all.columns)
    notionals = get_notionals().reindex(tickers).values

    index_out = position_returns_all.loc[position_returns_all.index >= split_date].index
    out = make_oos_table(index_out, tickers)

    for dt in index_out:
        end_loc = position_returns_all.index.get_loc(dt)
        train = get_rolling_slice(position_returns_all, end_loc, window)

        mu = train.mean().values
        cov = train.cov().values

        for i, t in enumerate(tickers):
            mu_loss = -notionals[i] * mu[i]
            sigma_loss = abs(notionals[i]) * np.sqrt(max(cov[i, i], 0.0))
            out.loc[dt, (t, "Mean")] = mu_loss

            for alpha in ALPHAS:
                level = int(alpha * 100)
                var, es = normal_var_es(mu_loss, sigma_loss, alpha)
                out.loc[dt, (t, f"VaR_{level}")] = max(var, 0.0)
                out.loc[dt, (t, f"ES_{level}")] = max(es, 0.0)

        mu_loss_p = float(-(notionals @ mu))
        sigma_loss_p = float(np.sqrt(max(notionals @ cov @ notionals, 0.0)))
        out.loc[dt, ("Portfolio", "Mean")] = mu_loss_p

        for alpha in ALPHAS:
            level = int(alpha * 100)
            var, es = normal_var_es(mu_loss_p, sigma_loss_p, alpha)
            out.loc[dt, ("Portfolio", f"VaR_{level}")] = max(var, 0.0)
            out.loc[dt, ("Portfolio", f"ES_{level}")] = max(es, 0.0)

    return out


def student_t_vc_oos_rolling(position_returns_all, losses_all, split_date, window=ROLLING_WINDOW):
    tickers = list(position_returns_all.columns)
    notionals = get_notionals().reindex(tickers).values

    index_out = position_returns_all.loc[position_returns_all.index >= split_date].index
    out = make_oos_table(index_out, tickers)

    for dt in index_out:
        end_loc = position_returns_all.index.get_loc(dt)
        train_ret = get_rolling_slice(position_returns_all, end_loc, window)
        train_loss = get_rolling_slice(losses_all.to_frame("Loss"), end_loc, window)["Loss"]

        mu = train_ret.mean().values
        cov = train_ret.cov().values

        std_losses = (train_loss - train_loss.mean()) / train_loss.std(ddof=1)
        nu = select_nu(std_losses)

        for i, t in enumerate(tickers):
            mu_loss = -notionals[i] * mu[i]
            sigma_loss = abs(notionals[i]) * np.sqrt(max(cov[i, i], 0.0))
            out.loc[dt, (t, "Mean")] = mu_loss

            for alpha in ALPHAS:
                level = int(alpha * 100)
                var, es = student_var_es(mu_loss, sigma_loss, alpha, nu)
                out.loc[dt, (t, f"VaR_{level}")] = max(var, 0.0)
                out.loc[dt, (t, f"ES_{level}")] = max(es, 0.0)

        mu_loss_p = float(-(notionals @ mu))
        sigma_loss_p = float(np.sqrt(max(notionals @ cov @ notionals, 0.0)))
        out.loc[dt, ("Portfolio", "Mean")] = mu_loss_p

        for alpha in ALPHAS:
            level = int(alpha * 100)
            var, es = student_var_es(mu_loss_p, sigma_loss_p, alpha, nu)
            out.loc[dt, ("Portfolio", f"VaR_{level}")] = max(var, 0.0)
            out.loc[dt, ("Portfolio", f"ES_{level}")] = max(es, 0.0)

    return out


def historical_simulation_oos_rolling(pnl_all, losses_all, split_date, window=ROLLING_WINDOW):
    tickers = list(pnl_all.columns)

    index_out = pnl_all.loc[pnl_all.index >= split_date].index
    out = make_oos_table(index_out, tickers)

    for dt in index_out:
        end_loc = pnl_all.index.get_loc(dt)
        train_pnl = get_rolling_slice(pnl_all, end_loc, window)
        train_loss = get_rolling_slice(losses_all.to_frame("Loss"), end_loc, window)["Loss"]

        for t in tickers:
            comp_loss = -train_pnl[t]
            out.loc[dt, (t, "Mean")] = float(comp_loss.mean())

            for alpha in ALPHAS:
                level = int(alpha * 100)
                var, es = historical_var_es(comp_loss, alpha)
                out.loc[dt, (t, f"VaR_{level}")] = max(var, 0.0)
                out.loc[dt, (t, f"ES_{level}")] = max(es, 0.0)

        out.loc[dt, ("Portfolio", "Mean")] = float(train_loss.mean())

        for alpha in ALPHAS:
            level = int(alpha * 100)
            var, es = historical_var_es(train_loss, alpha)
            out.loc[dt, ("Portfolio", f"VaR_{level}")] = max(var, 0.0)
            out.loc[dt, ("Portfolio", f"ES_{level}")] = max(es, 0.0)

    return out


def fit_garch_oos_setup(series):
    r = pd.Series(series).dropna().values
    mu = float(np.mean(r))
    s = float(np.std(r, ddof=1))
    if s <= 1e-12:
        return {
            "mu": 0.0,
            "s": 1.0,
            "omega": 1e-8,
            "alpha": 0.05,
            "beta": 0.90,
            "z": np.array([0.0, 0.0]),
            "last_sig2": 1e-8,
            "last_resid2": 0.0,
        }

    x = 100.0 * (r - mu) / s

    res = arch_model(
        x,
        mean="Zero",
        vol="Garch",
        p=1,
        q=1,
        dist="normal",
        rescale=False
    ).fit(disp="off", show_warning=False, options={"maxiter": 1000})

    z = np.asarray(res.std_resid, dtype=float)
    z = z[np.isfinite(z)]

    return {
        "mu": mu,
        "s": s,
        "omega": float(res.params["omega"]),
        "alpha": float(res.params["alpha[1]"]),
        "beta": float(res.params["beta[1]"]),
        "z": z,
        "last_sig2": max(float(res.conditional_volatility[-1] ** 2), 1e-12),
        "last_resid2": float(res.resid[-1] ** 2),
    }


def garch_ccc_oos_rolling(position_returns_all, split_date, window=ROLLING_WINDOW):
    tickers = list(position_returns_all.columns)
    notionals = get_notionals().reindex(tickers).values

    index_out = position_returns_all.loc[position_returns_all.index >= split_date].index
    out = make_oos_table(index_out, tickers)

    for dt in index_out:
        end_loc = position_returns_all.index.get_loc(dt)
        train = get_rolling_slice(position_returns_all, end_loc, window)

        mu = train.mean().values
        setup = {}
        for t in tickers:
            setup[t] = fit_garch_oos_setup(train[t])

        min_len = min(len(setup[t]["z"]) for t in tickers)
        Z = np.column_stack([setup[t]["z"][-min_len:] for t in tickers])
        R = np.corrcoef(Z, rowvar=False)

        sigma_vec = np.zeros(len(tickers))
        for i, t in enumerate(tickers):
            omega = setup[t]["omega"]
            alpha_g = setup[t]["alpha"]
            beta = setup[t]["beta"]
            sig2 = omega + alpha_g * setup[t]["last_resid2"] + beta * setup[t]["last_sig2"]
            sig2 = max(sig2, 1e-12)
            sigma_vec[i] = setup[t]["s"] * np.sqrt(sig2) / 100.0

            mu_loss = -notionals[i] * mu[i]
            sigma_loss = abs(notionals[i]) * sigma_vec[i]
            out.loc[dt, (t, "Mean")] = mu_loss

            for alpha_level in ALPHAS:
                level = int(alpha_level * 100)
                var, es = normal_var_es(mu_loss, sigma_loss, alpha_level)
                out.loc[dt, (t, f"VaR_{level}")] = max(var, 0.0)
                out.loc[dt, (t, f"ES_{level}")] = max(es, 0.0)

        H = np.diag(sigma_vec) @ R @ np.diag(sigma_vec)
        mu_loss_p = float(-(notionals @ mu))
        sigma_loss_p = float(np.sqrt(max(notionals @ H @ notionals, 0.0)))
        out.loc[dt, ("Portfolio", "Mean")] = mu_loss_p

        for alpha_level in ALPHAS:
            level = int(alpha_level * 100)
            var, es = normal_var_es(mu_loss_p, sigma_loss_p, alpha_level)
            out.loc[dt, ("Portfolio", f"VaR_{level}")] = max(var, 0.0)
            out.loc[dt, ("Portfolio", f"ES_{level}")] = max(es, 0.0)

    return out


def fhs_ewma_oos_rolling(position_returns_all, split_date, window=ROLLING_WINDOW):
    tickers = list(position_returns_all.columns)
    notionals = get_notionals().reindex(tickers).values

    index_out = position_returns_all.loc[position_returns_all.index >= split_date].index
    out = make_oos_table(index_out, tickers)

    for dt in index_out:
        end_loc = position_returns_all.index.get_loc(dt)
        train = get_rolling_slice(position_returns_all, end_loc, window)

        mus = train.mean()
        z_cols = []
        sigma_next = []

        for t in tickers:
            r = train[t].values
            mu = mus[t]
            eps = r - mu
            sigma_hist, s_next = ewma_vol(eps)
            z = eps[np.isfinite(eps)] / sigma_hist
            z = z[np.isfinite(z)]
            z_cols.append(z)
            sigma_next.append(s_next)

        min_len = min(len(z) for z in z_cols)
        Z = np.column_stack([z[-min_len:] for z in z_cols])

        for i, t in enumerate(tickers):
            mu = mus[t]
            sim_r = mu + sigma_next[i] * Z[:, i]
            sim_loss = -notionals[i] * sim_r
            out.loc[dt, (t, "Mean")] = float(sim_loss.mean())

            for alpha in ALPHAS:
                level = int(alpha * 100)
                var, es = historical_var_es(sim_loss, alpha)
                out.loc[dt, (t, f"VaR_{level}")] = max(var, 0.0)
                out.loc[dt, (t, f"ES_{level}")] = max(es, 0.0)

        mu_vec = mus.reindex(tickers).values
        sigma_vec = np.array(sigma_next, dtype=float)

        sim_ret_mat = mu_vec + Z * sigma_vec
        sim_pnl_mat = sim_ret_mat * notionals
        sim_port_loss = -sim_pnl_mat.sum(axis=1)
        out.loc[dt, ("Portfolio", "Mean")] = float(sim_port_loss.mean())

        for alpha in ALPHAS:
            level = int(alpha * 100)
            var, es = historical_var_es(sim_port_loss, alpha)
            out.loc[dt, ("Portfolio", f"VaR_{level}")] = max(var, 0.0)
            out.loc[dt, ("Portfolio", f"ES_{level}")] = max(es, 0.0)

    return out


def make_stress_subset(df):
    parts = []
    for _, (start, end) in STRESS_PERIODS.items():
        parts.append(df.loc[start:end])
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def save_summary_table(res, filename):
    pieces = []
    for name, df in res.items():
        tmp = df.copy()
        tmp.columns = pd.MultiIndex.from_product([[name], tmp.columns])
        pieces.append(tmp)
    summary = pd.concat(pieces, axis=1)
    summary.to_csv(filename)
    return summary


def save_oos_table(res, filename):
    pieces = []
    for name, df in res.items():
        tmp = df.copy()
        tmp.columns = pd.MultiIndex.from_tuples(
            [(name, c[0], c[1]) for c in tmp.columns]
        )
        pieces.append(tmp)
    out = pd.concat(pieces, axis=1)
    out.to_csv(filename)
    return out


def estimate_var_es(force=False):
    prices, returns = load_or_build(force)

    position_returns = build_position_returns(prices, returns)
    pnl = compute_component_pnl(prices, returns)
    losses = losses_from_pnl(pnl)

    common_index = position_returns.index.intersection(pnl.index).intersection(losses.index)
    position_returns = position_returns.loc[common_index]
    pnl = pnl.loc[common_index]
    losses = losses.loc[common_index]

    res_full = {
        "Normal": normal_vc(position_returns),
        "Student-t": student_t_vc(
            position_returns,
            losses,
            plot_qq=True,
            qq_file="05_qq_student_t_full.png"
        ),
        "Historical": historical_simulation(pnl, losses),
        "GARCH-CCC": garch_ccc(position_returns),
        "FHS-EWMA": fhs_ewma(position_returns),
    }

    summary_full = save_summary_table(res_full, FULL_SAMPLE_CSV)
    plot_var_es_comparison(summary_full)

    pnl_in = pnl.loc[pnl.index < SPLIT_DATE]
    losses_in = losses.loc[losses.index < SPLIT_DATE]
    pos_ret_in = position_returns.loc[position_returns.index < SPLIT_DATE]

    res_in = {
        "Normal": normal_vc(pos_ret_in),
        "Student-t": student_t_vc(pos_ret_in, losses_in, plot_qq=False),
        "Historical": historical_simulation(pnl_in, losses_in),
        "GARCH-CCC": garch_ccc(pos_ret_in),
        "FHS-EWMA": fhs_ewma(pos_ret_in),
    }

    summary_in = save_summary_table(res_in, INSAMPLE_CSV)

    res_oos = {
        "Normal": normal_vc_oos_rolling(position_returns, SPLIT_DATE, ROLLING_WINDOW),
        "Student-t": student_t_vc_oos_rolling(position_returns, losses, SPLIT_DATE, ROLLING_WINDOW),
        "Historical": historical_simulation_oos_rolling(pnl, losses, SPLIT_DATE, ROLLING_WINDOW),
        "GARCH-CCC": garch_ccc_oos_rolling(position_returns, SPLIT_DATE, ROLLING_WINDOW),
        "FHS-EWMA": fhs_ewma_oos_rolling(position_returns, SPLIT_DATE, ROLLING_WINDOW),
    }

    summary_oos = save_oos_table(res_oos, OOS_FORECASTS_CSV)

    pnl_out = pnl.loc[pnl.index >= SPLIT_DATE].reindex(summary_oos.index)
    realized_losses_oos = (-pnl_out).copy()
    realized_losses_oos.columns = [f"{c}_Loss" for c in realized_losses_oos.columns]
    realized_losses_oos["Portfolio_Loss"] = realized_losses_oos.sum(axis=1)
    realized_losses_oos.to_csv(REALIZED_LOSSES_OOS_CSV)

    stress_oos = make_stress_subset(summary_oos)
    stress_oos.to_csv(OOS_STRESS_CSV)

    return summary_full, summary_in, summary_oos, realized_losses_oos, stress_oos


if __name__ == "__main__":
    summary_full, summary_in, summary_oos, realized_losses_oos, stress_oos = estimate_var_es(force=False)

    print("\nFull-sample VaR / ES summary:")
    print(summary_full.round(2))

    print("\nIn-sample VaR / ES summary:")
    print(summary_in.round(2))