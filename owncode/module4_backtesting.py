import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RESULTS_CSV = DATA_DIR / "var_es_oos_forecasts.csv"
STRESS_RESULTS_CSV = DATA_DIR / "var_es_oos_stress_forecasts.csv"
LOSSES_CSV = DATA_DIR / "realized_losses_oos.csv"

PLOTS_DIR = DATA_DIR / "plots"
TABLES_DIR = DATA_DIR / "tables"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = [95, 99]
P = {95: 0.05, 99: 0.01}
ALPHA_TEST = 0.05
TRADING_DAYS = 250


def load_data(results_file):
    preds = pd.read_csv(results_file, index_col=0, header=[0, 1, 2], parse_dates=True)
    losses = pd.read_csv(LOSSES_CSV, index_col=0, parse_dates=True)["Portfolio_Loss"]

    df = preds.copy()
    df[("Realized", "Portfolio", "Loss")] = losses.reindex(df.index)
    df = df.dropna()

    return df, preds

def run_backtest(results_file, label):
    df, original_preds = load_data(results_file)
    loss = df[("Realized", "Portfolio", "Loss")]

    models = original_preds.columns.get_level_values(0).unique().tolist()
    results = {}

    for level in LEVELS:
        p_var = P[level]
        T_out = len(df)

        var_results = []
        es_results = []
        spaces_dict = {}

        fig_loss, axes_loss = plt.subplots(
            len(models), 1, figsize=(10, 3 * len(models)), sharex=True
        )
        axes_loss = np.atleast_1d(axes_loss)

        for i, model in enumerate(models):
            mean_series = df[(model, "Portfolio", "Mean")]
            var_series = df[(model, "Portfolio", f"VaR_{level}")]
            es_series = df[(model, "Portfolio", f"ES_{level}")]
            violations = loss > var_series

            n_hits = int(violations.sum())
            expected_hits = T_out * p_var
            hits_per_year = n_hits / (T_out / TRADING_DAYS)

            binom_res = stats.binomtest(n_hits, T_out, p_var)
            cL = int(stats.binom.ppf(ALPHA_TEST / 2, T_out, p_var))
            cU = int(stats.binom.ppf(1 - ALPHA_TEST / 2, T_out, p_var))

            var_results.append({
                "Model": model,
                "Level": f"{level}%",
                "Expected Hits": round(expected_hits, 2),
                "Actual Hits": n_hits,
                "Hits / Year": round(hits_per_year, 2),
                "Violation Rate": round(n_hits / T_out, 4),
                "cL": cL,
                "cU": cU,
                "Reject 5%": not (cL <= n_hits <= cU),
                "p-value": round(float(binom_res.pvalue), 4)
            })

            viol_days = df.index[violations]
            loss_viol = loss.loc[viol_days]
            es_viol = es_series.loc[viol_days]
            mu_viol = mean_series.loc[viol_days]
            denom = es_viol - mu_viol
            k_viol = (loss_viol - es_viol) / denom
            k_viol = k_viol.replace([np.inf, -np.inf], np.nan).dropna()

            t_stat = np.nan
            p_val = np.nan
            if len(k_viol) > 1:
                t_res = stats.ttest_1samp(k_viol, 0.0)
                t_stat = round(float(t_res.statistic), 4)
                p_val = round(float(t_res.pvalue), 4)

            es_results.append({
                "Model": model,
                "Level": f"{level}%",
                "N": n_hits,
                "Actual ES": round(float(loss_viol.mean()), 2) if n_hits > 0 else np.nan,
                "Expected ES": round(float(es_viol.mean()), 2) if n_hits > 0 else np.nan,
                "Mean Forecast": round(float(mu_viol.mean()), 2) if n_hits > 0 else np.nan,
                "K mean": round(float(k_viol.mean()), 6) if len(k_viol) > 0 else np.nan,
                "t-stat": t_stat,
                "p-value": p_val
            })

            v_idx = np.where(violations)[0] + 1
            spaces = np.diff(np.concatenate(([0], v_idx)))
            spaces_dict[model] = spaces

            ax = axes_loss[i]
            ax.plot(df.index, loss, color="grey", linewidth=1, label="Loss")
            ax.plot(df.index, var_series, color="orange", linewidth=1.2, label=f"VaR_{level}")
            ax.plot(df.index, es_series, color="blue", linestyle="--", linewidth=1.2, label=f"ES_{level}")
            ax.scatter(viol_days, loss_viol, color="red", s=10, label="Violation")

            ax.set_title(f"{model} ({level}%)")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)

        fig_loss.tight_layout()
        fig_loss.savefig(PLOTS_DIR / f"loss_{label}_{level}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_loss)

        fig_qq, axes_qq = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
        axes_qq = np.atleast_1d(axes_qq)

        for i, model in enumerate(models):
            spaces = spaces_dict[model]
            ax = axes_qq[i]

            if len(spaces) > 2:
                (osm, osr), (slope, intercept, _) = stats.probplot(
                    spaces, dist=stats.geom, sparams=(p_var,)
                )
                ax.scatter(osm, osr, alpha=0.6)
                x = np.array([osm.min(), osm.max()])
                ax.plot(x, slope * x + intercept, linestyle="--")
            else:
                ax.text(0.5, 0.5, "few points", ha="center", va="center")

            ax.set_title(model)
            ax.set_xlabel("Theoretical Quantiles")
            if i == 0:
                ax.set_ylabel("Empirical Spacings")
            ax.grid(True, alpha=0.3)

        fig_qq.tight_layout()
        fig_qq.savefig(PLOTS_DIR / f"qq_{label}_{level}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_qq)

        df_var = pd.DataFrame(var_results)
        df_es = pd.DataFrame(es_results)

        df_var.to_csv(TABLES_DIR / f"backtest_var_{label}_{level}.csv", index=False)
        df_es.to_csv(TABLES_DIR / f"backtest_es_{label}_{level}.csv", index=False)

        results[level] = {
            "var": df_var,
            "es": df_es
        }

    return results


if __name__ == "__main__":
    oos = run_backtest(RESULTS_CSV, "oos")
    stress = run_backtest(STRESS_RESULTS_CSV, "stress")