import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from module1_data import load_or_build
from module2_portfolio import PORTFOLIO_VALUE, WEIGHTS, LOAN_DURATION
from module3_var_es import (
    compute_component_pnl,
    losses_from_pnl,
    build_position_returns,
    normal_vc,
    student_t_vc,
    historical_simulation,
    garch_ccc,
    fhs_ewma,
)

data_folder = Path(__file__).parent / "data"
plots_folder = data_folder / "plots"

baseline_file = data_folder / "stress_test_baseline_portfolio_var.csv"
results_file = data_folder / "stress_test_results.csv"
plot_file = plots_folder / "11_stress_test_loss_vs_var99.png"

plots_folder.mkdir(parents=True, exist_ok=True)

models = ["Normal", "Student-t", "Historical", "GARCH-CCC", "FHS-EWMA"]

stress_scenarios = {
    "Stocks": [-0.40, -0.20, 0.20, 0.40],
    "Index": [-0.40, -0.20, 0.20, 0.40],
    "FX": [-0.20, -0.10, 0.10, 0.20],
    "Rate": [-0.03, -0.02, 0.02, 0.03],
}

def get_baseline_results():
    prices, returns = load_or_build(force_rebuild=False)

    pnl = compute_component_pnl(prices, returns)
    losses = losses_from_pnl(pnl)
    position_returns = build_position_returns(prices, returns)

    shared_dates = pnl.index.intersection(losses.index).intersection(position_returns.index)
    pnl = pnl.loc[shared_dates]
    losses = losses.loc[shared_dates]
    position_returns = position_returns.loc[shared_dates]

    all_model_results = {
        "Normal": normal_vc(position_returns),
        "Student-t": student_t_vc(position_returns, losses, plot_qq=False),
        "Historical": historical_simulation(pnl, losses),
        "GARCH-CCC": garch_ccc(position_returns),
        "FHS-EWMA": fhs_ewma(position_returns),
    }

    baseline_rows = []

    for model_name in models:
        result_table = all_model_results[model_name]

        baseline_rows.append({
            "Model": model_name,
            "VaR_95": float(result_table.loc["Portfolio", "VaR_95"]),
            "VaR_99": float(result_table.loc["Portfolio", "VaR_99"]),
            "ES_95": float(result_table.loc["Portfolio", "ES_95"]),
            "ES_99": float(result_table.loc["Portfolio", "ES_99"]),
        })

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(baseline_file, index=False)

    return baseline_df


def calculate_scenario_result(scenario_type, shock):
    pnl = 0.0

    if scenario_type == "Stocks":
        stock_names = ["AAPL", "MSFT", "ASML.AS"]
        for stock in stock_names:
            pnl += PORTFOLIO_VALUE * WEIGHTS[stock] * shock

    elif scenario_type == "Index":
        pnl += PORTFOLIO_VALUE * WEIGHTS["^GSPC"] * shock

    elif scenario_type == "FX":
        pnl += PORTFOLIO_VALUE * WEIGHTS["ASML.AS"] * shock

    elif scenario_type == "Rate":
        pnl += PORTFOLIO_VALUE * WEIGHTS["^IRX"] * (-LOAN_DURATION * shock)

    loss = -pnl
    return round(float(pnl), 2), round(float(loss), 2)


def run_stress_test():
    baseline_df = get_baseline_results()
    stress_rows = []

    for scenario_type, shock_list in stress_scenarios.items():
        for shock in shock_list:
            pnl, loss = calculate_scenario_result(scenario_type, shock)

            if scenario_type == "Rate":
                shock_label = f"{shock:+.0%} shift"
            else:
                shock_label = f"{shock:+.0%}"

            for _, baseline_row in baseline_df.iterrows():
                var_95 = round(float(baseline_row["VaR_95"]), 2)
                var_99 = round(float(baseline_row["VaR_99"]), 2)
                es_95 = round(float(baseline_row["ES_95"]), 2)
                es_99 = round(float(baseline_row["ES_99"]), 2)

                stress_rows.append({
                    "Category": scenario_type,
                    "Shock": shock_label,
                    "Shock_Value": shock,
                    "Model": baseline_row["Model"],
                    "Scenario_PnL": pnl,
                    "Scenario_Loss": loss,
                    "VaR_95": var_95,
                    "VaR_99": var_99,
                    "ES_95": es_95,
                    "ES_99": es_99,
                    "Loss_minus_VaR_95": round(loss - var_95, 2),
                    "Loss_minus_VaR_99": round(loss - var_99, 2),
                    "Loss_minus_ES_95": round(loss - es_95, 2),
                    "Loss_minus_ES_99": round(loss - es_99, 2),
                })

    results_df = pd.DataFrame(stress_rows)
    results_df.to_csv(results_file, index=False)

    return baseline_df, results_df


def make_plot(results_df):
    plot_df = (
        results_df.groupby(["Category", "Shock", "Shock_Value"], as_index=False)["Loss_minus_VaR_99"]
        .mean()
        .sort_values(["Category", "Shock_Value"])
    )

    x_labels = [
        f"{category} {shock}"
        for category, shock in zip(plot_df["Category"], plot_df["Shock"])
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(x_labels, plot_df["Loss_minus_VaR_99"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Stress test results")
    plt.ylabel("Scenario loss - VaR 99%")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    baseline_df, results_df = run_stress_test()
    make_plot(results_df)

    print("\nBaseline portfolio VaR / ES:")
    print(baseline_df.round(2).to_string(index=False))

    print("\nStress test results:")
    print(results_df.round(2).to_string(index=False))

    return baseline_df, results_df


if __name__ == "__main__":
    main()