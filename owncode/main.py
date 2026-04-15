from module1_data import load_or_build
from module2_portfolio import build_portfolio
from module3_var_es import estimate_var_es, OOS_FORECASTS_CSV, OOS_STRESS_CSV
from module4_backtesting import run_backtest, RESULTS_CSV, STRESS_RESULTS_CSV
from module6_multiday_var import run as run_multiday_var
from module7_stress_testing import main as stress_test_main


def main():
    # Module 1 — data download and cleaning
    print("\n" + "=" * 60)
    print("MODULE 1: Data")
    print("=" * 60)
    prices, returns = load_or_build(force_rebuild=False)

    # Module 2 — portfolio construction and P&L
    print("\n" + "=" * 60)
    print("MODULE 2: Portfolio")
    print("=" * 60)
    pnl, portfolio_pnl, losses = build_portfolio(force_rebuild=False)

    # Module 3 — VaR and ES estimation
    print("\n" + "=" * 60)
    print("MODULE 3: VaR / ES estimation")
    print("=" * 60)
    summary_full, summary_in, summary_oos, realized_losses_oos, stress_oos = estimate_var_es(force=False)

    print("\nFull-sample VaR / ES summary:")
    print(summary_full.round(2))
    print("\nIn-sample VaR / ES summary:")
    print(summary_in.round(2))

    # Module 4 — backtesting
    print("\n" + "=" * 60)
    print("MODULE 4: Backtesting")
    print("=" * 60)
    oos_results = run_backtest(RESULTS_CSV, "oos")
    stress_results = run_backtest(STRESS_RESULTS_CSV, "stress")

    # Module 6 — multi-day VaR
    print("\n" + "=" * 60)
    print("MODULE 6: Multi-day VaR")
    print("=" * 60)
    multiday_results, portfolio_table = run_multiday_var(force_rebuild=False)

    # Module 7 — stress testing
    print("\n" + "=" * 60)
    print("MODULE 7: Stress Testing")
    print("=" * 60)
    baseline_df, stress_df = stress_test_main()


if __name__ == "__main__":
    main()
