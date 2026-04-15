from module1_data import load_or_build
from module2_portfolio import build_portfolio
from module3_var_es import estimate_var_es, OOS_FORECASTS_CSV
from module4_backtesting import run_backtest
from module6_multiday_var import run as run_multiday_var
from module7_stress_testing import main as stress_test_main


def main():
    # Module 1 — data loading and cleaning
    print("\n" + "=" * 60)
    print("MODULE 1: Data")
    print("=" * 60)
    returns = load_or_build(force_rebuild=False)

    # Module 2 — portfolio construction and P&L
    print("\n" + "=" * 60)
    print("MODULE 2: Portfolio")
    print("=" * 60)
    pnl, portfolio_pnl, losses = build_portfolio(force_rebuild=False)

    # Module 3 — VaR and ES estimation
    print("\n" + "=" * 60)
    print("MODULE 3: VaR / ES estimation")
    print("=" * 60)
    uni_tbl, port_tbl, oos_tbl = estimate_var_es(force=False)

    # Module 4 — backtesting
    print("\n" + "=" * 60)
    print("MODULE 4: Backtesting")
    print("=" * 60)
    violations_tbl, hit = run_backtest()

    # Module 6 — multi-day VaR
    print("\n" + "=" * 60)
    print("MODULE 6: Multi-day VaR")
    print("=" * 60)
    multiday_tbl = run_multiday_var()

    # Module 7 — stress testing
    print("\n" + "=" * 60)
    print("MODULE 7: Stress Testing")
    print("=" * 60)
    stress_tbl = stress_test_main()


if __name__ == "__main__":
    main()
