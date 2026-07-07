# Notebooks — GMF Portfolio Optimization Project

This folder contains the Jupyter notebooks for the **Time Series Forecasting for Portfolio Management Optimization** challenge (GMF Investments). Each notebook corresponds to one task in the project pipeline, from raw data extraction through backtesting a final trading strategy.

## Contents

| Notebook | Task | Description |
|---|---|---|
| `task1_eda.ipynb` | Task 1 — Preprocess & Explore | Data extraction (yfinance), cleaning, EDA, stationarity testing, risk metrics |
| `task2_modeling.ipynb` | Task 2 — Forecasting Models | ARIMA/SARIMA and LSTM model development and comparison |
| `task3_forecast.ipynb` | Task 3 — Future Forecasts | 6–12 month TSLA forecast with confidence intervals and trend analysis |
| `task4_portfolio_optimization.ipynb` | Task 4 — MPT Optimization | Efficient Frontier, Max Sharpe & Min Volatility portfolios |
| `task5_backtesting.ipynb` | Task 5 — Strategy Backtesting | Strategy vs. 60/40 SPY/BND benchmark simulation |

> Rename/organize your actual notebook files to match the table above if they differ — this keeps the structure consistent with the project's required layout.

---

## Task 1 — Data Extraction & EDA

**Data:** TSLA, BND, SPY daily OHLCV data from Yahoo Finance, Jan 1, 2015 – Jun 30, 2026.

**What it covers:**
- Fetching and cleaning historical price data (missing values, dtypes, date indexing)
- Visualizing closing prices, daily returns, and rolling volatility
- Outlier detection on daily returns
- Augmented Dickey-Fuller (ADF) stationarity tests on prices vs. returns
- Foundational risk metrics: Value at Risk (VaR) and Sharpe Ratio

**Key findings:**

| Metric | TSLA | SPY | BND |
|---|---|---|---|
| Annualized Return | 45.42% | 14.43% | 2.00% |
| Annualized Volatility | 57.18% | 17.65% | 5.31% |
| Sharpe Ratio | 0.7595 | 0.7042 | -0.0008 |
| VaR (95%) | -5.17% | -1.67% | -0.48% |
| Max Drawdown | -73.63% | -33.72% | -18.58% |

- Closing prices are **non-stationary** for all three assets (p > 0.05); daily returns are **stationary** (p < 0.05) — confirms ARIMA needs `d = 1` differencing.
- Correlations: TSLA–SPY 0.49 (moderate), BND–SPY 0.12 (low), BND–TSLA 0.06 (near-zero) — BND is the strongest diversifier in the set.

---

## Task 2 — Forecasting Models

**Split:** chronological — train Jan 2015–Dec 2024, test Jan 2025–Jun 2026 (no shuffling).

**ARIMA:** best parameters via `auto_arima` → **ARIMA(0,1,0)**, i.e. a random-walk model — consistent with the Efficient Market Hypothesis.

**LSTM architecture:**
- 60-day lookback window
- LSTM(50) + Dropout(0.2) → LSTM(50) + Dropout(0.2) → Dense(25) → Dense(1)
- Adam optimizer, MSE loss, early stopping (patience = 5)

**Model comparison:**

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| ARIMA(0,1,0) | 54.44 | 70.54 | 17.24% |
| LSTM | 12.52 | 15.94 | 3.50% |

**Winner: LSTM**, roughly 4x more accurate on every metric — it captures TSLA's non-linear price dynamics that a linear ARIMA process cannot.

---

## Task 3 — Future Forecasts

- Uses the LSTM model to generate a 6–12 month forward forecast for TSLA, produced iteratively (predict → feed forward → predict).
- Plots historical prices, test-period predictions, and future forecast on one timeline, with confidence intervals shown separately from the historical/test series.
- Discusses how the confidence interval widens further out in the horizon, and what that implies about the reliability of long-range forecasts (i.e., near-term forecasts are more trustworthy than 9–12 month projections).
- Summarizes opportunities (expected upside scenarios) and risks (volatility, downside scenarios) implied by the forecast band.

*(Fill in this notebook's specific forecast direction, opportunity/risk bullets, and confidence-interval discussion once finalized.)*

---

## Task 4 — Portfolio Optimization (MPT)

- **Expected returns:** TSLA return is taken from the Task 2/3 LSTM forecast; BND and SPY use historical annualized average daily returns.
- **Covariance matrix:** computed from historical daily returns of all three assets.
- **Efficient Frontier:** generated with PyPortfolioOpt, plotting volatility (x) vs. expected return (y).
- Marks the **Maximum Sharpe Ratio (Tangency) Portfolio** and the **Minimum Volatility Portfolio**.
- Recommends a final portfolio with weights for TSLA/BND/SPY, expected annual return, expected volatility, and Sharpe Ratio, with a short justification for the choice.

*(Fill in this notebook's final recommended weights and metrics once the optimization is run.)*

---

## Task 5 — Strategy Backtesting

- **Backtest window:** Jan 2025 – Jan 2026 (held out from model training).
- **Benchmark:** static 60% SPY / 40% BND portfolio.
- **Strategy:** starts from the Task 4 optimal weights, either held fixed or rebalanced monthly.
- Compares cumulative returns, total return, annualized return, Sharpe Ratio, and Max Drawdown for strategy vs. benchmark.
- Concludes whether the model-driven strategy beat the passive benchmark, and notes the backtest's limitations (single historical window, no transaction costs, forecast uncertainty, etc.).

*(Fill in this notebook's final performance table and conclusion once the backtest is run.)*

---

## Running the Notebooks

```bash
# from the project root
pip install -r requirements.txt
jupyter lab notebooks/
```

Run notebooks in order (1 → 5); later notebooks depend on outputs (cleaned data, trained models, forecasts) saved from earlier ones — save any intermediate artifacts to `data/processed/`.

