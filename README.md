# Portfolio Optimization with Time Series Forecasting

**GMF Investments | Financial Analytics Project**

A complete end-to-end pipeline for portfolio optimization using time series forecasting on historical financial data for TSLA, BND, and SPY.

---

## Business Objective

Guide Me in Finance (GMF) Investments needs a data-driven system to:
- Predict future market trends using historical price data
- Optimize asset allocation across a multi-asset portfolio
- Minimize risk while maximizing returns for clients
- Provide actionable insights through quantitative analysis

The platform answers four key business questions:
1. What are the expected future price movements of high-growth assets like TSLA?
2. How should a portfolio be allocated across TSLA, BND, and SPY to maximize risk-adjusted returns?
3. Does a model-driven strategy outperform a simple passive benchmark?
4. What level of uncertainty exists in long-term forecasts?

---

## Project Structure

```
portfolio-optimization/
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   └── processed/          # Cleaned data and saved charts
├── notebooks/
│   ├── task1_eda.ipynb     # EDA, stationarity tests, risk metrics
│   └── task2_modeling.ipynb # ARIMA and LSTM forecasting models
├── src/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── scripts/
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Assets Analyzed

| Asset | Ticker | Description | Risk Profile |
|-------|--------|-------------|--------------|
| Tesla | TSLA | High-growth stock | High risk, high return |
| Vanguard Bond ETF | BND | U.S. investment-grade bonds | Low risk, stable income |
| S&P 500 ETF | SPY | Tracks S&P 500 Index | Moderate risk, broad exposure |

**Data Period:** January 1, 2015 — June 30, 2026  
**Source:** Yahoo Finance (via yfinance library)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Bethabrh/portfolio-optimization.git
cd portfolio-optimization

# Install dependencies
pip install -r requirements.txt
```

---

## Requirements

```
yfinance
pandas
numpy
matplotlib
seaborn
statsmodels
pmdarima
scikit-learn
tensorflow
keras
PyPortfolioOpt
scipy
jupyterlab
```

---

## Completed Tasks

### Task 1 — Data Extraction and EDA ✅

**What was done:**
- Downloaded 2,888 trading days of data for TSLA, BND, and SPY from Yahoo Finance
- Verified zero missing values across all assets
- Performed comprehensive Exploratory Data Analysis

**Key Findings:**

| Metric | TSLA | SPY | BND |
|--------|------|-----|-----|
| Annualized Return | 45.42% | 14.43% | 2.00% |
| Annualized Volatility | 57.18% | 17.65% | 5.31% |
| Sharpe Ratio | 0.7595 | 0.7042 | -0.0008 |
| Value at Risk (95%) | -5.17% | -1.67% | -0.48% |
| Max Drawdown | -73.63% | -33.72% | -18.58% |

**Stationarity Results (ADF Test):**
- Closing prices: NON-STATIONARY for all 3 assets (p > 0.05)
- Daily returns: STATIONARY for all 3 assets (p < 0.05)
- Implication: ARIMA requires d=1 differencing on price data

**Correlation Analysis:**
- TSLA & SPY: 0.4935 (moderate positive correlation)
- BND & SPY: 0.1157 (low — good diversifier)
- BND & TSLA: 0.0594 (very low — excellent diversifier)

**Outlier Detection:**
- TSLA had the most extreme single-day moves
- All outliers identified using 3 standard deviation threshold

---

### Task 2 — Time Series Forecasting Models ✅

**Train/Test Split:**
- Training: January 2015 — December 2024 (2,516 days)
- Testing: January 2025 — June 2026 (372 days)

**ARIMA Model:**
- Best parameters found via auto_arima: ARIMA(0,1,0)
- The random walk model (d=1, no AR or MA terms) performed best
- This is consistent with the Efficient Market Hypothesis

**LSTM Model Architecture:**
- Input: 60-day lookback window
- Layer 1: LSTM(50 units) + Dropout(0.2)
- Layer 2: LSTM(50 units) + Dropout(0.2)
- Layer 3: Dense(25 units)
- Output: Dense(1 unit)
- Optimizer: Adam | Loss: MSE | Early stopping (patience=5)

**Model Comparison:**

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA(0,1,0) | 54.4412 | 70.5393 | 17.24% |
| LSTM | 12.5247 | 15.9393 | 3.50% |

**Winner: LSTM** — 4x more accurate on all metrics

**Why LSTM outperforms ARIMA:**
- TSLA exhibits highly non-linear price behavior that ARIMA cannot capture
- LSTM learns complex temporal patterns from 60-day sequences
- ARIMA(0,1,0) is essentially a random walk — useful as a baseline but limited for volatile assets

---

## Next Steps (Upcoming Tasks)

### Task 3 — Forecast Future Market Trends
- Use the best-performing LSTM model to generate 6-12 month future forecasts
- Plot forecasts alongside historical data with confidence intervals
- Analyze widening confidence intervals over the forecast horizon
- Identify market opportunities and risks from the forecast

### Task 4 — Portfolio Optimization (Modern Portfolio Theory)
- Use LSTM-forecasted TSLA returns as the expected return input
- Use historical average returns for BND and SPY
- Compute the covariance matrix from historical daily returns
- Generate the Efficient Frontier using PyPortfolioOpt
- Identify the Maximum Sharpe Ratio Portfolio and Minimum Volatility Portfolio
- Recommend optimal portfolio weights for TSLA, BND, and SPY

### Task 5 — Strategy Backtesting
- Define backtesting window: January 2025 — January 2026
- Compare optimized portfolio against a 60% SPY / 40% BND benchmark
- Calculate total return, annualized return, Sharpe ratio, and maximum drawdown
- Evaluate whether the model-driven approach beats the passive benchmark
- Reflect on limitations and potential improvements

---

## Key Learnings So Far

- TSLA's closing price is non-stationary (as expected for stock prices) but daily returns are stationary, confirming standard financial theory
- LSTM significantly outperforms ARIMA for volatile, non-linear assets like TSLA
- BND provides the best diversification benefit with near-zero correlation to TSLA
- The Efficient Market Hypothesis is partially supported — ARIMA's best model is a random walk

---

## Repository

**GitHub:** https://github.com/Bethabrh/portfolio-optimization

---

## Team

- Kerod
- Mahbubah  
- Feven

**Challenge Period:** July 1 — July 7, 2026  
**Interim Submission:** July 5, 2026  
**Final Submission:** July 7, 2026