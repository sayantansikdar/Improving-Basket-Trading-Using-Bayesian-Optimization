# Improving Basket Trading Using Bayesian Optimization

## Overview

This project explores the use of Bayesian Optimization (BO) to improve cointegration-based basket trading performance. Traditional approaches like the Johansen test generate in-sample cointegrating weights that often fail to generalize out-of-sample. This project uses BO to search for optimal weights that maximize out-of-sample profitability.

## What This Project Does - Complete Overview

### The Problem It Solves

Basket trading (also known as pairs trading or statistical arbitrage) is a strategy that exploits temporary price deviations between related assets. The key challenge is finding the right "weights" (proportions) to combine multiple assets into a single spread that mean-reverts reliably.

**Traditional Approach (Johansen Test):**
- Uses statistical tests to find cointegrating relationships
- Generates weights based on in-sample statistical properties
- Often fails to perform well out-of-sample
- Relies on p-values rather than actual trading performance

**This Project's Solution:**
- Uses Bayesian Optimization to search for weights that maximize actual trading performance
- Optimizes directly on out-of-sample profitability metrics (Sharpe ratio, returns, profit factor)
- Treats the trading performance as a "black-box function" and efficiently searches for optimal parameters
- Validates results using rolling window evaluation for robustness

### Complete Workflow - Step by Step

#### 1. **Data Collection & Preprocessing** (`src/data_utils.py`)
   - Fetches historical price data for multiple assets (stocks, ETFs, currencies) using yfinance
   - Handles missing data, aligns dates across assets
   - Converts prices to log prices (required for cointegration analysis)
   - Calculates returns for analysis
   - Can work with custom data from CSV files

#### 2. **Baseline Cointegration Analysis** (`src/cointegration.py`)
   - Applies the Johansen cointegration test to identify long-term relationships
   - Extracts baseline cointegrating weights (traditional statistical approach)
   - Calculates spread series using these weights
   - Computes statistical test results (trace statistic, eigenvalue statistic)
   - Verifies stationarity using ADF test

#### 3. **Bayesian Optimization** (`src/bayesian_opt.py`)
   - **Objective Function Creation**: Builds a function that evaluates trading performance for any given set of weights
   - **Gaussian Process Model**: Uses probabilistic models to predict performance in unexplored regions
   - **Acquisition Function**: Balances exploration vs exploitation to find optimal weights efficiently
   - **Optimization Loop**: Iteratively tests different weight combinations and updates the model
   - **Multi-Objective Support**: Can optimize for multiple metrics simultaneously (Sharpe, return, profit factor)
   - **Constraint Handling**: Ensures weights are normalized (last weight = -1, standard cointegration form)

#### 4. **Trading Strategy Implementation** (`src/strategy.py`)
   - **Mean-Reversion Strategy**: 
     - Enters positions when spread deviates from mean (entry threshold in standard deviations)
     - Exits when spread returns toward mean (exit threshold)
     - Long/short spread based on direction of deviation
   - **Signal Generation**: Creates trading signals (1 = long, -1 = short, 0 = neutral)
   - **Returns Calculation**: Computes strategy returns including:
     - Spread reversion profits/losses
     - Transaction costs (configurable bid-ask spreads)
     - Position changes and entry/exit costs
   - **Backtesting**: Runs complete backtests and calculates performance metrics

#### 5. **Performance Evaluation** (`src/utils.py` & `evaluation.py`)
   - **Key Metrics Calculated**:
     - **Sharpe Ratio**: Risk-adjusted returns (annualized)
     - **Maximum Drawdown**: Worst peak-to-trough decline
     - **Profit Factor**: Gross profit divided by gross loss
     - **Half-Life**: Mean reversion speed (how quickly spread returns to mean)
     - **Total Return**: Cumulative return over the period
     - **Win Rate**: Percentage of profitable trades
     - **Number of Trades**: Trading frequency
   
   - **Rolling Window Evaluation**:
     - Splits data into multiple train/test windows
     - Tests strategy on each window independently
     - Computes average and standard deviation of metrics across windows
     - Provides robust out-of-sample performance estimates

#### 6. **Strategy Comparison** (`evaluation.py`)
   - Compares Johansen baseline vs BO-optimized strategies
   - Calculates improvement percentages across all metrics
   - Generates comprehensive comparison reports
   - Exports results to JSON and CSV formats

#### 7. **Results & Output** (`main.py`)
   - **Console Output**: Real-time progress and performance metrics
   - **JSON Files**: Complete results with all parameters and metrics
   - **CSV Files**: Rolling window results for detailed analysis
   - **Comparison Reports**: Side-by-side performance comparison

### Key Features & Capabilities

#### Core Features:
1. **Automatic Data Fetching**: Downloads price data directly from Yahoo Finance
2. **Statistical Cointegration**: Implements Johansen test with configurable parameters
3. **Bayesian Optimization**: Uses scikit-optimize with Gaussian Process models
4. **Mean-Reversion Trading**: Configurable entry/exit thresholds
5. **Transaction Costs**: Realistic cost modeling (bid-ask spreads, commissions)
6. **Multiple Metrics**: Optimize for Sharpe ratio, total return, or profit factor
7. **Multi-Objective**: Weighted combination of multiple objectives
8. **Rolling Validation**: Out-of-sample testing across multiple time windows
9. **Comprehensive Reporting**: Detailed metrics and comparison analysis

#### Advanced Features:
1. **Threshold Optimization**: Can optimize entry/exit thresholds (future extension)
2. **Slippage Modeling**: Includes realistic slippage in transaction costs
3. **Lookback Periods**: Configurable rolling statistics windows
4. **Normalization**: Standard cointegration weight normalization
5. **Error Handling**: Robust error handling for edge cases
6. **Flexible Input**: Works with any assets available on Yahoo Finance

### Technical Implementation Details

#### Bayesian Optimization Process:
1. **Search Space**: Defines bounds for each weight parameter
2. **Initial Points**: Starts with random exploration
3. **Model Fitting**: Fits Gaussian Process to observed performance
4. **Acquisition**: Uses Expected Improvement (EI) to select next point
5. **Evaluation**: Tests selected weights using trading strategy
6. **Update**: Updates model with new observation
7. **Iteration**: Repeats until convergence or max iterations

#### Trading Strategy Logic:
1. **Spread Calculation**: `spread = w1*log(price1) + w2*log(price2) + ... + wn*log(pricen)`
2. **Normalization**: Z-score of spread using rolling statistics
3. **Entry**: When |z-score| > entry_threshold
4. **Direction**: Long if z-score < -entry_threshold, Short if z-score > entry_threshold
5. **Exit**: When |z-score| < exit_threshold
6. **Returns**: Profit from spread mean reversion minus transaction costs

#### Why Bayesian Optimization Works:
- **Efficiency**: Finds good solutions with relatively few evaluations (30-100 trials)
- **Global Search**: Explores the entire parameter space, not just local optima
- **Adaptive**: Learns from each evaluation to guide future searches
- **Performance-Based**: Optimizes directly on what matters (trading performance)
- **No Gradient Needed**: Works with non-differentiable, noisy objective functions

### What Makes This Project Different

1. **Performance-Oriented**: Unlike statistical tests, optimizes for actual trading results
2. **Out-of-Sample Focus**: Rolling window evaluation ensures robustness
3. **Realistic Modeling**: Includes transaction costs and slippage
4. **Comprehensive Metrics**: Evaluates multiple aspects of trading performance
5. **Flexible Framework**: Easy to extend with new strategies or optimization methods
6. **Production-Ready**: Clean code structure, error handling, and documentation

### Use Cases

- **Quantitative Traders**: Improve basket trading strategies
- **Researchers**: Study cointegration and optimization methods
- **Portfolio Managers**: Find optimal hedges or arbitrage opportunities
- **Students**: Learn about statistical arbitrage and Bayesian optimization
- **Algorithm Developers**: Build and test new trading algorithms

## Features

- **Cointegration Analysis**: Johansen test for identifying cointegrated relationships
- **Bayesian Optimization**: Global optimization for finding optimal basket weights
- **Trading Strategy**: Mean-reversion strategy with configurable entry/exit thresholds
- **Performance Evaluation**: Rolling window backtesting with comprehensive metrics
- **Extensions**: Transaction costs, threshold tuning, multi-objective optimization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
python main.py --assets AAPL MSFT GOOGL --start-date 2020-01-01 --end-date 2023-12-31
```

### Advanced Options

```python
python main.py \
    --assets AAPL MSFT GOOGL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --optimize-thresholds \
    --include-costs \
    --n-trials 50
```

## Project Structure

```
BasketAnalysis/
├── data/               # Data storage
├── results/            # Results storage
├── src/
│   ├── __init__.py        # Package initialization
│   ├── data_utils.py      # Data fetching and preprocessing
│   ├── cointegration.py   # Johansen test implementation
│   ├── bayesian_opt.py    # Bayesian Optimization framework
│   ├── strategy.py        # Trading strategy implementation
│   └── utils.py           # Utility functions
├── evaluation.py       # Performance evaluation and comparison framework
├── main.py            # Main execution script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Key Metrics

- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Maximum Drawdown**: Peak-to-trough decline (percentage)
- **Profit Factor**: Gross profit / gross loss
- **Half-Life**: Mean reversion speed (in periods)
- **Total Return**: Cumulative return over the period
- **Win Rate**: Percentage of profitable trades

## Usage Examples

### Basic Usage
```bash
python main.py --assets AAPL MSFT GOOGL --start-date 2020-01-01 --end-date 2023-12-31
```

### With More Optimization Trials
```bash
python main.py \
    --assets AAPL MSFT GOOGL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --n-trials 100
```

### Multi-Objective Optimization
```bash
python main.py \
    --assets AAPL MSFT GOOGL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --multi-objective \
    --n-trials 100
```

### Custom Thresholds and Costs
```bash
python main.py \
    --assets AAPL MSFT GOOGL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --entry-threshold 2.5 \
    --exit-threshold 0.3 \
    --transaction-cost 0.002 \
    --metric sharpe
```

### Optimizing Different Metrics
```bash
# Optimize for profit factor
python main.py \
    --assets AAPL MSFT GOOGL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --metric profit_factor

# Optimize for return
python main.py \
    --assets AAPL MSFT GOOGL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --metric return
```

## How It Works

1. **Data Collection**: Fetches price data for specified assets using yfinance
2. **Cointegration Analysis**: Applies Johansen test to find baseline cointegrating weights
3. **Bayesian Optimization**: Searches for optimal weights that maximize trading performance
4. **Strategy Evaluation**: Runs mean-reversion trading strategy on both Johansen and BO weights
5. **Rolling Window Validation**: Evaluates strategies across multiple time windows for robustness
6. **Performance Comparison**: Compares metrics between baseline and optimized strategies

## Output

The script generates:
- Console output with performance metrics for both strategies
- JSON file with detailed results in `results/` directory
- CSV files with rolling window performance data

## Notes

- The project uses log prices for cointegration analysis
- Weights are normalized so the last weight is -1 (standard cointegration form)
- Transaction costs and slippage are included in backtesting
- Rolling window evaluation helps assess out-of-sample performance

## License

MIT
