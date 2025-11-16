"""
Main execution script for basket trading with Bayesian Optimization.
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

from src.data_utils import fetch_price_data, prepare_data
from src.cointegration import get_johansen_weights, calculate_spread
from src.bayesian_opt import BayesianOptimizer, create_objective_function, optimize_weights_multi_objective
from src.strategy import TradingStrategy
from src.utils import sharpe_ratio, max_drawdown, profit_factor, half_life
from evaluation import compare_strategies


def main():
    parser = argparse.ArgumentParser(
        description='Basket Trading with Bayesian Optimization'
    )
    
    parser.add_argument(
        '--assets', nargs='+', required=True,
        help='List of asset symbols (e.g., AAPL MSFT GOOGL)'
    )
    parser.add_argument(
        '--start-date', type=str, required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date', type=str, required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--n-trials', type=int, default=50,
        help='Number of BO optimization trials (default: 50)'
    )
    parser.add_argument(
        '--entry-threshold', type=float, default=2.0,
        help='Entry threshold in standard deviations (default: 2.0)'
    )
    parser.add_argument(
        '--exit-threshold', type=float, default=0.5,
        help='Exit threshold in standard deviations (default: 0.5)'
    )
    parser.add_argument(
        '--transaction-cost', type=float, default=0.001,
        help='Transaction cost per trade (default: 0.001 = 0.1%%)'
    )
    parser.add_argument(
        '--optimize-thresholds', action='store_true',
        help='Optimize entry/exit thresholds using BO'
    )
    parser.add_argument(
        '--metric', type=str, default='sharpe',
        choices=['sharpe', 'return', 'profit_factor'],
        help='Metric to optimize (default: sharpe)'
    )
    parser.add_argument(
        '--multi-objective', action='store_true',
        help='Use multi-objective optimization'
    )
    parser.add_argument(
        '--window-size', type=int, default=252,
        help='Test window size for rolling evaluation (default: 252)'
    )
    parser.add_argument(
        '--step-size', type=int, default=63,
        help='Step size between windows (default: 63)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Basket Trading with Bayesian Optimization")
    print("=" * 80)
    print(f"Assets: {args.assets}")
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Optimization Trials: {args.n_trials}")
    print(f"Metric: {args.metric}")
    print()
    
    # Fetch price data
    print("Fetching price data...")
    try:
        prices = fetch_price_data(args.assets, args.start_date, args.end_date)
        print(f"Fetched {len(prices)} records")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    # Prepare data (convert to log prices)
    print("Preparing data...")
    log_prices, returns = prepare_data(prices)
    print(f"Prepared {len(log_prices)} records")
    print()
    
    # Get Johansen baseline weights
    print("Computing Johansen baseline weights...")
    try:
        johansen_weights, johansen_stats = get_johansen_weights(log_prices)
        print(f"Johansen weights: {johansen_weights}")
        print(f"Trace statistic: {johansen_stats['trace_statistic']:.4f}")
        print(f"Eigenvalue statistic: {johansen_stats['eigenvalue_statistic']:.4f}")
        print()
    except Exception as e:
        print(f"Error in Johansen test: {e}")
        return
    
    # Calculate Johansen spread
    johansen_spread = calculate_spread(log_prices, johansen_weights)
    
    # Evaluate Johansen baseline
    print("Evaluating Johansen baseline strategy...")
    johansen_strategy = TradingStrategy(
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        transaction_cost=args.transaction_cost
    )
    johansen_signals = johansen_strategy.generate_signals(johansen_spread)
    johansen_strategy_returns = johansen_strategy.calculate_returns(
        johansen_spread, johansen_signals, johansen_weights, log_prices
    )
    
    johansen_metrics = johansen_strategy.backtest(
        johansen_spread, johansen_weights, log_prices
    )
    
    print("Johansen Baseline Results:")
    print(f"  Sharpe Ratio: {johansen_metrics['sharpe_ratio']:.4f}")
    print(f"  Total Return: {johansen_metrics['total_return']:.4f}")
    print(f"  Max Drawdown: {johansen_metrics['max_drawdown']:.4f}")
    print(f"  Profit Factor: {johansen_metrics['profit_factor']:.4f}")
    print(f"  Half-Life: {johansen_metrics['half_life']:.4f}")
    print(f"  Number of Trades: {johansen_metrics['num_trades']}")
    print()
    
    # Bayesian Optimization
    print("Running Bayesian Optimization...")
    print(f"  Objective metric: {args.metric}")
    print(f"  Trials: {args.n_trials}")
    
    if args.multi_objective:
        print("  Mode: Multi-objective optimization")
        bo_result = optimize_weights_multi_objective(
            log_prices,
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
            transaction_cost=args.transaction_cost,
            n_trials=args.n_trials
        )
        bo_weights = bo_result['best_weights']
        bo_score = bo_result['best_score']
    else:
        objective_func = create_objective_function(
            log_prices,
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
            transaction_cost=args.transaction_cost,
            metric=args.metric
        )
        
        optimizer = BayesianOptimizer(
            objective_func=objective_func,
            n_assets=len(args.assets),
            n_trials=args.n_trials,
            n_random_starts=10
        )
        
        bo_weights, bo_score = optimizer.optimize(verbose=False)
    
    print(f"BO-optimized weights: {bo_weights}")
    print(f"Best score: {bo_score:.4f}")
    print()
    
    # Calculate BO spread
    bo_spread = calculate_spread(log_prices, bo_weights)
    
    # Evaluate BO strategy
    print("Evaluating BO-optimized strategy...")
    bo_strategy = TradingStrategy(
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        transaction_cost=args.transaction_cost
    )
    bo_signals = bo_strategy.generate_signals(bo_spread)
    bo_strategy_returns = bo_strategy.calculate_returns(
        bo_spread, bo_signals, bo_weights, log_prices
    )
    
    bo_metrics = bo_strategy.backtest(
        bo_spread, bo_weights, log_prices
    )
    
    print("BO-Optimized Results:")
    print(f"  Sharpe Ratio: {bo_metrics['sharpe_ratio']:.4f}")
    print(f"  Total Return: {bo_metrics['total_return']:.4f}")
    print(f"  Max Drawdown: {bo_metrics['max_drawdown']:.4f}")
    print(f"  Profit Factor: {bo_metrics['profit_factor']:.4f}")
    print(f"  Half-Life: {bo_metrics['half_life']:.4f}")
    print(f"  Number of Trades: {bo_metrics['num_trades']}")
    print()
    
    # Rolling window evaluation
    print("=" * 80)
    print("Rolling Window Evaluation")
    print("=" * 80)
    
    eval_results = compare_strategies(
        log_prices,
        johansen_weights,
        bo_weights,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        transaction_cost=args.transaction_cost,
        window_size=args.window_size,
        step_size=args.step_size
    )
    
    if eval_results.get('comparison'):
        comp = eval_results['comparison']
        print("\nRolling Window Comparison:")
        print(f"\nJohansen Baseline (mean across windows):")
        print(f"  Sharpe Ratio: {comp['johansen']['mean_sharpe']:.4f} (±{comp['johansen']['std_sharpe']:.4f})")
        print(f"  Max Drawdown: {comp['johansen']['mean_max_drawdown']:.4f}")
        print(f"  Profit Factor: {comp['johansen']['mean_profit_factor']:.4f}")
        print(f"  Total Return: {comp['johansen']['mean_total_return']:.4f}")
        
        print(f"\nBO-Optimized (mean across windows):")
        print(f"  Sharpe Ratio: {comp['bo']['mean_sharpe']:.4f} (±{comp['bo']['std_sharpe']:.4f})")
        print(f"  Max Drawdown: {comp['bo']['mean_max_drawdown']:.4f}")
        print(f"  Profit Factor: {comp['bo']['mean_profit_factor']:.4f}")
        print(f"  Total Return: {comp['bo']['mean_total_return']:.4f}")
        
        print(f"\nImprovement:")
        print(f"  Sharpe Improvement: {comp['improvement']['sharpe_improvement']*100:.2f}%")
        print(f"  Drawdown Reduction: {comp['improvement']['drawdown_reduction']*100:.2f}%")
        print(f"  Return Improvement: {comp['improvement']['return_improvement']*100:.2f}%")
    
    # Save results
    print("\nSaving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'config': {
            'assets': args.assets,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'n_trials': args.n_trials,
            'entry_threshold': args.entry_threshold,
            'exit_threshold': args.exit_threshold,
            'transaction_cost': args.transaction_cost,
            'metric': args.metric,
            'multi_objective': args.multi_objective
        },
        'weights': {
            'johansen': johansen_weights.tolist(),
            'bo': bo_weights.tolist()
        },
        'full_sample_metrics': {
            'johansen': johansen_metrics,
            'bo': bo_metrics
        },
        'rolling_window_comparison': eval_results.get('comparison', {}),
        'johansen_stats': {
            'trace_statistic': float(johansen_stats['trace_statistic']),
            'eigenvalue_statistic': float(johansen_stats['eigenvalue_statistic'])
        }
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results = convert_numpy_types(results)
    
    results_file = os.path.join(args.output_dir, f'results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Save DataFrames if available
    if eval_results.get('johansen_df') is not None:
        johansen_df_file = os.path.join(args.output_dir, f'johansen_rolling_{timestamp}.csv')
        eval_results['johansen_df'].to_csv(johansen_df_file)
        print(f"Johansen rolling results saved to {johansen_df_file}")
    
    if eval_results.get('bo_df') is not None:
        bo_df_file = os.path.join(args.output_dir, f'bo_rolling_{timestamp}.csv')
        eval_results['bo_df'].to_csv(bo_df_file)
        print(f"BO rolling results saved to {bo_df_file}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

