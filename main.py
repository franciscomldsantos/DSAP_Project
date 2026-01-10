import pandas as pd
import numpy as np
import os
import sys

# Add src to path to import modules
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'src'))

from models.xgboost_model import train_and_predict as train_and_predict_xgb
from models.random_forest_model import train_and_predict as train_and_predict_rf
from models.lasso_model import train_and_predict as train_and_predict_lasso
from models.linear_model import train_and_predict as train_and_predict_linear
from backtest import process_year_portfolio, print_backtest_results

def standardize(x):
    """
    Standardizes a pandas Series using Z-score normalization.
    Formula: z = (x - mean) / std
    
    This is used for cross-sectional standardization (per date), ensuring
    that features represent relative rankings rather than absolute values.
    """
    if len(x) > 1 and x.std() > 0:
        return (x - x.mean()) / x.std()
    return 0.0

def main():
    """
    Main execution entry point for the project.
    
    Orchestrates the entire backtesting pipeline:
    1. Data Loading & Preprocessing (Liquidity, Targets).
    2. Walk-Forward Validation (Training on expanding windows).
    3. Model Training & Prediction (XGBoost, RF, Lasso, Linear).
    4. Portfolio Construction (Long/Short based on predictions).
    5. Performance Reporting.
    """
    # Define paths
    data_path = os.path.join(base_dir, 'data', 'sp500_with_momentum.csv')

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run src/calculate_momentum.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure sorted for shift operations (critical for time-series calculations)
    df = df.sort_values(['ticker', 'date'])

    # --- 0. Liquidity Calculation ---
    # The 3-month Average Daily Dollar Volume (ADV) is calculated.
    # This serves as a filter to ensure only liquid stocks are traded.
    print("Calculating liquidity (ADV 3M)...")
    df['dollar_vol'] = df['close'] * df['volume']
    df['adv_3m'] = df.groupby('ticker')['dollar_vol'].transform(lambda x: x.rolling(window=63, min_periods=21).mean())

    # --- 1. Create Target ---
    # The target is the 1-month forward return (approx. 21 trading days).
    # The close price is shifted backwards by 21 days to align today's features with future returns.
    print("Creating target (1-month forward return)...")
    df['fwd_ret_1m'] = df.groupby('ticker')['close'].shift(-21) / df['close'] - 1
    
    # Drop rows with missing targets (the last 21 days of data cannot be used for training)
    df = df.dropna(subset=['fwd_ret_1m'])

    print("Creating market-neutral target...")
    # To isolate stock-specific performance (Alpha), the daily market average return is subtracted.
    # This removes the influence of broad market movements (Beta).
    df['fwd_ret_1m_neutral'] = df.groupby('date')['fwd_ret_1m'].transform(lambda x: x - x.mean())

    # --- 2. Feature Preprocessing ---
    features = ['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_12m']
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features {missing_features}")
        return

    # --- 3. Walk-Forward Setup ---
    # The analysis starts from 2015, but testing begins in 2019 to allow for an initial training window.
    df = df[df['date'].dt.year >= 2015].copy()
    
    years = sorted(df['date'].dt.year.unique())
    test_years = [y for y in years if y >= 2019]
    
    if not test_years:
        print("Error: No data found for test years (2019+).")
        return

    print(f"Starting Walk-Forward Validation on years: {test_years}")
    
    # Strategy Parameters
    LONG_WEIGHT = 1.0
    SHORT_WEIGHT = 0.5
    
    models = {
        "xgb": train_and_predict_xgb,
        "rf": train_and_predict_rf,
        "lasso": train_and_predict_lasso,
        "linear": train_and_predict_linear
    }
    
    portfolio_returns = {k: [] for k in models}
    
    # Walk-Forward Loop: Train on history -> Predict next year -> Expand history
    for year in test_years:
        print(f"\n--- Processing Year {year} ---")
        
        # 1. Define the Test Set (The current year to be predicted)
        test_data = df[df['date'].dt.year == year].copy()
        
        # 2. Define the Train Set with the PURGE
        # Initially, all data before the test year is selected.
        train_data_raw = df[df['date'].dt.year < year].copy()
        
        # --- THE PURGE: Removing Overlap ---
        # Financial data has serial correlation. The target (1-month return) overlaps.
        # If training occurs on data from Dec 31st, it contains knowledge of January returns
        # To prevent Look-Ahead Bias, the last 35 days of the training set are removed.
        if not test_data.empty:
            first_test_date = test_data['date'].min()
            purge_buffer = pd.Timedelta(days=35) 
            cutoff_date = first_test_date - purge_buffer
            
            # Apply the cutoff
            train_data = train_data_raw[train_data_raw['date'] < cutoff_date].copy()
            print(f"   [Purge info] Training cutoff: {cutoff_date.date()} (Test starts: {first_test_date.date()})")
        else:
            train_data = train_data_raw # Fallback if test is empty

        if len(train_data) == 0 or len(test_data) == 0:
            print(f"Skipping {year}: Insufficient data.")
            continue

        # Standardize features cross-sectionally within the loop
        # Features are normalized per day (Cross-Sectional Z-Score).
        # This ensures the model learns from the relative rank of momentum, not the absolute value.
        for col in features:
            # try/except or fillna is used to handle edges where std=0
            train_data[col] = train_data.groupby('date')[col].transform(standardize).fillna(0)
            test_data[col] = test_data.groupby('date')[col].transform(standardize).fillna(0)

        for name, train_func in models.items():
            print(f"Training {name} on {len(train_data)} samples...")
            # The model is trained on historical data and scores are predicted for the test year
            test_data_with_scores = train_func(
                train_data, 
                test_data, 
                features, 
                target_col='fwd_ret_1m_neutral'
            )
            
            print(f"Rebalancing {name}...")
            # The portfolio is constructed based on predictions (Long top 10%, Short bottom 10%)
            year_results = process_year_portfolio(
                test_data_with_scores, 
                past_returns=list(portfolio_returns[name]), 
                long_weight=LONG_WEIGHT, 
                short_weight=SHORT_WEIGHT
            )
            portfolio_returns[name].extend(year_results)

    # --- 6. Backtest Outputs ---
    results_dir = os.path.join(base_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    for name in models:
        results_df = pd.DataFrame(portfolio_returns[name])
        results_df = results_df.sort_values('date').reset_index(drop=True)

        print(f"[INFO] Model: {name} | Rebalance periods: {len(results_df)}")
        
        print("\n" + "="*30)
        print(f">>> {name.upper()} RESULTS <<<")
        print_backtest_results(results_df, test_years)
        
        output_path = os.path.join(results_dir, f'backtest_results_{name}.csv')
        if not results_df.empty:
            results_df.to_csv(output_path, index=False)
            print(f"{name} monthly returns saved to {output_path}")

if __name__ == "__main__":
    main()