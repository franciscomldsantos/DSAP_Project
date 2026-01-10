import pandas as pd
import numpy as np
import os

def main():
    # Define paths based on the project structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'sp500_long_data.csv')
    output_path = os.path.join(project_root, 'data', 'sp500_with_momentum.csv')

    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Standardize column names to lowercase for easier access
    df.columns = df.columns.str.lower()

    # Identify necessary columns (assuming standard names 'date', 'ticker', 'close')
    if 'date' not in df.columns or 'ticker' not in df.columns: 
        print(f"Error: Data must contain 'date' and 'ticker' columns. Found: {df.columns.tolist()}")
        return

    price_col = 'close'
    if price_col not in df.columns:
        print(f"Error: Could not identify price column '{price_col}'. Found: {df.columns.tolist()}")
        return

    # Pre-processing
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['ticker', 'date'])

    # Calculate Log Returns from prices
    df['log_ret'] = df.groupby('ticker')[price_col].transform(lambda x: np.log(x).diff())

    # Define momentum windows in trading days (approx 21 days/month)
    windows = {
        'momentum_1m': 21,
        'momentum_3m': 63,
        'momentum_6m': 126,
        'momentum_12m': 252
    }

    print("Calculating momentum features...")
    for name, days in windows.items():
        # Rolling sum of log returns grouped by ticker
        df[name] = df.groupby('ticker')['log_ret'].transform(lambda x: x.rolling(window=days).sum())
        # Convert back to simple cumulative return: exp(sum_log_ret) - 1
        df[name] = np.expm1(df[name])

    # Cleanup temporary column
    df.drop(columns=['log_ret'], inplace=True)

    print(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")

# Entry point: Checks if the script is run directly (not imported as a module)
if __name__ == "__main__":
    main()