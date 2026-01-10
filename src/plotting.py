import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import os

def set_style():
    """Sets a professional plotting style."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except (OSError, AttributeError):
        try:
            plt.style.use('seaborn-darkgrid')
        except (OSError, AttributeError):
            plt.style.use('ggplot')


def plot_performance_comparison(results_dir):
    """Generates the performance comparison plot for all models."""
    print("\n--- Starting Model Performance Comparison ---")
    output_path = os.path.join(results_dir, 'model_comparison.png')

    models = {
        'XGBoost': 'backtest_results_xgb.csv',
        'Random Forest': 'backtest_results_rf.csv',
        'Lasso': 'backtest_results_lasso.csv',
        'Linear Regression': 'backtest_results_linear.csv'
    }

    colors = {
        'XGBoost': '#1f77b4',       # Blue
        'Random Forest': '#ff7f0e', # Orange
        'Lasso': '#2ca02c',         # Green
        'Linear Regression': '#d62728' # Red
    }

    data_frames = {}
    
    # Load Data
    for name, filename in models.items():
        file_path = os.path.join(results_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found. Skipping {name}.")
            continue
            
        try:
            df = pd.read_csv(file_path)
            if 'date' not in df.columns or 'vol_targeted_ret' not in df.columns:
                print(f"Warning: Missing columns in {filename}. Skipping.")
                continue
                
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            data_frames[name] = df
            print(f"Loaded {name}: {len(df)} periods")
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not data_frames:
        print("Error: No valid data loaded for comparison.")
        return

    # Process Data
    plot_data = {}
    for name, df in data_frames.items():
        df['cum_ret'] = (1 + df['vol_targeted_ret']).cumprod() * 100
        df['peak'] = df['cum_ret'].cummax()
        df['drawdown'] = (df['cum_ret'] - df['peak']) / df['peak']
        plot_data[name] = df

    # Plotting
    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Cumulative Returns
    for name, df in plot_data.items():
        ax1.plot(df['date'], df['cum_ret'], label=name, color=colors.get(name, 'black'), linewidth=2)

    ax1.set_title('Model Performance Comparison (Cumulative Returns)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Portfolio Value (Start = 100)', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    ax1.grid(True, alpha=0.5)

    # Drawdowns
    for name, df in plot_data.items():
        ax2.plot(df['date'], df['drawdown'], label=name, color=colors.get(name, 'black'), linewidth=1.5, alpha=0.8)

    ax2.set_title('Drawdown Profile', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    print(f"Saving comparison plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_rolling_metrics(results_dir):
    """Generates the rolling Sharpe ratio comparison plot."""
    print("\n--- Starting Rolling Metrics Analysis ---")
    output_path = os.path.join(results_dir, 'consistency_check.png')

    models = {
        'XGBoost': 'backtest_results_xgb.csv',
        'Random Forest': 'backtest_results_rf.csv',
        'Lasso': 'backtest_results_lasso.csv',
        'Linear Regression': 'backtest_results_linear.csv'
    }

    colors = {
        'XGBoost': '#1f77b4',       # Blue
        'Random Forest': '#ff7f0e', # Orange
        'Lasso': '#2ca02c',         # Green
        'Linear Regression': '#d62728' # Red
    }

    data_frames = {}
    
    # Load Data
    for name, filename in models.items():
        file_path = os.path.join(results_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found. Skipping {name}.")
            continue
            
        try:
            df = pd.read_csv(file_path)
            if 'date' not in df.columns or 'vol_targeted_ret' not in df.columns:
                print(f"Warning: Missing columns in {filename}. Skipping.")
                continue
                
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Calculate Rolling Sharpe (12-month window)
            rolling_mean = df['vol_targeted_ret'].rolling(window=12).mean()
            rolling_std = df['vol_targeted_ret'].rolling(window=12).std()
            
            # Formula: (mean / std) * sqrt(12). Handle 0/NaN.
            rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)
            df['rolling_sharpe'] = rolling_sharpe.fillna(0).replace([np.inf, -np.inf], 0)
            
            data_frames[name] = df
            print(f"Loaded {name}: {len(df)} periods")
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not data_frames:
        print("Error: No valid data loaded for rolling metrics.")
        return

    # Plotting
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, df in data_frames.items():
        ax.plot(df['date'], df['rolling_sharpe'], label=name, color=colors.get(name, 'black'), linewidth=2)

    # Reference Lines
    ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero Threshold')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Excellent (1.0)')

    ax.set_title('Rolling 12-Month Sharpe Ratio (Consistency Check)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Sharpe Ratio (Annualized)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.5)

    plt.tight_layout()
    print(f"Saving consistency check plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_volatility_scaling(results_dir):
    """Generates the volatility scaling factor plot."""
    print("\n--- Starting Volatility Scaling Analysis ---")
    input_path = os.path.join(results_dir, 'backtest_results_xgb.csv')
    output_path = os.path.join(results_dir, 'risk_manager_scale.png')
    
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} not found. Skipping volatility scaling plot.")
        return

    try:
        df = pd.read_csv(input_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'scale' not in df.columns:
        print("Error: 'scale' column missing in data.")
        return

    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Scale
    ax.plot(df['date'], df['scale'], label='Volatility Multiplier', color='#1f77b4', linewidth=2)

    # Reference Lines
    ax.axhline(y=2.0, color='green', linestyle='--', linewidth=1.5, label='Max Cap (2.0)')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Min Floor (0.5)')
    ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5, label='Unlevered (1.0)')

    # Annotation for COVID (March 2020)
    # We look for a date close to March 2020 to place the annotation
    covid_target = pd.Timestamp('2020-03-20')
    if df['date'].min() <= covid_target <= df['date'].max():
        # Find closest date in data
        closest_idx = (df['date'] - covid_target).abs().idxmin()
        closest_date = df.loc[closest_idx, 'date']
        closest_val = df.loc[closest_idx, 'scale']
        
        # Only annotate if within reasonable range (e.g. 60 days)
        if abs((closest_date - covid_target).days) < 60:
            ax.annotate('COVID De-leveraging', 
                        xy=(closest_date, closest_val), 
                        xytext=(closest_date, closest_val + 0.5),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=10, fontweight='bold', ha='center')

    ax.set_title('Volatility Targeting: Leverage Scale Over Time', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Leverage Multiplier', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.5)

    plt.tight_layout()
    print(f"Saving volatility scaling plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_annual_returns(results_dir):
    """Generates the annual returns bar chart comparison."""
    print("\n--- Starting Annual Returns Analysis ---")
    output_path = os.path.join(results_dir, 'annual_returns.png')

    models = {
        'XGBoost': 'backtest_results_xgb.csv',
        'Random Forest': 'backtest_results_rf.csv',
        'Lasso': 'backtest_results_lasso.csv',
        'Linear Regression': 'backtest_results_linear.csv'
    }

    colors = {
        'XGBoost': '#1f77b4',       # Blue
        'Random Forest': '#ff7f0e', # Orange
        'Lasso': '#2ca02c',         # Green
        'Linear Regression': '#d62728' # Red
    }

    annual_data = {}

    # Load and Process Data
    for name, filename in models.items():
        file_path = os.path.join(results_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found. Skipping {name}.")
            continue
            
        try:
            df = pd.read_csv(file_path)
            if 'date' not in df.columns or 'vol_targeted_ret' not in df.columns:
                continue
                
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            
            # Calculate Annual Return: (1+r).prod() - 1
            ann_ret = df.groupby('year')['vol_targeted_ret'].apply(lambda x: (1 + x).prod() - 1)
            annual_data[name] = ann_ret
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not annual_data:
        print("Error: No valid data for annual returns.")
        return

    # Create DataFrame for plotting (Index=Years, Columns=Models)
    df_plot = pd.DataFrame(annual_data)
    df_plot = df_plot.sort_index()

    # Plotting
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Map colors to columns
    plot_colors = [colors.get(col, 'black') for col in df_plot.columns]
    
    # Plot grouped bars
    df_plot.plot(kind='bar', ax=ax, color=plot_colors, width=0.8, edgecolor='black', linewidth=0.5, rot=0)

    # Styling
    ax.set_title('Annual Returns by Model (Volatility Targeted)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Annual Return', fontsize=12)
    ax.set_xlabel('Year', fontsize=12, labelpad=15)
    
    # Format Y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    
    # Horizontal line at 0
    ax.axhline(y=0.0, color='white', linestyle='-', linewidth=1, alpha=0.5)

    # Annotations
    if hasattr(ax, 'bar_label'):
        for container in ax.containers:
            labels = [f'{v:.1%}' for v in container.datavalues]
            ax.bar_label(container, labels=labels, label_type='edge', padding=3, fontsize=9, rotation=90)

    # Adjust layout to prevent overlap
    ax.tick_params(axis='x', pad=25)  # Move x-axis labels (years) down
    ax.margins(y=0.2)                 # Add vertical breathing room for labels

    # Legend
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    print(f"Saving annual returns plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'Results')
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found at {results_dir}")
        print("Please run the backtest first to generate results.")
        return

    plot_performance_comparison(results_dir)
    plot_rolling_metrics(results_dir)
    plot_volatility_scaling(results_dir)
    plot_annual_returns(results_dir)
    print("\nAll plotting tasks completed.")

if __name__ == "__main__":
    main()