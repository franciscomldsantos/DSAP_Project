import pandas as pd
import numpy as np

def process_year_portfolio(test_data_with_scores, past_returns, long_weight=1.0, short_weight=0.5, vol_target=0.15):
    """
    Processes the portfolio construction for a specific year (test_data).
    Returns a list of result dictionaries for the year.
    """
    # The portfolio is rebalanced monthly using the first available date of each month in the test set.
    test_data = test_data_with_scores.copy()
    test_data['ym'] = test_data['date'].dt.to_period('M')
    rebalance_dates = test_data.groupby('ym')['date'].min()
    
    year_results = []
    
    for ym, date in rebalance_dates.items():
        # The cross-section of stocks for the current rebalance date is selected.
        day_data = test_data[test_data['date'] == date].copy()
        
        # --- Liquidity Universe Filter ---
        # To ensure tradeability, the universe is restricted to the top 500 stocks by 3-month Average Daily Volume (ADV).
        day_data = day_data.dropna(subset=['adv_3m'])
        day_data = day_data.sort_values('adv_3m', ascending=False).iloc[:500]

        if len(day_data) < 10: # Execution is skipped if insufficient stocks are available.
            continue
            
        # Stocks are ranked by predicted score. The top 10% form the long leg, and the bottom 10% form the short leg.
        n_stocks = len(day_data)
        n_select = int(np.floor(0.1 * n_stocks))
        
        if n_select == 0:
            continue

        day_data = day_data.sort_values('score', ascending=False)
        long_legs = day_data.iloc[:n_select]
        short_legs = day_data.iloc[-n_select:]
        
        # Equal-weighted asset returns are calculated.
        # Assumption: Positions are entered at Close(t) and held until Close(t+21).
        long_asset_ret = long_legs['fwd_ret_1m'].mean()
        short_asset_ret = short_legs['fwd_ret_1m'].mean()
        
        # Portfolio contributions are calculated based on defined weights (Long: 1.0, Short: 0.5).
        long_contrib = long_weight * long_asset_ret
        short_contrib = -short_weight * short_asset_ret
        
        # The asymmetric return is derived from the combined long and short contributions.
        asymmetric_ret = long_contrib + short_contrib
        
        # --- Volatility Targeting ---
        # Volatility targeting is applied to stabilize returns (Target: 15%, Window: 12 months).
        scale_t = 1.0
        
        # Historical returns are combined with current year results to measure realized volatility.
        all_past_returns = [x['asymmetric_ret'] for x in past_returns] + [x['asymmetric_ret'] for x in year_results]
        
        if len(all_past_returns) >= 12:
            # The standard deviation of the last 12 months is calculated.
            past_rets_window = all_past_returns[-12:]
            realized_vol = np.std(past_rets_window, ddof=1) * np.sqrt(12)
            
            if realized_vol > 0:
                scale_t = vol_target / realized_vol
                # Leverage is capped between 0.5x and 2.0x to manage risk.
                scale_t = max(0.5, min(2.0, scale_t))
        
        vol_targeted_ret = asymmetric_ret * scale_t

        result_entry = {
            'date': date,
            'year': date.year,
            'long_asset_ret': long_asset_ret,
            'short_asset_ret': short_asset_ret,
            'long_contrib': long_contrib,
            'short_contrib': short_contrib,
            'asymmetric_ret': asymmetric_ret,
            'vol_targeted_ret': vol_targeted_ret,
            'scale': scale_t
        }
        year_results.append(result_entry)
        
    return year_results

def calculate_metrics(series):
    """
    Calculates key performance metrics: Sharpe Ratio, Annualized Return, 
    Max Drawdown, and CAGR.
    """
    mean = series.mean()
    std = series.std()
    sharpe = (mean / std) * np.sqrt(12) if std > 0 else 0.0
    ann_ret = mean * 12
    cumulative = (1 + series).prod()
    years = len(series) / 12
    cagr = cumulative**(1 / years) - 1 if years > 0 else 0.0
    cum_ret = (1 + series).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    return sharpe, ann_ret, max_dd, cagr

def print_backtest_results(results_df, test_years):
    """
    Formats and prints the backtest performance report comparing the 
    Asymmetric strategy vs. the Volatility Targeted strategy.
    """
    if results_df.empty:
        print("No portfolio returns calculated.")
        return

    # Data is sorted and cleaned to ensure chronological order.
    results_df = results_df.sort_values('date').reset_index(drop=True)
    results_df = results_df.dropna(subset=['asymmetric_ret', 'vol_targeted_ret'])

    # Sanity Check: Ensures the total return matches the sum of components.
    if not np.allclose(results_df['asymmetric_ret'], results_df['long_contrib'] + results_df['short_contrib']):
        raise ValueError("Sanity check failed: asymmetric_ret != long_contrib + short_contrib")

    sharpe_asym, ann_ret_asym, max_dd_asym, cagr_asym = calculate_metrics(results_df['asymmetric_ret'])
    sharpe_vol, ann_ret_vol, max_dd_vol, cagr_vol = calculate_metrics(results_df['vol_targeted_ret'])
    
    # Component Returns
    asym_long = results_df['long_contrib'].mean() * 12
    asym_short = results_df['short_contrib'].mean() * 12
    
    vol_long = (results_df['long_contrib'] * results_df['scale']).mean() * 12
    vol_short = (results_df['short_contrib'] * results_df['scale']).mean() * 12
    
    print("\n" + "="*30)
    print("Asymmetric Momentum Strategy (Model Comparison)")
    print("="*30)
    print(f"Period: {test_years[0]} - {test_years[-1]}")
    print(f"Total Rebalance Periods: {len(results_df)}")
    print("-" * 55)
    print(f"{'Metric':<20} {'Asymmetric':<15} {'Vol-Target':<15}")
    print("-" * 55)
    print(f"{'Sharpe Ratio':<20} {sharpe_asym:<15.4f} {sharpe_vol:<15.4f}")
    print(f"{'Avg Annual Return':<20} {ann_ret_asym:<15.2%} {ann_ret_vol:<15.2%}")
    print(f"{'CAGR':<20} {cagr_asym:<15.2%} {cagr_vol:<15.2%}")
    print(f"{'Max Drawdown':<20} {max_dd_asym:<15.2%} {max_dd_vol:<15.2%}")
    print("-" * 55)
    print(f"{'Avg Long Return':<20} {asym_long:<15.2%} {vol_long:<15.2%}")
    print(f"{'Avg Short Return':<20} {asym_short:<15.2%} {vol_short:<15.2%}")
    print("="*30)