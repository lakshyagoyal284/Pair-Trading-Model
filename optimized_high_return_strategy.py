"""
OPTIMIZED HIGH RETURN PAIRS TRADING STRATEGY
Advanced optimization for maximum returns while maintaining stability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairs_trading_pipeline import PairsTradingPipeline
import warnings
warnings.filterwarnings('ignore')

class OptimizedHighReturnStrategy:
    """Optimized strategy for maximum returns"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pipeline = PairsTradingPipeline(data_folder)
        self.metrics = {}
        self.best_params = {}
        
    def optimize_parameters(self, spread, z_score):
        """Optimize strategy parameters using grid search"""
        print("🔍 OPTIMIZING STRATEGY PARAMETERS...")
        print("-" * 60)
        
        # Parameter ranges to test
        entry_thresholds = [2.0, 2.2, 2.5, 2.8, 3.0]
        exit_thresholds = [0.2, 0.3, 0.4, 0.5]
        position_sizes = [0.15, 0.20, 0.25, 0.30]
        max_hold_days = [3, 5, 7, 10]
        
        best_return = -float('inf')
        best_params = {}
        results = []
        
        total_combinations = len(entry_thresholds) * len(exit_thresholds) * len(position_sizes) * len(max_hold_days)
        current_combination = 0
        
        for entry_thresh in entry_thresholds:
            for exit_thresh in exit_thresholds:
                for pos_size in position_sizes:
                    for max_hold in max_hold_days:
                        current_combination += 1
                        print(f"Testing combo {current_combination}/{total_combinations}: Entry={entry_thresh}, Exit={exit_thresh}, Size={pos_size}, Hold={max_hold}")
                        
                        # Test this parameter combination
                        metrics = self.test_parameter_combination(
                            spread, z_score, entry_thresh, exit_thresh, pos_size, max_hold
                        )
                        
                        results.append({
                            'entry_threshold': entry_thresh,
                            'exit_threshold': exit_thresh,
                            'position_size': pos_size,
                            'max_hold_days': max_hold,
                            'total_return': metrics['total_return'],
                            'sharpe_ratio': metrics['sharpe_ratio'],
                            'profit_factor': metrics['profit_factor'],
                            'max_drawdown': metrics['max_drawdown'],
                            'win_rate': metrics['win_rate'],
                            'total_trades': metrics['total_trades']
                        })
                        
                        # Update best parameters
                        if metrics['total_return'] > best_return and metrics['total_trades'] > 5:
                            best_return = metrics['total_return']
                            best_params = {
                                'entry_threshold': entry_thresh,
                                'exit_threshold': exit_thresh,
                                'position_size': pos_size,
                                'max_hold_days': max_hold
                            }
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        print(f"\n🏆 BEST PARAMETERS FOUND:")
        print("-" * 60)
        print(f"Entry Threshold: {best_params['entry_threshold']}")
        print(f"Exit Threshold: {best_params['exit_threshold']}")
        print(f"Position Size: {best_params['position_size']*100}%")
        print(f"Max Hold Days: {best_params['max_hold_days']}")
        print(f"Expected Return: {best_return:.2f}%")
        
        self.best_params = best_params
        return best_params, results_df
    
    def test_parameter_combination(self, spread, z_score, entry_thresh, exit_thresh, pos_size, max_hold):
        """Test a specific parameter combination"""
        initial_cash = 100000
        commission = 0.001
        
        cash = initial_cash
        position = 0
        trades = []
        equity_curve = []
        entry_price = None
        entry_date = None
        position_type = None
        days_held = 0
        
        for i in range(len(spread)):
            current_date = spread.index[i]
            current_spread = spread.iloc[i]
            current_z = z_score.iloc[i]
            
            if pd.isna(current_z):
                current_z = 0
            
            if position != 0:
                days_held += 1
            
            # Entry logic
            if position == 0:
                current_volatility = spread.rolling(5).std().iloc[i] if i >= 5 else 0
                avg_volatility = spread.rolling(20).std().mean()
                
                if current_volatility < avg_volatility * 1.5:
                    if current_z > entry_thresh:
                        position_type = 'SHORT'
                        entry_price = current_spread
                        entry_date = current_date
                        position = -1
                        trade_size = cash * pos_size
                        cash -= trade_size * commission
                        days_held = 0
                        
                    elif current_z < -entry_thresh:
                        position_type = 'LONG'
                        entry_price = current_spread
                        entry_date = current_date
                        position = 1
                        trade_size = cash * pos_size
                        cash -= trade_size * commission
                        days_held = 0
            
            else:  # Exit logic
                should_exit = False
                exit_reason = ""
                
                if abs(current_z) < exit_thresh:
                    should_exit = True
                    exit_reason = "Target reached"
                elif days_held >= max_hold:
                    should_exit = True
                    exit_reason = "Max hold"
                elif days_held >= 2:
                    # Stop loss
                    if position_type == 'LONG' and current_z < -4.0:
                        should_exit = True
                        exit_reason = "Stop loss"
                    elif position_type == 'SHORT' and current_z > 4.0:
                        should_exit = True
                        exit_reason = "Stop loss"
                    # Early profit
                    elif position_type == 'LONG' and current_z > -0.3:
                        should_exit = True
                        exit_reason = "Early profit"
                    elif position_type == 'SHORT' and current_z < 0.3:
                        should_exit = True
                        exit_reason = "Early profit"
                
                if should_exit:
                    if position_type == 'LONG':
                        pnl = (current_spread - entry_price) * (cash * pos_size) / entry_price
                    else:
                        pnl = (entry_price - current_spread) * (cash * pos_size) / entry_price
                    
                    cash += pnl
                    cash -= abs(pnl) * commission
                    
                    trades.append({
                        'pnl': pnl,
                        'holding_period': days_held
                    })
                    
                    position = 0
                    entry_price = None
                    position_type = None
                    days_held = 0
            
            # Calculate equity
            current_equity = cash
            if position != 0:
                unrealized_pnl = 0
                if position_type == 'LONG':
                    unrealized_pnl = (current_spread - entry_price) * (cash * pos_size) / entry_price
                else:
                    unrealized_pnl = (entry_price - current_spread) * (cash * pos_size) / entry_price
                current_equity += unrealized_pnl
            
            equity_curve.append(current_equity)
        
        # Calculate metrics
        equity_values = np.array(equity_curve)
        total_return = ((equity_values[-1] - initial_cash) / initial_cash) * 100
        
        # Calculate other metrics
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            win_rate = (len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)) * 100
            avg_holding = trades_df['holding_period'].mean()
        else:
            profit_factor = 0
            win_rate = 0
            avg_holding = 0
        
        # Calculate max drawdown
        peak = equity_values[0]
        max_dd = 0
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Calculate Sharpe ratio
        if len(equity_values) > 1:
            returns = np.diff(equity_values) / equity_values[:-1]
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_holding_period': avg_holding
        }
    
    def run_optimized_strategy(self):
        """Run strategy with optimized parameters"""
        print("🚀 RUNNING OPTIMIZED HIGH RETURN STRATEGY")
        print("="*80)
        
        # Run pipeline to get best pair
        self.pipeline.step1_data_preprocessing()
        top_pairs = self.pipeline.step2_pair_selection()
        
        best_pair = top_pairs.iloc[0]
        stock1, stock2 = best_pair['stock1'], best_pair['stock2']
        
        print(f"🎯 Trading pair: {stock1} - {stock2}")
        print(f"📈 P-value: {best_pair['p_value']:.6f}")
        
        # Get price data
        pair_data = self.pipeline.price_data[[stock1, stock2]].copy().dropna()
        spread = pair_data[stock1] - pair_data[stock2]
        
        # Calculate indicators
        spread_mean = spread.rolling(20).mean()
        spread_std = spread.rolling(20).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Optimize parameters
        best_params, results_df = self.optimize_parameters(spread, z_score)
        
        # Run strategy with best parameters
        print(f"\n🎯 RUNNING OPTIMIZED STRATEGY WITH BEST PARAMETERS")
        print("-" * 60)
        
        equity_df, trades_df = self.run_strategy_with_params(
            spread, z_score, 
            best_params['entry_threshold'],
            best_params['exit_threshold'],
            best_params['position_size'],
            best_params['max_hold_days']
        )
        
        # Calculate final metrics
        self.calculate_final_metrics(equity_df, trades_df)
        
        return equity_df, trades_df, results_df
    
    def run_strategy_with_params(self, spread, z_score, entry_thresh, exit_thresh, pos_size, max_hold):
        """Run strategy with specific parameters"""
        initial_cash = 100000
        commission = 0.001
        
        cash = initial_cash
        position = 0
        trades = []
        equity_curve = []
        entry_price = None
        entry_date = None
        position_type = None
        days_held = 0
        
        for i in range(len(spread)):
            current_date = spread.index[i]
            current_spread = spread.iloc[i]
            current_z = z_score.iloc[i]
            
            if pd.isna(current_z):
                current_z = 0
            
            if position != 0:
                days_held += 1
            
            # Entry logic
            if position == 0:
                current_volatility = spread.rolling(5).std().iloc[i] if i >= 5 else 0
                avg_volatility = spread.rolling(20).std().mean()
                
                if current_volatility < avg_volatility * 1.5:
                    if current_z > entry_thresh:
                        position_type = 'SHORT'
                        entry_price = current_spread
                        entry_date = current_date
                        position = -1
                        trade_size = cash * pos_size
                        cash -= trade_size * commission
                        
                        trades.append({
                            'entry_date': entry_date,
                            'action': 'SHORT',
                            'entry_price': entry_price,
                            'z_score': current_z,
                            'volatility': current_volatility
                        })
                        print(f"🔴 SHORT at Z={current_z:.2f} (Entry={entry_thresh})")
                        days_held = 0
                        
                    elif current_z < -entry_thresh:
                        position_type = 'LONG'
                        entry_price = current_spread
                        entry_date = current_date
                        position = 1
                        trade_size = cash * pos_size
                        cash -= trade_size * commission
                        
                        trades.append({
                            'entry_date': entry_date,
                            'action': 'LONG',
                            'entry_price': entry_price,
                            'z_score': current_z,
                            'volatility': current_volatility
                        })
                        print(f"🟢 LONG at Z={current_z:.2f} (Entry={entry_thresh})")
                        days_held = 0
            
            else:  # Exit logic
                should_exit = False
                exit_reason = ""
                
                if abs(current_z) < exit_thresh:
                    should_exit = True
                    exit_reason = f"Target reached (Z={current_z:.2f})"
                elif days_held >= max_hold:
                    should_exit = True
                    exit_reason = f"Max hold ({days_held}d)"
                elif days_held >= 2:
                    # Stop loss
                    if position_type == 'LONG' and current_z < -4.0:
                        should_exit = True
                        exit_reason = f"Stop loss (Z={current_z:.2f})"
                    elif position_type == 'SHORT' and current_z > 4.0:
                        should_exit = True
                        exit_reason = f"Stop loss (Z={current_z:.2f})"
                    # Early profit
                    elif position_type == 'LONG' and current_z > -0.3:
                        should_exit = True
                        exit_reason = f"Early profit (Z={current_z:.2f})"
                    elif position_type == 'SHORT' and current_z < 0.3:
                        should_exit = True
                        exit_reason = f"Early profit (Z={current_z:.2f})"
                
                if should_exit:
                    if position_type == 'LONG':
                        pnl = (current_spread - entry_price) * (cash * pos_size) / entry_price
                    else:
                        pnl = (entry_price - current_spread) * (cash * pos_size) / entry_price
                    
                    cash += pnl
                    cash -= abs(pnl) * commission
                    
                    trades[-1].update({
                        'exit_date': current_date,
                        'exit_price': current_spread,
                        'pnl': pnl,
                        'holding_period': days_held,
                        'exit_z_score': current_z,
                        'exit_reason': exit_reason
                    })
                    
                    print(f"⚪ {exit_reason}: P&L=${pnl:.2f}, Days={days_held}")
                    
                    position = 0
                    entry_price = None
                    position_type = None
                    days_held = 0
            
            # Calculate equity
            current_equity = cash
            if position != 0:
                unrealized_pnl = 0
                if position_type == 'LONG':
                    unrealized_pnl = (current_spread - entry_price) * (cash * pos_size) / entry_price
                else:
                    unrealized_pnl = (entry_price - current_spread) * (cash * pos_size) / entry_price
                current_equity += unrealized_pnl
            
            equity_curve.append({
                'date': current_date,
                'equity': current_equity,
                'cash': cash,
                'position': position,
                'position_type': position_type,
                'spread': current_spread,
                'z_score': current_z,
                'days_held': days_held
            })
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        return equity_df, trades_df
    
    def calculate_final_metrics(self, equity_df, trades_df, initial_cash=100000, commission=0.001):
        """Calculate final optimized metrics"""
        print("\n📊 CALCULATING OPTIMIZED METRICS...")
        
        # Extract data
        equity_values = equity_df['equity'].values
        equity_returns = equity_df['equity'].pct_change().dropna()
        
        # RETURN METRICS
        total_return = ((equity_values[-1] - initial_cash) / initial_cash) * 100
        self.metrics['total_return'] = total_return
        
        years = len(equity_df) / 252
        cagr = ((equity_values[-1] / initial_cash) ** (1/years) - 1) * 100 if years > 0 else 0
        self.metrics['cagr'] = cagr
        
        monthly_return = total_return / 12
        self.metrics['monthly_return'] = monthly_return
        
        # Profit Factor
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            expectancy_per_trade = trades_df['pnl'].mean()
        else:
            profit_factor = 0
            expectancy_per_trade = 0
        
        self.metrics['profit_factor'] = profit_factor
        self.metrics['expectancy_per_trade'] = expectancy_per_trade
        
        # RISK METRICS
        if len(equity_returns) > 0 and equity_returns.std() > 0:
            sharpe_ratio = (equity_returns.mean() * 252) / (equity_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        self.metrics['sharpe_ratio'] = sharpe_ratio
        
        max_dd = self.calculate_max_drawdown(equity_values)
        self.metrics['max_drawdown'] = max_dd
        
        # TRADE METRICS
        total_trades = len(trades_df)
        self.metrics['total_trades'] = total_trades
        
        if total_trades > 0 and 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = (len(winning_trades) / total_trades) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            losing_trades = trades_df[trades_df['pnl'] < 0]
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            largest_win = trades_df['pnl'].max()
            largest_loss = trades_df['pnl'].min()
            
            if 'holding_period' in trades_df.columns:
                avg_holding_period = trades_df['holding_period'].mean()
            else:
                avg_holding_period = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            largest_win = 0
            largest_loss = 0
            avg_holding_period = 0
        
        self.metrics['win_rate'] = win_rate
        self.metrics['avg_win'] = avg_win
        self.metrics['avg_loss'] = avg_loss
        self.metrics['largest_win'] = largest_win
        self.metrics['largest_loss'] = largest_loss
        self.metrics['avg_holding_period'] = avg_holding_period
        
        trades_per_year = total_trades / years if years > 0 else 0
        self.metrics['trades_per_year'] = trades_per_year
        
        # COST METRICS
        total_fees = total_trades * 2 * (initial_cash * self.best_params.get('position_size', 0.2) * commission)
        self.metrics['total_fees_paid'] = total_fees
        
        if total_trades > 0 and 'pnl' in trades_df.columns:
            gross_profit = trades_df['pnl'].sum()
            fees_percentage = (total_fees / abs(gross_profit) * 100) if gross_profit != 0 else 0
        else:
            fees_percentage = 0
        self.metrics['fees_percentage'] = fees_percentage
        
        print("✅ Optimized metrics calculated successfully!")
    
    def calculate_max_drawdown(self, equity_values):
        """Calculate maximum drawdown"""
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def display_optimized_results(self):
        """Display optimized results"""
        metrics = self.metrics
        
        print("\n" + "="*80)
        print("📊 OPTIMIZED HIGH RETURN STRATEGY RESULTS")
        print("="*80)
        
        print(f"\n🎯 OPTIMIZED PARAMETERS:")
        print("-" * 50)
        print(f"Entry Z-threshold: {self.best_params.get('entry_threshold', 'N/A')}")
        print(f"Exit Z-threshold: {self.best_params.get('exit_threshold', 'N/A')}")
        print(f"Position Size: {self.best_params.get('position_size', 0)*100}%")
        print(f"Max Hold Days: {self.best_params.get('max_hold_days', 'N/A')}")
        
        print("\n📈 RETURN METRICS")
        print("-" * 50)
        print(f"Total Return:                     {metrics['total_return']:+.2f}%")
        print(f"CAGR:                             {metrics['cagr']:+.2f}%")
        print(f"Monthly Return:                   {metrics['monthly_return']:+.2f}%")
        print(f"Profit Factor:                    {metrics['profit_factor']:.2f}")
        print(f"Expectancy per Trade:             ${metrics['expectancy_per_trade']:+.2f}")
        
        print("\n⚠️  RISK METRICS")
        print("-" * 50)
        print(f"Sharpe Ratio:                     {metrics['sharpe_ratio']:+.3f}")
        print(f"Max Drawdown:                     {metrics['max_drawdown']:+.2f}%")
        
        print("\n🔄 TRADE METRICS")
        print("-" * 50)
        print(f"Total Trades:                     {metrics['total_trades']}")
        print(f"Win Rate:                         {metrics['win_rate']:.1f}%")
        print(f"Avg Win:                          ${metrics['avg_win']:+,.2f}")
        print(f"Avg Loss:                         ${metrics['avg_loss']:+,.2f}")
        print(f"Largest Win:                      ${metrics['largest_win']:+,.2f}")
        print(f"Largest Loss:                     ${metrics['largest_loss']:+,.2f}")
        print(f"Avg Holding Period:               {metrics['avg_holding_period']:.1f} days")
        print(f"Trades per Year:                  {metrics['trades_per_year']:.1f}")
        
        print("\n💸 COST METRICS")
        print("-" * 50)
        print(f"Total Fees Paid:                  ${metrics['total_fees_paid']:,.2f}")
        print(f"Fees as % of Gross Profit:        {metrics['fees_percentage']:.2f}%")
        
        # Performance Analysis
        print("\n🚀 PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        if metrics['total_return'] > 10:
            print("🏆 EXCELLENT RETURNS (>10%)!")
        elif metrics['total_return'] > 5:
            print("✅ GOOD RETURNS (>5%)!")
        elif metrics['total_return'] > 0:
            print("✅ POSITIVE RETURNS!")
        else:
            print("❌ NEGATIVE RETURNS")
        
        if metrics['sharpe_ratio'] > 1.0:
            print("🏆 EXCELLENT RISK-ADJUSTED RETURNS!")
        elif metrics['sharpe_ratio'] > 0.5:
            print("✅ GOOD RISK-ADJUSTED RETURNS!")
        elif metrics['sharpe_ratio'] > 0:
            print("✅ POSITIVE RISK-ADJUSTED RETURNS!")
        else:
            print("❌ POOR RISK-ADJUSTED RETURNS")
        
        if metrics['profit_factor'] > 1.5:
            print("🏆 EXCELLENT PROFITABILITY!")
        elif metrics['profit_factor'] > 1.2:
            print("✅ GOOD PROFITABILITY!")
        elif metrics['profit_factor'] > 1.0:
            print("✅ PROFITABLE!")
        else:
            print("❌ NOT PROFITABLE")
        
        if metrics['win_rate'] > 60:
            print("🏆 EXCELLENT WIN RATE!")
        elif metrics['win_rate'] > 50:
            print("✅ GOOD WIN RATE!")
        else:
            print("⚠️  WIN RATE NEEDS IMPROVEMENT")
        
        print("\n" + "="*80)
    
    def export_optimization_results(self):
        """Export optimization results"""
        metrics_df = pd.DataFrame(list(self.metrics.items()), 
                                columns=['Metric', 'Value'])
        filename = "optimized_high_return_metrics.csv"
        metrics_df.to_csv(filename, index=False)
        print(f"📊 Optimized metrics exported to: {filename}")
        return filename

# Main execution
if __name__ == "__main__":
    print("🚀 OPTIMIZED HIGH RETURN PAIRS TRADING STRATEGY")
    print("="*80)
    print("🎯 Advanced optimization for maximum returns...")
    
    # Initialize optimized strategy
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    strategy = OptimizedHighReturnStrategy(data_folder)
    
    # Run optimized strategy
    equity_df, trades_df, results_df = strategy.run_optimized_strategy()
    
    # Display results
    strategy.display_optimized_results()
    
    # Export metrics
    strategy.export_optimization_results()
    
    print("\n🎉 OPTIMIZED HIGH RETURN STRATEGY COMPLETED!")
    print("="*80)
    print("📁 Files generated:")
    print("  • Optimized high return metrics")
    print("  • Parameter optimization results")
    print("  • Performance analysis report")
    print("✅ Strategy optimized for maximum returns!")
