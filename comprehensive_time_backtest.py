"""
COMPREHENSIVE TIME-BASED BACKTESTING
Complete backtesting with all requested metrics for different time configurations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairs_trading_pipeline import PairsTradingPipeline
from datetime import time, datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveTimeBacktest:
    """Comprehensive time-based backtesting with all metrics"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pipeline = PairsTradingPipeline(data_folder)
        self.backtest_results = {}
        
    def run_comprehensive_backtest(self):
        """Run comprehensive backtesting with all time configurations"""
        print("📊 COMPREHENSIVE TIME-BASED BACKTESTING")
        print("="*80)
        print("🎯 Running complete backtesting with all requested metrics...")
        
        # Run pipeline to get data
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
        
        # Define time configurations
        time_configs = {
            'All_Day': {'start_time': None, 'end_time': None, 'days': None},
            'Market_Hours': {'start_time': time(9, 15), 'end_time': time(15, 30), 'days': None},
            'Morning_Session': {'start_time': time(9, 15), 'end_time': time(12, 0), 'days': None},
            'Afternoon_Session': {'start_time': time(12, 0), 'end_time': time(15, 30), 'days': None},
            'First_Hour': {'start_time': time(9, 15), 'end_time': time(10, 15), 'days': None},
            'Last_Hour': {'start_time': time(14, 30), 'end_time': time(15, 30), 'days': None},
            'Monday_Only': {'start_time': None, 'end_time': None, 'days': [0]},
            'Friday_Only': {'start_time': None, 'end_time': None, 'days': [4]},
            'Mid_Week': {'start_time': None, 'end_time': None, 'days': [1, 2, 3]},
            'Avoid_Monday': {'start_time': None, 'end_time': None, 'days': [1, 2, 3, 4]},
            'Peak_Hours': {'start_time': time(10, 0), 'end_time': time(14, 0), 'days': None},
            'Quiet_Hours': {'start_time': time(15, 30), 'end_time': time(9, 15), 'days': None}
        }
        
        # Run backtest for each configuration
        all_results = []
        
        for config_name, config in time_configs.items():
            print(f"\n🕐 Testing: {config_name}")
            print("-" * 50)
            
            # Run comprehensive backtest
            metrics = self.run_complete_backtest_with_metrics(
                spread, z_score, config['start_time'], config['end_time'], config['days']
            )
            
            metrics['configuration'] = config_name
            all_results.append(metrics)
            
            # Display key results
            print(f"Total Return: {metrics['total_return']:+.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:+.3f}")
            print(f"Win Rate: {metrics['win_rate']:.1f}%")
            print(f"Total Trades: {metrics['total_trades']}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Display comprehensive results
        self.display_comprehensive_results(results_df)
        
        # Generate detailed report
        self.generate_detailed_report(results_df)
        
        return results_df
    
    def run_complete_backtest_with_metrics(self, spread, z_score, start_time=None, end_time=None, days=None):
        """Run complete backtest with all requested metrics"""
        initial_cash = 100000
        commission = 0.001
        
        # Strategy parameters
        entry_threshold = 2.0
        exit_threshold = 0.5
        position_size = 0.3
        max_hold_days = 3
        
        cash = initial_cash
        position = 0
        trades = []
        equity_curve = []
        predictions = []
        actuals = []
        
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
            
            # Check time filter
            if not self.is_time_allowed(current_date, start_time, end_time, days):
                # Update equity and check exits
                if position != 0:
                    days_held += 1
                    
                    if days_held >= max_hold_days:
                        should_exit = True
                        exit_reason = "Max hold (time filter)"
                        
                        if should_exit:
                            if position_type == 'LONG':
                                pnl = (current_spread - entry_price) * (cash * position_size) / entry_price
                            else:
                                pnl = (entry_price - current_spread) * (cash * position_size) / entry_price
                            
                            cash += pnl
                            cash -= abs(pnl) * commission
                            
                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': current_date,
                                'action': position_type,
                                'entry_price': entry_price,
                                'exit_price': current_spread,
                                'pnl': pnl,
                                'holding_period': days_held,
                                'exit_reason': exit_reason
                            })
                            
                            position = 0
                            entry_price = None
                            position_type = None
                            days_held = 0
                
                current_equity = cash
                if position != 0:
                    unrealized_pnl = 0
                    if position_type == 'LONG':
                        unrealized_pnl = (current_spread - entry_price) * (cash * position_size) / entry_price
                    else:
                        unrealized_pnl = (entry_price - current_spread) * (cash * position_size) / entry_price
                    current_equity += unrealized_pnl
                
                equity_curve.append(current_equity)
                continue
            
            # Normal trading logic
            if position != 0:
                days_held += 1
            
            # Entry logic
            if position == 0:
                current_volatility = spread.rolling(5).std().iloc[i] if i >= 5 else 0
                avg_volatility = spread.rolling(20).std().mean()
                
                if current_volatility < avg_volatility * 1.5:
                    if current_z > entry_threshold:
                        position_type = 'SHORT'
                        entry_price = current_spread
                        entry_date = current_date
                        position = -1
                        days_held = 0
                        
                    elif current_z < -entry_threshold:
                        position_type = 'LONG'
                        entry_price = current_spread
                        entry_date = current_date
                        position = 1
                        days_held = 0
            
            else:  # Exit logic
                should_exit = False
                exit_reason = ""
                
                if abs(current_z) < exit_threshold:
                    should_exit = True
                    exit_reason = f"Target reached (Z={current_z:.2f})"
                elif days_held >= max_hold_days:
                    should_exit = True
                    exit_reason = f"Max hold ({days_held}d)"
                elif days_held >= 2:
                    if position_type == 'LONG' and current_z < -4.0:
                        should_exit = True
                        exit_reason = f"Stop loss (Z={current_z:.2f})"
                    elif position_type == 'SHORT' and current_z > 4.0:
                        should_exit = True
                        exit_reason = f"Stop loss (Z={current_z:.2f})"
                    elif position_type == 'LONG' and current_z > -0.3:
                        should_exit = True
                        exit_reason = f"Early profit (Z={current_z:.2f})"
                    elif position_type == 'SHORT' and current_z < 0.3:
                        should_exit = True
                        exit_reason = f"Early profit (Z={current_z:.2f})"
                
                if should_exit:
                    if position_type == 'LONG':
                        pnl = (current_spread - entry_price) * (cash * position_size) / entry_price
                    else:
                        pnl = (entry_price - current_spread) * (cash * position_size) / entry_price
                    
                    cash += pnl
                    cash -= abs(pnl) * commission
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'action': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_spread,
                        'pnl': pnl,
                        'holding_period': days_held,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    entry_price = None
                    position_type = None
                    days_held = 0
            
            # Calculate current equity
            current_equity = cash
            if position != 0:
                unrealized_pnl = 0
                if position_type == 'LONG':
                    unrealized_pnl = (current_spread - entry_price) * (cash * position_size) / entry_price
                else:
                    unrealized_pnl = (entry_price - current_spread) * (cash * position_size) / entry_price
                current_equity += unrealized_pnl
            
            equity_curve.append(current_equity)
        
        # Calculate all requested metrics
        return self.calculate_all_metrics(equity_curve, trades, initial_cash, commission)
    
    def calculate_all_metrics(self, equity_curve, trades, initial_cash, commission):
        """Calculate all requested metrics"""
        equity_values = np.array(equity_curve)
        equity_returns = pd.Series(equity_values).pct_change().dropna()
        
        # RETURN METRICS
        total_return = ((equity_values[-1] - initial_cash) / initial_cash) * 100
        
        years = len(equity_curve) / 252
        cagr = ((equity_values[-1] / initial_cash) ** (1/years) - 1) * 100 if years > 0 else 0
        
        monthly_return = total_return / 12
        
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            expectancy_per_trade = trades_df['pnl'].mean()
        else:
            profit_factor = 0
            expectancy_per_trade = 0
        
        # RISK METRICS
        if len(equity_returns) > 0 and equity_returns.std() > 0:
            sharpe_ratio = (equity_returns.mean() * 252) / (equity_returns.std() * np.sqrt(252))
            
            # Sortino Ratio (downside deviation)
            downside_returns = equity_returns[equity_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (equity_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))
            else:
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Calmar Ratio (return / max drawdown)
        max_dd = self.calculate_max_drawdown(equity_values)
        calmar_ratio = total_return / abs(max_dd) if max_dd != 0 else 0
        
        # Average drawdown
        avg_drawdown = self.calculate_avg_drawdown(equity_values)
        
        # Max drawdown duration
        max_dd_duration = self.calculate_max_drawdown_duration(equity_values)
        
        # TRADE METRICS
        total_trades = len(trades)
        
        if total_trades > 0:
            trades_df = pd.DataFrame(trades)
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
        
        trades_per_year = total_trades / years if years > 0 else 0
        
        # COST METRICS
        total_fees = total_trades * 2 * (initial_cash * 0.3 * commission)
        
        if total_trades > 0 and len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            fees_percentage = (total_fees / abs(gross_profit) * 100) if gross_profit != 0 else 0
        else:
            fees_percentage = 0
        
        # EFFICIENCY METRICS
        time_in_market = self.calculate_time_in_market(equity_curve, trades)
        return_drawdown = total_return / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'monthly_return': monthly_return,
            'profit_factor': profit_factor,
            'expectancy_per_trade': expectancy_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_holding_period': avg_holding_period,
            'trades_per_year': trades_per_year,
            'total_fees_paid': total_fees,
            'fees_percentage': fees_percentage,
            'time_in_market': time_in_market,
            'return_drawdown': return_drawdown
        }
    
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
    
    def calculate_avg_drawdown(self, equity_values):
        """Calculate average drawdown"""
        peak = equity_values[0]
        drawdowns = []
        in_drawdown = False
        current_dd = 0
        
        for value in equity_values:
            if value > peak:
                if in_drawdown and current_dd > 0:
                    drawdowns.append(current_dd)
                peak = value
                in_drawdown = False
                current_dd = 0
            else:
                dd = (peak - value) / peak * 100
                current_dd = max(current_dd, dd)
                in_drawdown = True
        
        if in_drawdown and current_dd > 0:
            drawdowns.append(current_dd)
        
        return np.mean(drawdowns) if drawdowns else 0
    
    def calculate_max_drawdown_duration(self, equity_values):
        """Calculate maximum drawdown duration in days"""
        peak = equity_values[0]
        peak_date = 0
        max_duration = 0
        current_duration = 0
        
        for i, value in enumerate(equity_values):
            if value > peak:
                peak = value
                peak_date = i
                current_duration = 0
            else:
                current_duration = i - peak_date
                max_duration = max(max_duration, current_duration)
        
        return max_duration
    
    def calculate_time_in_market(self, equity_curve, trades):
        """Calculate percentage of time in market"""
        if not trades:
            return 0
        
        total_periods = len(equity_curve)
        market_periods = 0
        
        for trade in trades:
            if 'holding_period' in trade:
                market_periods += trade['holding_period']
        
        return (market_periods / total_periods * 100) if total_periods > 0 else 0
    
    def is_time_allowed(self, timestamp, start_time=None, end_time=None, days=None):
        """Check if timestamp is allowed based on time filter"""
        # Convert to datetime if it's a date
        if isinstance(timestamp, pd.Timestamp):
            dt = timestamp
        elif hasattr(timestamp, 'time'):
            dt = timestamp
        else:
            # If it's a date, convert to datetime at midnight
            dt = pd.Timestamp.combine(timestamp, pd.Timestamp.min.time())
        
        # Check day filter
        if days is not None:
            if dt.dayofweek not in days:
                return False
        
        # Check time filter
        if start_time is not None and end_time is not None:
            current_time = dt.time()
            
            # Handle overnight sessions (e.g., 15:30 to 9:15)
            if start_time > end_time:
                if current_time >= start_time or current_time <= end_time:
                    return True
                else:
                    return False
            else:
                if start_time <= current_time <= end_time:
                    return True
                else:
                    return False
        
        return True
    
    def display_comprehensive_results(self, results_df):
        """Display comprehensive backtesting results"""
        print("\n" + "="*100)
        print("📊 COMPREHENSIVE TIME-BASED BACKTESTING RESULTS")
        print("="*100)
        
        # Find best configuration by total return
        best_return = results_df.loc[results_df['total_return'].idxmax()]
        
        print(f"\n🏆 BEST CONFIGURATION BY TOTAL RETURN: {best_return['configuration']}")
        print("-" * 100)
        
        # Display all metrics for best configuration
        self.display_all_metrics(best_return)
        
        # Display comparison table
        print(f"\n📊 CONFIGURATION COMPARISON")
        print("-" * 100)
        
        # Top 5 configurations by return
        top_configs = results_df.nlargest(5, 'total_return')
        
        print(f"{'Configuration':<20} {'Return':<10} {'Sharpe':<8} {'Win Rate':<10} {'Trades':<8} {'Max DD':<8}")
        print("-" * 100)
        
        for _, config in top_configs.iterrows():
            print(f"{config['configuration']:<20} {config['total_return']:+8.2f}% {config['sharpe_ratio']:+7.3f} {config['win_rate']:8.1f}% {config['total_trades']:7d} {config['max_drawdown']:7.2f}%")
        
        # Display detailed metrics for all configurations
        print(f"\n📈 DETAILED METRICS FOR ALL CONFIGURATIONS")
        print("-" * 100)
        
        for _, config in results_df.iterrows():
            print(f"\n🎯 {config['configuration']}")
            print("-" * 50)
            self.display_all_metrics(config)
    
    def display_all_metrics(self, config):
        """Display all metrics for a configuration"""
        print(f"📈 RETURN METRICS")
        print(f"   Total Return:                     {config['total_return']:+.2f}%")
        print(f"   CAGR:                             {config['cagr']:+.2f}%")
        print(f"   Monthly Return:                   {config['monthly_return']:+.2f}%")
        print(f"   Profit Factor:                    {config['profit_factor']:.2f}")
        print(f"   Expectancy per Trade:             ${config['expectancy_per_trade']:+.2f}")
        
        print(f"\n⚠️  RISK METRICS")
        print(f"   Sharpe Ratio:                     {config['sharpe_ratio']:+.3f}")
        print(f"   Sortino Ratio:                    {config['sortino_ratio']:+.3f}")
        print(f"   Calmar Ratio:                     {config['calmar_ratio']:+.3f}")
        print(f"   Max Drawdown:                     {config['max_drawdown']:+.2f}%")
        print(f"   Avg Drawdown:                     {config['avg_drawdown']:+.2f}%")
        print(f"   Max Drawdown Duration:            {config['max_drawdown_duration']} days")
        
        print(f"\n🔄 TRADE METRICS")
        print(f"   Total Trades:                     {config['total_trades']}")
        print(f"   Win Rate:                         {config['win_rate']:.1f}%")
        print(f"   Avg Win:                          ${config['avg_win']:+,.2f}")
        print(f"   Avg Loss:                         ${config['avg_loss']:+,.2f}")
        print(f"   Largest Win:                      ${config['largest_win']:+,.2f}")
        print(f"   Largest Loss:                     ${config['largest_loss']:+,.2f}")
        print(f"   Avg Holding Period:               {config['avg_holding_period']:.1f} days")
        print(f"   Trades per Year:                  {config['trades_per_year']:.1f}")
        
        print(f"\n💸 COST METRICS")
        print(f"   Total Fees Paid:                  ${config['total_fees_paid']:,.2f}")
        print(f"   Fees as % of Gross Profit:        {config['fees_percentage']:.2f}%")
        
        print(f"\n⚡ EFFICIENCY")
        print(f"   Time in Market:                   {config['time_in_market']:.1f}%")
        print(f"   Return / Drawdown:                {config['return_drawdown']:.3f}")
    
    def generate_detailed_report(self, results_df):
        """Generate detailed CSV report"""
        # Create detailed report
        report_data = []
        
        # Header
        report_data.append([
            'Configuration', 'Total_Return_%', 'CAGR_%', 'Monthly_Return_%', 'Profit_Factor', 
            'Expectancy_Per_Trade', 'Sharpe_Ratio', 'Sortino_Ratio', 'Calmar_Ratio',
            'Max_Drawdown_%', 'Avg_Drawdown_%', 'Max_Drawdown_Duration_Days',
            'Total_Trades', 'Win_Rate_%', 'Avg_Win_$', 'Avg_Loss_$', 
            'Largest_Win_$', 'Largest_Loss_$', 'Avg_Holding_Period_Days', 'Trades_Per_Year',
            'Total_Fees_Paid_$', 'Fees_%_Gross_Profit', 'Time_in_Market_%', 'Return_Drawdown'
        ])
        
        # Data rows
        for _, config in results_df.iterrows():
            report_data.append([
                config['configuration'],
                f"{config['total_return']:.2f}",
                f"{config['cagr']:.2f}",
                f"{config['monthly_return']:.2f}",
                f"{config['profit_factor']:.2f}",
                f"{config['expectancy_per_trade']:.2f}",
                f"{config['sharpe_ratio']:.3f}",
                f"{config['sortino_ratio']:.3f}",
                f"{config['calmar_ratio']:.3f}",
                f"{config['max_drawdown']:.2f}",
                f"{config['avg_drawdown']:.2f}",
                str(config['max_drawdown_duration']),
                str(config['total_trades']),
                f"{config['win_rate']:.1f}",
                f"{config['avg_win']:.2f}",
                f"{config['avg_loss']:.2f}",
                f"{config['largest_win']:.2f}",
                f"{config['largest_loss']:.2f}",
                f"{config['avg_holding_period']:.1f}",
                f"{config['trades_per_year']:.1f}",
                f"{config['total_fees_paid']:.2f}",
                f"{config['fees_percentage']:.2f}",
                f"{config['time_in_market']:.1f}",
                f"{config['return_drawdown']:.3f}"
            ])
        
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data[1:], columns=report_data[0])
        report_df.to_csv('comprehensive_time_backtest_report.csv', index=False)
        
        print(f"\n📊 Detailed report exported to: comprehensive_time_backtest_report.csv")
        return 'comprehensive_time_backtest_report.csv'

# Main execution
if __name__ == "__main__":
    print("📊 COMPREHENSIVE TIME-BASED BACKTESTING")
    print("="*100)
    print("🎯 Running complete backtesting with all requested metrics...")
    
    # Initialize comprehensive backtest
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    backtest = ComprehensiveTimeBacktest(data_folder)
    
    # Run comprehensive backtest
    results_df = backtest.run_comprehensive_backtest()
    
    print("\n🎉 COMPREHENSIVE BACKTESTING COMPLETED!")
    print("="*100)
    print("📁 Files generated:")
    print("  • comprehensive_time_backtest_report.csv - Complete metrics report")
    print("✅ All requested metrics calculated and analyzed!")
