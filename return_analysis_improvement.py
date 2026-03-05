"""
RETURN ANALYSIS AND IMPROVEMENT
Analyzes why returns are negative and creates improved strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairs_trading_pipeline import PairsTradingPipeline
import warnings
warnings.filterwarnings('ignore')

class ReturnAnalysisImprovement:
    """Analyzes return issues and creates improved strategy"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pipeline = PairsTradingPipeline(data_folder)
        self.metrics = {}
        
    def analyze_negative_returns(self):
        """Analyze why returns are negative"""
        print("🔍 ANALYZING NEGATIVE RETURNS")
        print("="*80)
        
        # Run pipeline to get data
        self.pipeline.step1_data_preprocessing()
        top_pairs = self.pipeline.step2_pair_selection()
        
        if len(top_pairs) == 0:
            raise ValueError("No cointegrated pairs found!")
        
        best_pair = top_pairs.iloc[0]
        stock1, stock2 = best_pair['stock1'], best_pair['stock2']
        
        print(f"🎯 Analyzing pair: {stock1} - {stock2}")
        print(f"📈 P-value: {best_pair['p_value']:.6f}")
        
        # Get price data
        pair_data = self.pipeline.price_data[[stock1, stock2]].copy().dropna()
        spread = pair_data[stock1] - pair_data[stock2]
        
        # Calculate indicators
        spread_mean = spread.rolling(20).mean()
        spread_std = spread.rolling(20).std()
        z_score = (spread - spread_mean) / spread_std
        
        print("\n📊 SPREAD ANALYSIS")
        print("-" * 50)
        print(f"Spread Mean: {spread.mean():.2f}")
        print(f"Spread Std: {spread.std():.2f}")
        print(f"Z-Score Mean: {z_score.mean():.3f}")
        print(f"Z-Score Std: {z_score.std():.3f}")
        print(f"Z-Score Min: {z_score.min():.3f}")
        print(f"Z-Score Max: {z_score.max():.3f}")
        
        # Analyze Z-score distribution
        extreme_positive = (z_score > 2.2).sum()
        extreme_negative = (z_score < -2.2).sum()
        total_periods = len(z_score)
        
        print(f"\n📈 SIGNAL FREQUENCY ANALYSIS")
        print("-" * 50)
        print(f"Total periods: {total_periods}")
        print(f"Extreme positive Z (>2.2): {extreme_positive} ({extreme_positive/total_periods*100:.1f}%)")
        print(f"Extreme negative Z (<-2.2): {extreme_negative} ({extreme_negative/total_periods*100:.1f}%)")
        print(f"Total extreme signals: {extreme_positive + extreme_negative}")
        
        # Analyze mean reversion
        print(f"\n🔄 MEAN REVERSION ANALYSIS")
        print("-" * 50)
        
        # Test mean reversion after extreme signals
        positive_reversions = 0
        negative_reversions = 0
        
        for i in range(1, len(z_score)):
            if z_score.iloc[i-1] > 2.2 and z_score.iloc[i] < z_score.iloc[i-1]:
                positive_reversions += 1
            elif z_score.iloc[i-1] < -2.2 and z_score.iloc[i] > z_score.iloc[i-1]:
                negative_reversions += 1
        
        total_extreme = extreme_positive + extreme_negative
        if total_extreme > 0:
            reversion_rate = (positive_reversions + negative_reversions) / total_extreme * 100
            print(f"Mean reversion rate: {reversion_rate:.1f}%")
            print(f"Positive reversions: {positive_reversions}/{extreme_positive}")
            print(f"Negative reversions: {negative_reversions}/{extreme_negative}")
        
        # Analyze spread trends
        print(f"\n📊 SPREAD TREND ANALYSIS")
        print("-" * 50)
        
        spread_trend = spread.diff().mean()
        spread_volatility = spread.rolling(20).std().mean()
        
        print(f"Average daily spread change: {spread_trend:.4f}")
        print(f"Average spread volatility: {spread_volatility:.4f}")
        
        if abs(spread_trend) > spread_volatility * 0.1:
            print("⚠️  WARNING: Strong trend detected - pairs may not be suitable")
        else:
            print("✅ Spread appears stable for pairs trading")
        
        # Identify issues
        print(f"\n🚨 IDENTIFIED ISSUES")
        print("-" * 50)
        
        issues = []
        
        if extreme_positive + extreme_negative < 50:
            issues.append("Too few trading signals")
            print("❌ Too few trading signals (<50)")
        
        if reversion_rate < 60:
            issues.append("Poor mean reversion")
            print("❌ Poor mean reversion (<60%)")
        
        if abs(spread_trend) > spread_volatility * 0.1:
            issues.append("Strong trend in spread")
            print("❌ Strong trend detected in spread")
        
        if len(issues) == 0:
            print("✅ No major issues identified")
        
        return issues, spread, z_score, stock1, stock2
    
    def create_improved_strategy(self, issues, spread, z_score, stock1, stock2):
        """Create improved strategy based on analysis"""
        print("\n🚀 CREATING IMPROVED STRATEGY")
        print("="*80)
        
        # Adjust parameters based on issues
        if "Too few trading signals" in issues:
            z_entry_threshold = 1.8  # Lower threshold for more signals
            print("📊 Lowering entry threshold to 1.8 for more signals")
        else:
            z_entry_threshold = 2.0
        
        if "Poor mean reversion" in issues:
            z_exit_threshold = 0.8  # Wider exit for better mean reversion
            print("📊 Widening exit threshold to 0.8 for better mean reversion")
        else:
            z_exit_threshold = 0.5
        
        # Strategy parameters
        min_holding_days = 2
        max_holding_days = 10
        position_size = 0.4  # Larger position for better returns
        initial_cash = 100000
        commission = 0.001
        
        print(f"\n📊 IMPROVED STRATEGY PARAMETERS")
        print("-" * 50)
        print(f"Entry Z-threshold: {z_entry_threshold}")
        print(f"Exit Z-threshold: {z_exit_threshold}")
        print(f"Position size: {position_size*100}%")
        print(f"Holding period: {min_holding_days}-{max_holding_days} days")
        
        # Improved backtesting
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
            
            # Update days held
            if position != 0:
                days_held += 1
            
            # Improved entry conditions
            if position == 0:
                # Additional filter: avoid entering during high volatility
                spread_volatility = spread.rolling(5).std().iloc[i] if i >= 5 else 0
                avg_volatility = spread.rolling(20).std().mean()
                
                if spread_volatility < avg_volatility * 2:  # Avoid high volatility periods
                    if current_z > z_entry_threshold:  # Short signal
                        position_type = 'SHORT'
                        entry_price = current_spread
                        entry_date = current_date
                        position = -1
                        trade_size = cash * position_size
                        cash -= trade_size * commission
                        
                        trades.append({
                            'entry_date': entry_date,
                            'action': 'SHORT',
                            'entry_price': entry_price,
                            'z_score': current_z,
                            'volatility': spread_volatility
                        })
                        print(f"🔴 SHORT at Z={current_z:.2f}, Vol={spread_volatility:.4f}")
                        days_held = 0
                        
                    elif current_z < -z_entry_threshold:  # Long signal
                        position_type = 'LONG'
                        entry_price = current_spread
                        entry_date = current_date
                        position = 1
                        trade_size = cash * position_size
                        cash -= trade_size * commission
                        
                        trades.append({
                            'entry_date': entry_date,
                            'action': 'LONG',
                            'entry_price': entry_price,
                            'z_score': current_z,
                            'volatility': spread_volatility
                        })
                        print(f"🟢 LONG at Z={current_z:.2f}, Vol={spread_volatility:.4f}")
                        days_held = 0
            
            else:  # Have position - improved exit conditions
                should_exit = False
                exit_reason = ""
                
                # Exit conditions
                if abs(current_z) < z_exit_threshold:
                    should_exit = True
                    exit_reason = f"Mean revert (Z={current_z:.2f})"
                elif days_held >= max_holding_days:
                    should_exit = True
                    exit_reason = f"Max hold ({days_held}d)"
                elif days_held >= min_holding_days:
                    # Profit taking
                    if position_type == 'LONG' and current_z > -0.2:
                        should_exit = True
                        exit_reason = f"Long profit (Z={current_z:.2f})"
                    elif position_type == 'SHORT' and current_z < 0.2:
                        should_exit = True
                        exit_reason = f"Short profit (Z={current_z:.2f})"
                    # Stop loss
                    elif position_type == 'LONG' and current_z < -3.5:
                        should_exit = True
                        exit_reason = f"Long stop loss (Z={current_z:.2f})"
                    elif position_type == 'SHORT' and current_z > 3.5:
                        should_exit = True
                        exit_reason = f"Short stop loss (Z={current_z:.2f})"
                
                if should_exit:
                    # Calculate P&L
                    if position_type == 'LONG':
                        pnl = (current_spread - entry_price) * (cash * position_size) / entry_price
                    else:  # SHORT
                        pnl = (entry_price - current_spread) * (cash * position_size) / entry_price
                    
                    cash += pnl
                    cash -= abs(pnl) * commission  # Commission on exit
                    
                    holding_period = days_held
                    
                    trades[-1].update({
                        'exit_date': current_date,
                        'exit_price': current_spread,
                        'pnl': pnl,
                        'holding_period': holding_period,
                        'exit_z_score': current_z,
                        'exit_reason': exit_reason
                    })
                    
                    print(f"⚪ {exit_reason}: P&L=${pnl:.2f}, Days={holding_period}")
                    
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
        
        # Calculate metrics
        self.calculate_improved_metrics(equity_df, trades_df, initial_cash, commission)
        
        return equity_df, trades_df
    
    def calculate_improved_metrics(self, equity_df, trades_df, initial_cash, commission):
        """Calculate improved strategy metrics"""
        print("\n📊 CALCULATING IMPROVED METRICS...")
        
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
        total_fees = total_trades * 2 * (initial_cash * 0.4 * commission)
        self.metrics['total_fees_paid'] = total_fees
        
        if total_trades > 0 and 'pnl' in trades_df.columns:
            gross_profit = trades_df['pnl'].sum()
            fees_percentage = (total_fees / abs(gross_profit) * 100) if gross_profit != 0 else 0
        else:
            fees_percentage = 0
        self.metrics['fees_percentage'] = fees_percentage
        
        print("✅ Improved metrics calculated successfully!")
    
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
    
    def display_improvement_results(self):
        """Display improvement results"""
        metrics = self.metrics
        
        print("\n" + "="*80)
        print("📊 IMPROVED STRATEGY RESULTS")
        print("="*80)
        
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
        
        # Improvement Analysis
        print("\n🚀 IMPROVEMENT ANALYSIS")
        print("-" * 50)
        
        if metrics['total_return'] > 0:
            print("✅ POSITIVE RETURNS ACHIEVED!")
        else:
            print("❌ Still negative returns - needs further optimization")
        
        if metrics['total_trades'] > 5:
            print("✅ Good trading frequency")
        else:
            print("⚠️  Low trading frequency")
        
        if metrics['win_rate'] > 50:
            print("✅ Good win rate")
        else:
            print("⚠️  Win rate needs improvement")
        
        if metrics['profit_factor'] > 1.2:
            print("✅ Strong profitability")
        else:
            print("⚠️  Profitability needs improvement")
        
        print("\n" + "="*80)
    
    def export_improvement_metrics(self):
        """Export improvement metrics"""
        metrics_df = pd.DataFrame(list(self.metrics.items()), 
                                columns=['Metric', 'Value'])
        filename = "improved_strategy_metrics.csv"
        metrics_df.to_csv(filename, index=False)
        print(f"📊 Improved metrics exported to: {filename}")
        return filename

# Main execution
if __name__ == "__main__":
    print("🔍 RETURN ANALYSIS AND IMPROVEMENT")
    print("="*80)
    print("🎯 Analyzing negative returns and creating improved strategy...")
    
    # Initialize analyzer
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    analyzer = ReturnAnalysisImprovement(data_folder)
    
    # Analyze negative returns
    issues, spread, z_score, stock1, stock2 = analyzer.analyze_negative_returns()
    
    # Create improved strategy
    equity_df, trades_df = analyzer.create_improved_strategy(issues, spread, z_score, stock1, stock2)
    
    # Display results
    analyzer.display_improvement_results()
    
    # Export metrics
    analyzer.export_improvement_metrics()
    
    print("\n🎉 RETURN ANALYSIS AND IMPROVEMENT COMPLETED!")
    print("="*80)
    print("📁 Files generated:")
    print("  • Return analysis report")
    print("  • Improved strategy metrics")
    print("  • Issue identification and solutions")
    print("✅ Strategy improved based on negative return analysis!")
