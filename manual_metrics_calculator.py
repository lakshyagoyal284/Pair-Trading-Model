"""
MANUAL COMPREHENSIVE METRICS CALCULATOR
Calculates all requested backtesting metrics manually to avoid library issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairs_trading_pipeline import PairsTradingPipeline
import warnings
warnings.filterwarnings('ignore')

class ManualMetricsCalculator:
    """Manual calculation of comprehensive backtesting metrics"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pipeline = PairsTradingPipeline(data_folder)
        self.metrics = {}
        
    def run_manual_backtest(self, initial_cash=100000, commission=0.001):
        """Run manual backtest with detailed trade tracking"""
        print("🚀 RUNNING MANUAL COMPREHENSIVE BACKTEST")
        print("="*80)
        
        # Run pipeline to get best pair
        self.pipeline.step1_data_preprocessing()
        top_pairs = self.pipeline.step2_pair_selection()
        
        if len(top_pairs) == 0:
            raise ValueError("No cointegrated pairs found!")
        
        best_pair = top_pairs.iloc[0]
        stock1, stock2 = best_pair['stock1'], best_pair['stock2']
        
        print(f"🎯 Selected pair: {stock1} - {stock2}")
        print(f"📈 P-value: {best_pair['p_value']:.6f}")
        
        # Get price data
        pair_data = self.pipeline.price_data[[stock1, stock2]].copy().dropna()
        spread = pair_data[stock1] - pair_data[stock2]
        
        # Calculate indicators
        spread_mean = spread.rolling(20).mean()
        spread_std = spread.rolling(20).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Manual backtesting
        cash = initial_cash
        position = 0
        trades = []
        equity_curve = []
        entry_price = None
        entry_date = None
        position_type = None
        
        print("🔄 Executing manual backtest...")
        
        for i in range(len(spread)):
            current_date = spread.index[i]
            current_spread = spread.iloc[i]
            current_z = z_score.iloc[i]
            
            # Trading logic
            if position == 0:  # No position
                if current_z > 2.0:  # Short signal
                    position_type = 'SHORT'
                    entry_price = current_spread
                    entry_date = current_date
                    position = -1
                    trade_size = cash * 0.5  # Use 50% of cash
                    cash -= trade_size * commission
                    trades.append({
                        'entry_date': entry_date,
                        'action': 'SHORT',
                        'entry_price': entry_price,
                        'z_score': current_z
                    })
                    print(f"🔴 SHORT at Z-score: {current_z:.2f}")
                    
                elif current_z < -2.0:  # Long signal
                    position_type = 'LONG'
                    entry_price = current_spread
                    entry_date = current_date
                    position = 1
                    trade_size = cash * 0.5
                    cash -= trade_size * commission
                    trades.append({
                        'entry_date': entry_date,
                        'action': 'LONG',
                        'entry_price': entry_price,
                        'z_score': current_z
                    })
                    print(f"🟢 LONG at Z-score: {current_z:.2f}")
            
            else:  # Have position
                if abs(current_z) < 0.5:  # Close signal
                    # Calculate P&L
                    if position_type == 'LONG':
                        pnl = (current_spread - entry_price) * (cash * 0.5) / entry_price
                    else:  # SHORT
                        pnl = (entry_price - current_spread) * (cash * 0.5) / entry_price
                    
                    cash += pnl
                    cash -= abs(pnl) * commission  # Commission on exit
                    
                    holding_period = (current_date - entry_date).days
                    
                    trades[-1].update({
                        'exit_date': current_date,
                        'exit_price': current_spread,
                        'pnl': pnl,
                        'holding_period': holding_period,
                        'exit_z_score': current_z
                    })
                    
                    print(f"⚪ CLOSE at Z-score: {current_z:.2f}, P&L: ${pnl:.2f}")
                    
                    position = 0
                    entry_price = None
                    position_type = None
            
            # Calculate current equity
            current_equity = cash
            if position != 0:
                unrealized_pnl = 0
                if position_type == 'LONG':
                    unrealized_pnl = (current_spread - entry_price) * (cash * 0.5) / entry_price
                else:
                    unrealized_pnl = (entry_price - current_spread) * (cash * 0.5) / entry_price
                current_equity += unrealized_pnl
            
            equity_curve.append({
                'date': current_date,
                'equity': current_equity,
                'cash': cash,
                'position': position,
                'spread': current_spread,
                'z_score': current_z
            })
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        trades_df = pd.DataFrame(trades)
        
        # Calculate all metrics
        self.calculate_all_metrics(equity_df, trades_df, initial_cash, commission)
        
        return equity_df, trades_df
    
    def calculate_all_metrics(self, equity_df, trades_df, initial_cash, commission):
        """Calculate all comprehensive metrics"""
        print("\n📊 CALCULATING COMPREHENSIVE METRICS...")
        
        # Extract data
        equity_values = equity_df['equity'].values
        equity_returns = equity_df['equity'].pct_change().dropna()
        
        # RETURN METRICS
        print("📈 Calculating Return Metrics...")
        
        # Total Return
        total_return = ((equity_values[-1] - initial_cash) / initial_cash) * 100
        self.metrics['total_return'] = total_return
        
        # CAGR
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        years = (end_date - start_date).days / 365.25
        cagr = ((equity_values[-1] / initial_cash) ** (1/years) - 1) * 100
        self.metrics['cagr'] = cagr
        
        # Monthly Return
        monthly_returns = equity_df['equity'].resample('M').last().pct_change().dropna()
        avg_monthly_return = monthly_returns.mean() * 100
        self.metrics['monthly_return'] = avg_monthly_return
        
        # Profit Factor
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = 0
        self.metrics['profit_factor'] = profit_factor
        
        # Expectancy per Trade
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            expectancy_per_trade = trades_df['pnl'].mean()
        else:
            expectancy_per_trade = 0
        self.metrics['expectancy_per_trade'] = expectancy_per_trade
        
        # RISK METRICS
        print("⚠️  Calculating Risk Metrics...")
        
        # Sharpe Ratio
        if len(equity_returns) > 0:
            sharpe_ratio = (equity_returns.mean() * 252) / (equity_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        self.metrics['sharpe_ratio'] = sharpe_ratio
        
        # Sortino Ratio
        if len(equity_returns) > 0:
            downside_returns = equity_returns[equity_returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252)
                sortino_ratio = (equity_returns.mean() * 252) / downside_std
            else:
                sortino_ratio = float('inf')
        else:
            sortino_ratio = 0
        self.metrics['sortino_ratio'] = sortino_ratio
        
        # Calmar Ratio
        max_dd = self.calculate_max_drawdown(equity_values)
        calmar_ratio = cagr / abs(max_dd) if max_dd != 0 else 0
        self.metrics['calmar_ratio'] = calmar_ratio
        
        # Max Drawdown
        self.metrics['max_drawdown'] = max_dd
        
        # Average Drawdown
        drawdown_series = self.calculate_drawdown_series(equity_values)
        avg_drawdown = drawdown_series[drawdown_series < 0].mean() if len(drawdown_series[drawdown_series < 0]) > 0 else 0
        self.metrics['avg_drawdown'] = avg_drawdown
        
        # Max Drawdown Duration
        max_dd_duration = self.calculate_max_drawdown_duration(drawdown_series)
        self.metrics['max_drawdown_duration'] = max_dd_duration
        
        # TRADE METRICS
        print("🔄 Calculating Trade Metrics...")
        
        total_trades = len(trades_df)
        self.metrics['total_trades'] = total_trades
        
        if total_trades > 0 and 'pnl' in trades_df.columns:
            # Win Rate
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = (len(winning_trades) / total_trades) * 100
            self.metrics['win_rate'] = win_rate
            
            # Average Win and Loss
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            losing_trades = trades_df[trades_df['pnl'] < 0]
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            self.metrics['avg_win'] = avg_win
            self.metrics['avg_loss'] = avg_loss
            
            # Largest Win and Loss
            largest_win = trades_df['pnl'].max()
            largest_loss = trades_df['pnl'].min()
            
            self.metrics['largest_win'] = largest_win
            self.metrics['largest_loss'] = largest_loss
            
            # Average Holding Period
            if 'holding_period' in trades_df.columns:
                avg_holding_period = trades_df['holding_period'].mean()
            else:
                avg_holding_period = 0
            self.metrics['avg_holding_period'] = avg_holding_period
            
        else:
            self.metrics['win_rate'] = 0
            self.metrics['avg_win'] = 0
            self.metrics['avg_loss'] = 0
            self.metrics['largest_win'] = 0
            self.metrics['largest_loss'] = 0
            self.metrics['avg_holding_period'] = 0
        
        # Trades per Year
        trades_per_year = total_trades / years if years > 0 else 0
        self.metrics['trades_per_year'] = trades_per_year
        
        # COST METRICS
        print("💸 Calculating Cost Metrics...")
        
        # Total Fees Paid
        total_fees = total_trades * 2 * (initial_cash * 0.5 * commission)  # Entry + exit fees
        self.metrics['total_fees_paid'] = total_fees
        
        # Fees as % of Gross Profit
        if total_trades > 0 and 'pnl' in trades_df.columns:
            gross_profit = trades_df['pnl'].sum()
            fees_percentage = (total_fees / abs(gross_profit) * 100) if gross_profit != 0 else 0
        else:
            fees_percentage = 0
        self.metrics['fees_percentage'] = fees_percentage
        
        # EFFICIENCY METRICS
        print("⚡ Calculating Efficiency Metrics...")
        
        # Time in Market
        if total_trades > 0 and 'holding_period' in trades_df.columns:
            total_time_in_market = trades_df['holding_period'].sum()
            time_in_market = (total_time_in_market / (years * 365)) * 100
        else:
            time_in_market = 0
        self.metrics['time_in_market'] = time_in_market
        
        # Return / Drawdown Ratio
        return_drawdown_ratio = abs(total_return / max_dd) if max_dd != 0 else float('inf')
        self.metrics['return_drawdown_ratio'] = return_drawdown_ratio
        
        print("✅ All metrics calculated successfully!")
    
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
    
    def calculate_drawdown_series(self, equity_values):
        """Calculate drawdown series"""
        peak = equity_values[0]
        drawdowns = []
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            drawdowns.append(dd)
        
        return pd.Series(drawdowns)
    
    def calculate_max_drawdown_duration(self, drawdown_series):
        """Calculate maximum drawdown duration in days"""
        in_drawdown = False
        current_duration = 0
        max_duration = 0
        
        for dd in drawdown_series:
            if dd > 0:
                if not in_drawdown:
                    in_drawdown = True
                current_duration += 1
                if current_duration > max_duration:
                    max_duration = current_duration
            else:
                in_drawdown = False
                current_duration = 0
        
        return max_duration
    
    def display_comprehensive_results(self):
        """Display all metrics in professional format"""
        metrics = self.metrics
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE BACKTESTING METRICS")
        print("="*80)
        
        # RETURN METRICS
        print("\n📈 RETURN METRICS")
        print("-" * 50)
        print(f"Total Return:                     {metrics['total_return']:+.2f}%")
        print(f"CAGR:                             {metrics['cagr']:+.2f}%")
        print(f"Monthly Return:                   {metrics['monthly_return']:+.2f}%")
        print(f"Profit Factor:                    {metrics['profit_factor']:.2f}")
        print(f"Expectancy per Trade:             ${metrics['expectancy_per_trade']:+.2f}")
        
        # RISK METRICS
        print("\n⚠️  RISK METRICS")
        print("-" * 50)
        print(f"Sharpe Ratio:                     {metrics['sharpe_ratio']:+.3f}")
        print(f"Sortino Ratio:                    {metrics['sortino_ratio']:+.3f}")
        print(f"Calmar Ratio:                     {metrics['calmar_ratio']:+.3f}")
        print(f"Max Drawdown:                     {metrics['max_drawdown']:+.2f}%")
        print(f"Avg Drawdown:                     {metrics['avg_drawdown']:+.2f}%")
        print(f"Max Drawdown Duration:            {metrics['max_drawdown_duration']} days")
        
        # TRADE METRICS
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
        
        # COST METRICS
        print("\n💸 COST METRICS")
        print("-" * 50)
        print(f"Total Fees Paid:                  ${metrics['total_fees_paid']:,.2f}")
        print(f"Fees as % of Gross Profit:        {metrics['fees_percentage']:.2f}%")
        
        # EFFICIENCY METRICS
        print("\n⚡ EFFICIENCY METRICS")
        print("-" * 50)
        print(f"Time in Market:                   {metrics['time_in_market']:.1f}%")
        print(f"Return / Drawdown:                {metrics['return_drawdown_ratio']:.2f}")
        
        print("\n" + "="*80)
        
        # Performance Analysis
        self.print_performance_analysis()
    
    def print_performance_analysis(self):
        """Print detailed performance analysis"""
        metrics = self.metrics
        
        print("\n🎯 PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Performance Rating
        sharpe = metrics['sharpe_ratio']
        if sharpe > 2:
            rating = "EXCELLENT"
        elif sharpe > 1:
            rating = "GOOD"
        elif sharpe > 0.5:
            rating = "MODERATE"
        else:
            rating = "POOR"
        
        print(f"📊 Strategy Rating:                {rating}")
        print(f"📈 Risk-Adjusted Return:           {sharpe:.3f} Sharpe Ratio")
        
        # Profitability Analysis
        profit_factor = metrics['profit_factor']
        if profit_factor > 2:
            profitability = "HIGHLY PROFITABLE"
        elif profit_factor > 1.5:
            profitability = "PROFITABLE"
        elif profit_factor > 1:
            profitability = "MARGINALLY PROFITABLE"
        else:
            profitability = "UNPROFITABLE"
        
        print(f"💰 Profitability:                  {profitability}")
        print(f"📊 Profit Factor:                  {profit_factor:.2f}")
        
        # Risk Assessment
        max_dd = abs(metrics['max_drawdown'])
        if max_dd < 5:
            risk_level = "LOW RISK"
        elif max_dd < 15:
            risk_level = "MODERATE RISK"
        elif max_dd < 25:
            risk_level = "HIGH RISK"
        else:
            risk_level = "VERY HIGH RISK"
        
        print(f"⚠️  Risk Level:                    {risk_level}")
        print(f"📉 Maximum Drawdown:               {max_dd:.2f}%")
        
        # Trading Frequency
        trades_per_year = metrics['trades_per_year']
        if trades_per_year > 50:
            frequency = "HIGH FREQUENCY"
        elif trades_per_year > 20:
            frequency = "MEDIUM FREQUENCY"
        else:
            frequency = "LOW FREQUENCY"
        
        print(f"🔄 Trading Frequency:              {frequency}")
        print(f"📊 Trades per Year:                {trades_per_year:.1f}")
        
        # Efficiency
        efficiency_score = (metrics['sharpe_ratio'] * metrics['profit_factor']) / (1 + abs(metrics['max_drawdown']/100))
        print(f"⚡ Overall Efficiency Score:        {efficiency_score:.3f}")
        
        print("\n" + "="*80)
    
    def export_metrics_to_csv(self):
        """Export all metrics to CSV file"""
        # Create DataFrame for export
        metrics_df = pd.DataFrame(list(self.metrics.items()), 
                                columns=['Metric', 'Value'])
        
        # Add category information
        categories = {
            'total_return': 'Return Metrics',
            'cagr': 'Return Metrics',
            'monthly_return': 'Return Metrics',
            'profit_factor': 'Return Metrics',
            'expectancy_per_trade': 'Return Metrics',
            'sharpe_ratio': 'Risk Metrics',
            'sortino_ratio': 'Risk Metrics',
            'calmar_ratio': 'Risk Metrics',
            'max_drawdown': 'Risk Metrics',
            'avg_drawdown': 'Risk Metrics',
            'max_drawdown_duration': 'Risk Metrics',
            'total_trades': 'Trade Metrics',
            'win_rate': 'Trade Metrics',
            'avg_win': 'Trade Metrics',
            'avg_loss': 'Trade Metrics',
            'largest_win': 'Trade Metrics',
            'largest_loss': 'Trade Metrics',
            'avg_holding_period': 'Trade Metrics',
            'trades_per_year': 'Trade Metrics',
            'total_fees_paid': 'Cost Metrics',
            'fees_percentage': 'Cost Metrics',
            'time_in_market': 'Efficiency Metrics',
            'return_drawdown_ratio': 'Efficiency Metrics'
        }
        
        metrics_df['Category'] = metrics_df['Metric'].map(categories)
        
        # Save to CSV
        filename = "comprehensive_backtesting_metrics.csv"
        metrics_df.to_csv(filename, index=False)
        print(f"📊 Metrics exported to: {filename}")
        
        return filename

# Main execution
if __name__ == "__main__":
    print("🚀 MANUAL COMPREHENSIVE METRICS CALCULATOR")
    print("="*80)
    
    # Initialize calculator
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    calculator = ManualMetricsCalculator(data_folder)
    
    # Run manual backtest
    equity_df, trades_df = calculator.run_manual_backtest(
        initial_cash=100000,
        commission=0.001
    )
    
    # Display comprehensive results
    calculator.display_comprehensive_results()
    
    # Export metrics
    calculator.export_metrics_to_csv()
    
    print("\n🎉 COMPREHENSIVE BACKTESTING ANALYSIS COMPLETED!")
    print("="*80)
    print("📁 Files generated:")
    print("  • Comprehensive metrics report")
    print("  • CSV export of all metrics")
    print("  • Performance analysis")
    print("✅ All professional metrics calculated manually!")
