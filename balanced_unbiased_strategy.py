"""
BALANCED UNBIASED PAIRS TRADING STRATEGY
Optimized balance between bias elimination and return generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairs_trading_pipeline import PairsTradingPipeline
import warnings
warnings.filterwarnings('ignore')

class BalancedUnbiasedStrategy:
    """Balanced unbiased pairs trading strategy"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pipeline = PairsTradingPipeline(data_folder)
        self.metrics = {}
        
        # Balanced parameters
        self.z_entry_threshold = 2.2  # Moderate threshold
        self.z_exit_threshold = 0.5   # Reasonable exit
        self.min_holding_days = 3     # Short minimum hold
        self.max_holding_days = 12    # Reasonable maximum hold
        self.position_size = 0.35     # Balanced position size
        self.bias_control_enabled = True  # Enable bias control
        self.max_bias_ratio = 0.4     # Max 40% imbalance
        self.volatility_filter = True  # Enable volatility filtering
        self.momentum_confirmation = False  # Disable for more trades
        
    def run_balanced_backtest(self, initial_cash=100000, commission=0.001):
        """Run balanced unbiased backtest"""
        print("🚀 RUNNING BALANCED UNBIASED STRATEGY")
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
        
        # Additional filters
        spread_volatility = spread.rolling(5).std()
        spread_trend = spread.rolling(10).mean() - spread.rolling(30).mean()
        
        # Balanced backtesting
        cash = initial_cash
        position = 0
        trades = []
        equity_curve = []
        entry_price = None
        entry_date = None
        position_type = None
        days_held = 0
        long_count = 0
        short_count = 0
        
        print("🔄 Executing balanced unbiased backtest...")
        print(f"📊 Balanced Parameters:")
        print(f"   • Entry Z-threshold: {self.z_entry_threshold}")
        print(f"   • Exit Z-threshold: {self.z_exit_threshold}")
        print(f"   • Position size: {self.position_size*100}%")
        print(f"   • Holding period: {self.min_holding_days}-{self.max_holding_days} days")
        print(f"   • Max bias ratio: {self.max_bias_ratio*100}%")
        
        for i in range(len(spread)):
            current_date = spread.index[i]
            current_spread = spread.iloc[i]
            current_z = z_score.iloc[i]
            
            # Handle NaN values
            if pd.isna(current_z):
                current_z = 0
            
            # Calculate metrics
            current_volatility = spread_volatility.iloc[i] if not pd.isna(spread_volatility.iloc[i]) else 0
            current_trend = spread_trend.iloc[i] if not pd.isna(spread_trend.iloc[i]) else 0
            
            # Update days held
            if position != 0:
                days_held += 1
            
            # Balanced entry conditions
            if position == 0:  # No position
                # Calculate current bias ratio
                total_trades = long_count + short_count
                if total_trades > 0:
                    bias_ratio = abs(long_count - short_count) / total_trades
                else:
                    bias_ratio = 0
                
                # Quality filters
                quality_pass = True
                
                # Volatility filter
                if self.volatility_filter and current_volatility < spread_volatility.quantile(0.25):
                    quality_pass = False
                
                # Trend filter (avoid strong trends)
                if abs(current_trend) > current_volatility * 2:
                    quality_pass = False
                
                # Bias control
                if self.bias_control_enabled and bias_ratio > self.max_bias_ratio:
                    quality_pass = False
                
                # Entry logic
                if quality_pass:
                    if current_z > self.z_entry_threshold:  # Short signal
                        position_type = 'SHORT'
                        entry_price = current_spread
                        entry_date = current_date
                        position = -1
                        trade_size = cash * self.position_size
                        cash -= trade_size * commission
                        short_count += 1
                        
                        trades.append({
                            'entry_date': entry_date,
                            'action': 'SHORT',
                            'entry_price': entry_price,
                            'z_score': current_z,
                            'volatility': current_volatility,
                            'bias_ratio': bias_ratio,
                            'trade_balance': f"L:{long_count} S:{short_count}"
                        })
                        print(f"🔴 SHORT at Z={current_z:.2f} (Bias={bias_ratio:.2f}, Bal=L:{long_count} S:{short_count})")
                        days_held = 0
                        
                    elif current_z < -self.z_entry_threshold:  # Long signal
                        position_type = 'LONG'
                        entry_price = current_spread
                        entry_date = current_date
                        position = 1
                        trade_size = cash * self.position_size
                        cash -= trade_size * commission
                        long_count += 1
                        
                        trades.append({
                            'entry_date': entry_date,
                            'action': 'LONG',
                            'entry_price': entry_price,
                            'z_score': current_z,
                            'volatility': current_volatility,
                            'bias_ratio': bias_ratio,
                            'trade_balance': f"L:{long_count} S:{short_count}"
                        })
                        print(f"🟢 LONG at Z={current_z:.2f} (Bias={bias_ratio:.2f}, Bal=L:{long_count} S:{short_count})")
                        days_held = 0
            
            else:  # Have position - balanced exit conditions
                should_exit = False
                exit_reason = ""
                
                # Exit conditions
                if abs(current_z) < self.z_exit_threshold:
                    should_exit = True
                    exit_reason = f"Neutral Z ({current_z:.2f})"
                elif days_held >= self.max_holding_days:
                    should_exit = True
                    exit_reason = f"Max hold ({days_held}d)"
                elif days_held >= self.min_holding_days:
                    if position_type == 'LONG' and current_z > 0.2:
                        should_exit = True
                        exit_reason = f"Long profit (Z={current_z:.2f})"
                    elif position_type == 'SHORT' and current_z < -0.2:
                        should_exit = True
                        exit_reason = f"Short profit (Z={current_z:.2f})"
                    elif abs(current_z) < 1.0:
                        should_exit = True
                        exit_reason = f"Early exit (Z={current_z:.2f})"
                
                if should_exit:
                    # Calculate P&L
                    if position_type == 'LONG':
                        pnl = (current_spread - entry_price) * (cash * self.position_size) / entry_price
                    else:  # SHORT
                        pnl = (entry_price - current_spread) * (cash * self.position_size) / entry_price
                    
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
                    unrealized_pnl = (current_spread - entry_price) * (cash * self.position_size) / entry_price
                else:
                    unrealized_pnl = (entry_price - current_spread) * (cash * self.position_size) / entry_price
                current_equity += unrealized_pnl
            
            equity_curve.append({
                'date': current_date,
                'equity': current_equity,
                'cash': cash,
                'position': position,
                'position_type': position_type,
                'spread': current_spread,
                'z_score': current_z,
                'days_held': days_held,
                'long_count': long_count,
                'short_count': short_count
            })
        
        # Convert to DataFrame with proper datetime index
        equity_df = pd.DataFrame(equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        # Calculate all metrics
        self.calculate_all_metrics(equity_df, trades_df, initial_cash, commission, long_count, short_count)
        
        return equity_df, trades_df
    
    def calculate_all_metrics(self, equity_df, trades_df, initial_cash, commission, long_count, short_count):
        """Calculate all comprehensive metrics"""
        print("\n📊 CALCULATING BALANCED METRICS...")
        
        # Extract data
        equity_values = equity_df['equity'].values
        equity_returns = equity_df['equity'].pct_change().dropna()
        
        # RETURN METRICS
        print("📈 Calculating Return Metrics...")
        
        # Total Return
        total_return = ((equity_values[-1] - initial_cash) / initial_cash) * 100
        self.metrics['total_return'] = total_return
        
        # CAGR
        years = len(equity_df) / 252  # Assuming daily data
        cagr = ((equity_values[-1] / initial_cash) ** (1/years) - 1) * 100 if years > 0 else 0
        self.metrics['cagr'] = cagr
        
        # Monthly Return
        monthly_return = total_return / 12
        self.metrics['monthly_return'] = monthly_return
        
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
        if len(equity_returns) > 0 and equity_returns.std() > 0:
            sharpe_ratio = (equity_returns.mean() * 252) / (equity_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        self.metrics['sharpe_ratio'] = sharpe_ratio
        
        # Sortino Ratio
        if len(equity_returns) > 0:
            downside_returns = equity_returns[equity_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
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
        total_fees = total_trades * 2 * (initial_cash * self.position_size * commission)
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
            time_in_market = (total_time_in_market / (years * 252)) * 100
        else:
            time_in_market = 0
        self.metrics['time_in_market'] = time_in_market
        
        # Return / Drawdown Ratio
        return_drawdown_ratio = abs(total_return / max_dd) if max_dd != 0 else float('inf')
        self.metrics['return_drawdown_ratio'] = return_drawdown_ratio
        
        # BIAS ANALYSIS
        print("🎯 Analyzing Trading Bias...")
        
        # Trade Balance Bias
        total_signals = long_count + short_count
        if total_signals > 0:
            trade_balance_bias = abs(long_count - short_count) / total_signals * 100
        else:
            trade_balance_bias = 0
        self.metrics['trade_balance_bias'] = trade_balance_bias
        
        # Signal Quality
        if len(trades_df) > 0 and 'z_score' in trades_df.columns:
            avg_signal_strength = trades_df['z_score'].abs().mean()
            self.metrics['avg_signal_strength'] = avg_signal_strength
        
        print("✅ All balanced metrics calculated successfully!")
    
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
    
    def display_balanced_results(self):
        """Display balanced results"""
        metrics = self.metrics
        
        print("\n" + "="*80)
        print("📊 BALANCED UNBIASED TRADING RESULTS")
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
        
        # BIAS ANALYSIS
        print("\n🎯 BIAS ANALYSIS")
        print("-" * 50)
        bias_level = metrics['trade_balance_bias']
        if bias_level < 10:
            bias_rating = "EXCELLENT UNBIASED ✓"
        elif bias_level < 20:
            bias_rating = "GOOD UNBIASED ✓"
        elif bias_level < 30:
            bias_rating = "MODERATE BIAS ⚠"
        else:
            bias_rating = "HIGH BIAS ✗"
        print(f"Trade Balance Bias:              {bias_level:.1f}% ({bias_rating})")
        
        if 'avg_signal_strength' in metrics:
            strength = metrics['avg_signal_strength']
            print(f"Avg Signal Strength:             {strength:.2f}")
        
        print("\n" + "="*80)
        
        # Performance Analysis
        self.print_balanced_analysis()
    
    def print_balanced_analysis(self):
        """Print balanced performance analysis"""
        metrics = self.metrics
        
        print("\n🎯 BALANCED PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Overall Performance Rating
        sharpe = metrics['sharpe_ratio']
        total_return = metrics['total_return']
        bias_score = 100 - metrics.get('trade_balance_bias', 0)
        
        if sharpe > 0.8 and total_return > 12 and bias_score > 80:
            rating = "EXCELLENT BALANCED"
        elif sharpe > 0.5 and total_return > 8 and bias_score > 70:
            rating = "GOOD BALANCED"
        elif sharpe > 0.2 and total_return > 4 and bias_score > 60:
            rating = "MODERATE BALANCED"
        else:
            rating = "NEEDS ADJUSTMENT"
        
        print(f"📊 Strategy Rating:                {rating}")
        print(f"📈 Risk-Adjusted Return:           {sharpe:.3f} Sharpe Ratio")
        print(f"🎯 Balanced Score:                 {bias_score:.1f}/100")
        
        # Return Quality Assessment
        profit_factor = metrics['profit_factor']
        win_rate = metrics['win_rate']
        
        if profit_factor > 1.3 and win_rate > 52:
            quality = "HIGH QUALITY RETURNS"
        elif profit_factor > 1.1 and win_rate > 48:
            quality = "GOOD QUALITY RETURNS"
        elif profit_factor > 0.9 and win_rate > 45:
            quality = "ACCEPTABLE RETURNS"
        else:
            quality = "LOW QUALITY RETURNS"
        
        print(f"💎 Return Quality:                 {quality}")
        print(f"📊 Profit Factor:                  {profit_factor:.2f}")
        print(f"🎯 Win Rate:                       {win_rate:.1f}%")
        
        # Trading Efficiency
        trades_per_year = metrics['trades_per_year']
        if trades_per_year < 15:
            efficiency = "CONSERVATIVE"
        elif trades_per_year < 35:
            efficiency = "BALANCED"
        elif trades_per_year < 60:
            efficiency = "ACTIVE"
        else:
            efficiency = "OVER-ACTIVE"
        
        print(f"⚡ Trading Efficiency:             {efficiency}")
        print(f"📊 Trades per Year:                {trades_per_year:.1f}")
        
        # Risk-Return Balance
        max_dd = abs(metrics['max_drawdown'])
        if max_dd < 12 and total_return > 10:
            balance = "EXCELLENT BALANCE"
        elif max_dd < 20 and total_return > 6:
            balance = "GOOD BALANCE"
        elif max_dd < 30 and total_return > 2:
            balance = "MODERATE BALANCE"
        else:
            balance = "POOR BALANCE"
        
        print(f"⚖️  Risk-Return Balance:            {balance}")
        print(f"📉 Risk/Reward Ratio:              {max_dd:.2f}% risk vs {total_return:+.2f}% return")
        
        # Balance Success Metrics
        print("\n🚀 BALANCE SUCCESS METRICS")
        print("-" * 50)
        
        if bias_score > 80:
            print("✅ Excellent bias control (>80%)")
        elif bias_score > 70:
            print("✅ Good bias control (>70%)")
        else:
            print("❌ Bias control needs improvement")
        
        if total_return > 8:
            print("✅ Positive returns achieved (>8%)")
        elif total_return > 0:
            print("✅ Positive returns achieved")
        else:
            print("❌ Returns need improvement")
        
        if sharpe > 0.5:
            print("✅ Good risk-adjusted returns (>0.5)")
        else:
            print("❌ Risk-adjusted returns need improvement")
        
        if profit_factor > 1.1:
            print("✅ Good profitability (>1.1)")
        else:
            print("❌ Profitability needs improvement")
        
        if 15 < trades_per_year < 50:
            print("✅ Optimal trading frequency")
        else:
            print("⚠️ Trading frequency could be optimized")
        
        print("\n" + "="*80)
    
    def export_balanced_metrics(self):
        """Export balanced metrics to CSV"""
        # Create DataFrame for export
        metrics_df = pd.DataFrame(list(self.metrics.items()), 
                                columns=['Metric', 'Value'])
        
        # Save to CSV
        filename = "balanced_unbiased_metrics.csv"
        metrics_df.to_csv(filename, index=False)
        print(f"📊 Balanced metrics exported to: {filename}")
        
        return filename

# Main execution
if __name__ == "__main__":
    print("🚀 BALANCED UNBIASED PAIRS TRADING STRATEGY")
    print("="*80)
    print("🎯 Optimized balance between bias elimination and return generation...")
    
    # Initialize balanced strategy
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    strategy = BalancedUnbiasedStrategy(data_folder)
    
    # Run balanced backtest
    equity_df, trades_df = strategy.run_balanced_backtest(
        initial_cash=100000,
        commission=0.001
    )
    
    # Display balanced results
    strategy.display_balanced_results()
    
    # Export metrics
    strategy.export_balanced_metrics()
    
    print("\n🎉 BALANCED UNBIASED STRATEGY COMPLETED!")
    print("="*80)
    print("📁 Files generated:")
    print("  • Balanced unbiased metrics")
    print("  • Bias control report")
    print("  • Balance success metrics")
    print("✅ Strategy balanced for optimal unbiased returns!")
