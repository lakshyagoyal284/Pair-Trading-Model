"""
CAPITAL OPTIMIZATION BACKTEST
Testing different capital levels to optimize performance and scalability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairs_trading_pipeline import PairsTradingPipeline
from datetime import time, datetime
import warnings
warnings.filterwarnings('ignore')

class CapitalOptimizationBacktest:
    """Capital optimization backtesting system"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pipeline = PairsTradingPipeline(data_folder)
        self.capital_results = {}
        
    def run_capital_optimization(self):
        """Run comprehensive capital optimization backtest"""
        print("💰 CAPITAL OPTIMIZATION BACKTEST")
        print("="*80)
        print("🎯 Testing different capital levels for optimal performance...")
        
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
        
        # Define capital levels to test
        capital_levels = [
            50000,      # $50K - Small capital
            100000,     # $100K - Baseline
            250000,     # $250K - Medium capital
            500000,     # $500K - Large capital
            1000000,    # $1M - Very large capital
            2500000,    # $2.5M - Institutional
            5000000,    # $5M - Large institutional
            10000000    # $10M - Very large institutional
        ]
        
        print(f"\n💰 Testing {len(capital_levels)} capital levels:")
        for capital in capital_levels:
            print(f"  • ${capital:,}")
        
        # Run backtest for each capital level
        all_results = []
        
        for capital in capital_levels:
            print(f"\n💰 Testing Capital: ${capital:,}")
            print("-" * 50)
            
            # Run backtest with this capital level
            metrics = self.run_backtest_with_capital(
                spread, z_score, capital
            )
            
            metrics['capital'] = capital
            all_results.append(metrics)
            
            # Display key results
            print(f"Total Return: {metrics['total_return']:+.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:+.3f}")
            print(f"Win Rate: {metrics['win_rate']:.1f}%")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Max Drawdown: {metrics['max_drawdown']:+.2f}%")
            print(f"Slippage Impact: {metrics['slippage_impact']:+.2f}%")
            print(f"Liquidity Score: {metrics['liquidity_score']:.1f}/100")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Display comprehensive results
        self.display_capital_optimization_results(results_df)
        
        # Generate visualizations
        self.generate_capital_visualizations(results_df)
        
        # Export report
        self.export_capital_report(results_df)
        
        return results_df
    
    def run_backtest_with_capital(self, spread, z_score, initial_capital):
        """Run backtest with specific capital level"""
        commission = 0.001
        
        # Strategy parameters (optimized from previous analysis)
        entry_threshold = 2.0
        exit_threshold = 0.5
        position_size = 0.3
        max_hold_days = 3
        avoid_monday = True
        
        # Capital-specific parameters
        base_position_size = position_size
        max_position_value = initial_capital * 0.1  # Max 10% per position
        
        # Slippage estimation based on capital size
        slippage_bps = self.estimate_slippage(initial_capital)
        
        cash = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        entry_price = None
        entry_date = None
        position_type = None
        days_held = 0
        position_value = 0
        
        for i in range(len(spread)):
            current_date = spread.index[i]
            current_spread = spread.iloc[i]
            current_z = z_score.iloc[i]
            
            if pd.isna(current_z):
                current_z = 0
            
            # Check if we should avoid Monday
            if avoid_monday:
                # Convert to datetime if it's a date
                if isinstance(current_date, pd.Timestamp):
                    day_of_week = current_date.dayofweek
                elif hasattr(current_date, 'weekday'):
                    day_of_week = current_date.weekday()
                else:
                    # If it's a date, convert to datetime
                    dt = pd.Timestamp.combine(current_date, pd.Timestamp.min.time())
                    day_of_week = dt.dayofweek
                
                if day_of_week == 0:  # Monday
                    # Force exit any open position
                    if position != 0:
                        days_held += 1
                        
                        # Calculate P&L with slippage
                        if position_type == 'LONG':
                            exit_price_with_slippage = current_spread * (1 - slippage_bps/10000)
                            pnl = (exit_price_with_slippage - entry_price) * position_value / entry_price
                        else:
                            exit_price_with_slippage = current_spread * (1 + slippage_bps/10000)
                            pnl = (entry_price - exit_price_with_slippage) * position_value / entry_price
                        
                        cash += pnl
                        cash -= abs(pnl) * commission
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'action': position_type,
                            'entry_price': entry_price,
                            'exit_price': current_spread,
                            'exit_price_with_slippage': current_spread * (1 - slippage_bps/10000) if position_type == 'LONG' else current_spread * (1 + slippage_bps/10000),
                            'pnl': pnl,
                            'holding_period': days_held,
                            'exit_reason': 'Avoid Monday',
                            'position_value': position_value,
                            'slippage_bps': slippage_bps
                        })
                        
                        position = 0
                        entry_price = None
                        position_type = None
                        days_held = 0
                        position_value = 0
                    
                    # Update equity
                    current_equity = cash
                    equity_curve.append(current_equity)
                    continue
            
            # Normal trading logic
            if position != 0:
                days_held += 1
            
            # Entry logic with capital constraints
            if position == 0:
                current_volatility = spread.rolling(5).std().iloc[i] if i >= 5 else 0
                avg_volatility = spread.rolling(20).std().mean()
                
                if current_volatility < avg_volatility * 1.5:
                    # Calculate position size based on capital
                    calculated_position_value = cash * base_position_size
                    
                    # Apply maximum position constraint
                    position_value = min(calculated_position_value, max_position_value)
                    
                    # Apply slippage to entry
                    if current_z > entry_threshold:  # Short signal
                        entry_price_with_slippage = current_spread * (1 + slippage_bps/10000)
                        position_type = 'SHORT'
                        entry_price = entry_price_with_slippage
                        entry_date = current_date
                        position = -1
                        cash -= position_value * commission
                        
                    elif current_z < -entry_threshold:  # Long signal
                        entry_price_with_slippage = current_spread * (1 - slippage_bps/10000)
                        position_type = 'LONG'
                        entry_price = entry_price_with_slippage
                        entry_date = current_date
                        position = 1
                        cash -= position_value * commission
                    
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
                    # Calculate P&L with slippage
                    if position_type == 'LONG':
                        exit_price_with_slippage = current_spread * (1 - slippage_bps/10000)
                        pnl = (exit_price_with_slippage - entry_price) * position_value / entry_price
                    else:
                        exit_price_with_slippage = current_spread * (1 + slippage_bps/10000)
                        pnl = (entry_price - exit_price_with_slippage) * position_value / entry_price
                    
                    cash += pnl
                    cash -= abs(pnl) * commission
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'action': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_spread,
                        'exit_price_with_slippage': exit_price_with_slippage,
                        'pnl': pnl,
                        'holding_period': days_held,
                        'exit_reason': exit_reason,
                        'position_value': position_value,
                        'slippage_bps': slippage_bps
                    })
                    
                    position = 0
                    entry_price = None
                    position_type = None
                    days_held = 0
                    position_value = 0
            
            # Calculate current equity
            current_equity = cash
            if position != 0:
                unrealized_pnl = 0
                if position_type == 'LONG':
                    unrealized_pnl = (current_spread - entry_price) * position_value / entry_price
                else:
                    unrealized_pnl = (entry_price - current_spread) * position_value / entry_price
                current_equity += unrealized_pnl
            
            equity_curve.append(current_equity)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_capital_metrics(equity_curve, trades, initial_capital, commission)
        
        return metrics
    
    def estimate_slippage(self, capital):
        """Estimate slippage in basis points based on capital size"""
        # Slippage increases with capital size due to market impact
        if capital <= 100000:
            return 2  # 2 bps for small capital
        elif capital <= 500000:
            return 5  # 5 bps for medium capital
        elif capital <= 1000000:
            return 10  # 10 bps for large capital
        elif capital <= 5000000:
            return 20  # 20 bps for very large capital
        else:
            return 35  # 35 bps for institutional capital
    
    def calculate_capital_metrics(self, equity_curve, trades, initial_capital, commission):
        """Calculate comprehensive metrics for capital optimization"""
        equity_values = np.array(equity_curve)
        equity_returns = pd.Series(equity_values).pct_change().dropna()
        
        # RETURN METRICS
        total_return = ((equity_values[-1] - initial_capital) / initial_capital) * 100
        years = len(equity_curve) / 252
        cagr = ((equity_values[-1] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            expectancy_per_trade = trades_df['pnl'].mean()
            
            # Calculate slippage impact
            total_slippage = trades_df['slippage_bps'].sum() * trades_df['position_value'].mean() / 10000
            slippage_impact = (total_slippage / initial_capital) * 100
        else:
            profit_factor = 0
            expectancy_per_trade = 0
            slippage_impact = 0
        
        # RISK METRICS
        if len(equity_returns) > 0 and equity_returns.std() > 0:
            sharpe_ratio = (equity_returns.mean() * 252) / (equity_returns.std() * np.sqrt(252))
            
            downside_returns = equity_returns[equity_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (equity_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))
            else:
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        max_dd = self.calculate_max_drawdown(equity_values)
        calmar_ratio = total_return / abs(max_dd) if max_dd != 0 else 0
        
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
                
            # Average position size
            avg_position_value = trades_df['position_value'].mean()
            avg_position_pct = (avg_position_value / initial_capital) * 100
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            largest_win = 0
            largest_loss = 0
            avg_holding_period = 0
            avg_position_value = 0
            avg_position_pct = 0
        
        trades_per_year = total_trades / years if years > 0 else 0
        
        # CAPITAL-SPECIFIC METRICS
        total_fees = total_trades * 2 * commission * avg_position_value if total_trades > 0 else 0
        fees_percentage = (total_fees / abs(gross_profit) * 100) if gross_profit != 0 else 0
        
        # Liquidity score (based on slippage and position size)
        liquidity_score = max(0, 100 - (slippage_impact * 10) - (avg_position_pct * 2))
        
        # Scalability score
        scalability_score = max(0, 100 - (initial_capital / 1000000) * 10)
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'expectancy_per_trade': expectancy_per_trade,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_holding_period': avg_holding_period,
            'trades_per_year': trades_per_year,
            'avg_position_value': avg_position_value,
            'avg_position_pct': avg_position_pct,
            'total_fees': total_fees,
            'fees_percentage': fees_percentage,
            'slippage_impact': slippage_impact,
            'liquidity_score': liquidity_score,
            'scalability_score': scalability_score
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
    
    def display_capital_optimization_results(self, results_df):
        """Display comprehensive capital optimization results"""
        print("\n" + "="*100)
        print("💰 CAPITAL OPTIMIZATION RESULTS")
        print("="*100)
        
        # Find best capital by different metrics
        best_return = results_df.loc[results_df['total_return'].idxmax()]
        best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        best_scalability = results_df.loc[results_df['scalability_score'].idxmax()]
        best_liquidity = results_df.loc[results_df['liquidity_score'].idxmax()]
        
        print(f"\n🏆 BEST CAPITAL BY TOTAL RETURN: ${best_return['capital']:,.0f}")
        print("-" * 100)
        print(f"Total Return:                     {best_return['total_return']:+.2f}%")
        print(f"Sharpe Ratio:                     {best_return['sharpe_ratio']:+.3f}")
        print(f"Win Rate:                         {best_return['win_rate']:.1f}%")
        print(f"Total Trades:                     {best_return['total_trades']}")
        print(f"Max Drawdown:                     {best_return['max_drawdown']:+.2f}%")
        print(f"Avg Position Size:                {best_return['avg_position_pct']:.1f}%")
        print(f"Slippage Impact:                  {best_return['slippage_impact']:+.2f}%")
        print(f"Liquidity Score:                  {best_return['liquidity_score']:.1f}/100")
        
        print(f"\n📊 CAPITAL COMPARISON TABLE")
        print("-" * 100)
        
        # Format the comparison table
        comparison_data = []
        for _, row in results_df.iterrows():
            comparison_data.append([
                f"${row['capital']:,.0f}",
                f"{row['total_return']:+.2f}%",
                f"{row['sharpe_ratio']:+.3f}",
                f"{row['win_rate']:.1f}%",
                f"{row['total_trades']}",
                f"{row['max_drawdown']:+.2f}%",
                f"{row['avg_position_pct']:.1f}%",
                f"{row['slippage_impact']:+.2f}%",
                f"{row['liquidity_score']:.1f}",
                f"{row['scalability_score']:.1f}"
            ])
        
        # Print table
        headers = ["Capital", "Return", "Sharpe", "Win Rate", "Trades", "Max DD", "Pos Size", "Slippage", "Liquidity", "Scalability"]
        print(f"{'Capital':<12} {'Return':<10} {'Sharpe':<8} {'Win Rate':<10} {'Trades':<8} {'Max DD':<10} {'Pos Size':<10} {'Slippage':<10} {'Liquidity':<10} {'Scalability':<10}")
        print("-" * 100)
        
        for row in comparison_data:
            print(f"{row[0]:<12} {row[1]:<10} {row[2]:<8} {row[3]:<10} {row[4]:<8} {row[5]:<10} {row[6]:<10} {row[7]:<10} {row[8]:<10} {row[9]:<10}")
        
        print(f"\n🎯 OPTIMAL CAPITAL RECOMMENDATIONS")
        print("-" * 100)
        
        # Analyze optimal ranges
        small_cap = results_df[results_df['capital'] <= 250000]
        medium_cap = results_df[(results_df['capital'] > 250000) & (results_df['capital'] <= 1000000)]
        large_cap = results_df[results_df['capital'] > 1000000]
        
        if not small_cap.empty:
            best_small = small_cap.loc[small_cap['total_return'].idxmax()]
            print(f"💰 Small Capital (<$250K): ${best_small['capital']:,.0f} - {best_small['total_return']:+.2f}% return")
        
        if not medium_cap.empty:
            best_medium = medium_cap.loc[medium_cap['total_return'].idxmax()]
            print(f"💰 Medium Capital ($250K-$1M): ${best_medium['capital']:,.0f} - {best_medium['total_return']:+.2f}% return")
        
        if not large_cap.empty:
            best_large = large_cap.loc[large_cap['total_return'].idxmax()]
            print(f"💰 Large Capital (>$1M): ${best_large['capital']:,.0f} - {best_large['total_return']:+.2f}% return")
        
        print(f"\n⚠️  CAPITAL SCALABILITY ANALYSIS")
        print("-" * 100)
        
        # Analyze scalability trends
        returns_by_capital = results_df.sort_values('capital')
        
        print("📈 Performance vs Capital Size:")
        for _, row in returns_by_capital.iterrows():
            trend = "📈" if row['total_return'] > 50 else "📊" if row['total_return'] > 20 else "📉"
            print(f"  {trend} ${row['capital']:,.0f}: {row['total_return']:+.2f}% (Liquidity: {row['liquidity_score']:.1f})")
        
        # Identify capital sweet spot
        sweet_spot = results_df[(results_df['liquidity_score'] > 70) & (results_df['total_return'] > 50)]
        if not sweet_spot.empty:
            optimal = sweet_spot.loc[sweet_spot['total_return'].idxmax()]
            print(f"\n🎯 CAPITAL SWEET SPOT: ${optimal['capital']:,.0f}")
            print(f"   • Balance of high returns ({optimal['total_return']:+.2f}%)")
            print(f"   • Good liquidity ({optimal['liquidity_score']:.1f}/100)")
            print(f"   • Manageable slippage ({optimal['slippage_impact']:+.2f}%)")
    
    def generate_capital_visualizations(self, results_df):
        """Generate capital optimization visualizations"""
        print("\n📊 GENERATING CAPITAL OPTIMIZATION VISUALIZATIONS...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CAPITAL OPTIMIZATION ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Return vs Capital
        ax1 = axes[0, 0]
        ax1.plot(results_df['capital'], results_df['total_return'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_title('Total Return vs Capital Size', fontweight='bold')
        ax1.set_xlabel('Capital ($)')
        ax1.set_ylabel('Total Return (%)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Sharpe Ratio vs Capital
        ax2 = axes[0, 1]
        ax2.plot(results_df['capital'], results_df['sharpe_ratio'], 'o-', linewidth=2, markersize=8, color='#F18F01')
        ax2.set_title('Sharpe Ratio vs Capital Size', fontweight='bold')
        ax2.set_xlabel('Capital ($)')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 3. Liquidity Score vs Capital
        ax3 = axes[0, 2]
        ax3.plot(results_df['capital'], results_df['liquidity_score'], 'o-', linewidth=2, markersize=8, color='#A23B72')
        ax3.set_title('Liquidity Score vs Capital Size', fontweight='bold')
        ax3.set_xlabel('Capital ($)')
        ax3.set_ylabel('Liquidity Score (0-100)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 4. Slippage Impact vs Capital
        ax4 = axes[1, 0]
        ax4.plot(results_df['capital'], results_df['slippage_impact'], 'o-', linewidth=2, markersize=8, color='#C73E1D')
        ax4.set_title('Slippage Impact vs Capital Size', fontweight='bold')
        ax4.set_xlabel('Capital ($)')
        ax4.set_ylabel('Slippage Impact (%)')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 5. Position Size vs Capital
        ax5 = axes[1, 1]
        ax5.plot(results_df['capital'], results_df['avg_position_pct'], 'o-', linewidth=2, markersize=8, color='#5C946E')
        ax5.set_title('Avg Position Size vs Capital', fontweight='bold')
        ax5.set_xlabel('Capital ($)')
        ax5.set_ylabel('Position Size (%)')
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 6. Multi-metrics Radar
        ax6 = axes[1, 2]
        
        # Normalize metrics for radar chart
        metrics_for_radar = ['total_return', 'sharpe_ratio', 'liquidity_score', 'scalability_score']
        categories = ['Return', 'Sharpe', 'Liquidity', 'Scalability']
        
        # Use best performing capital for radar
        best_capital = results_df.loc[results_df['total_return'].idxmax()]
        values = [
            min(best_capital['total_return'] / 100, 1),  # Normalize return
            min(best_capital['sharpe_ratio'], 1),        # Normalize sharpe
            best_capital['liquidity_score'] / 100,       # Normalize liquidity
            best_capital['scalability_score'] / 100      # Normalize scalability
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax6.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
        ax6.fill(angles, values, alpha=0.25, color='#2E86AB')
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title(f'Performance Radar (${best_capital["capital"]:,.0f})', fontweight='bold')
        ax6.grid(True)
        
        plt.tight_layout()
        plt.savefig('capital_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Capital optimization visualizations generated and saved!")
    
    def export_capital_report(self, results_df):
        """Export comprehensive capital optimization report"""
        # Create detailed report
        report_data = []
        
        # Header
        report_data.append([
            'Capital', 'Total_Return_%', 'CAGR_%', 'Sharpe_Ratio', 'Sortino_Ratio', 'Calmar_Ratio',
            'Max_Drawdown_%', 'Profit_Factor', 'Total_Trades', 'Win_Rate_%', 'Avg_Win_$', 'Avg_Loss_$',
            'Avg_Position_Size_%', 'Slippage_Impact_%', 'Total_Fees_$', 'Liquidity_Score', 'Scalability_Score'
        ])
        
        # Data rows
        for _, row in results_df.iterrows():
            report_data.append([
                f"${row['capital']:.0f}",
                f"{row['total_return']:.2f}",
                f"{row['cagr']:.2f}",
                f"{row['sharpe_ratio']:.3f}",
                f"{row['sortino_ratio']:.3f}",
                f"{row['calmar_ratio']:.3f}",
                f"{row['max_drawdown']:.2f}",
                f"{row['profit_factor']:.2f}",
                str(row['total_trades']),
                f"{row['win_rate']:.1f}",
                f"{row['avg_win']:.2f}",
                f"{row['avg_loss']:.2f}",
                f"{row['avg_position_pct']:.2f}",
                f"{row['slippage_impact']:.2f}",
                f"{row['total_fees']:.2f}",
                f"{row['liquidity_score']:.1f}",
                f"{row['scalability_score']:.1f}"
            ])
        
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data[1:], columns=report_data[0])
        report_df.to_csv('capital_optimization_report.csv', index=False)
        
        print(f"\n📊 Capital optimization report exported to: capital_optimization_report.csv")
        return 'capital_optimization_report.csv'

# Main execution
if __name__ == "__main__":
    print("💰 CAPITAL OPTIMIZATION BACKTEST")
    print("="*80)
    print("🎯 Testing different capital levels for optimal performance...")
    
    # Initialize capital optimization
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    optimizer = CapitalOptimizationBacktest(data_folder)
    
    # Run capital optimization
    results_df = optimizer.run_capital_optimization()
    
    print("\n🎉 CAPITAL OPTIMIZATION COMPLETED!")
    print("="*80)
    print("📁 Files generated:")
    print("  • capital_optimization_report.csv - Detailed capital analysis")
    print("  • capital_optimization_analysis.png - Visual capital charts")
    print("✅ Capital optimization completed!")
