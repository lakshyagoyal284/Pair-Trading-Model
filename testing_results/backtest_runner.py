"""
BACKTESTING RESULTS MODULE
Handles backtesting execution and result management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

class BacktestRunner:
    """Runs backtests and manages results"""
    
    def __init__(self, results_folder="../testing_results"):
        self.results_folder = results_folder
        self.backtest_results = {}
        self.portfolio_results = {}
        
        # Create results folder if it doesn't exist
        os.makedirs(results_folder, exist_ok=True)
    
    def run_single_backtest(self, strategy, pair, data, params):
        """Run backtest for a single pair"""
        stock1, stock2 = pair
        
        # Get price data
        stock1_data = data[stock1]
        stock2_data = data[stock2]
        
        # Run strategy
        result = strategy.backtest_pair(stock1_data, stock2_data, params)
        
        # Store result
        self.backtest_results[pair] = result
        
        return result
    
    def run_portfolio_backtest(self, strategy, pairs, data, params):
        """Run portfolio backtest for multiple pairs"""
        print(f"📊 Running portfolio backtest for {len(pairs)} pairs...")
        
        # Run individual backtests
        individual_results = {}
        for pair in pairs:
            try:
                result = self.run_single_backtest(strategy, pair, data, params)
                individual_results[pair] = result
                print(f"✅ Completed {pair[0]}-{pair[1]}: {result['total_return']:+.2f}%")
            except Exception as e:
                print(f"❌ Failed {pair[0]}-{pair[1]}: {e}")
                continue
        
        # Aggregate portfolio results
        portfolio_result = self.aggregate_portfolio_results(individual_results)
        self.portfolio_results['main'] = portfolio_result
        
        return portfolio_result
    
    def aggregate_portfolio_results(self, individual_results):
        """Aggregate individual pair results into portfolio"""
        if not individual_results:
            return {}
        
        # Equal weight portfolio
        portfolio_daily_returns = []
        portfolio_equity = []
        
        # Get the maximum length
        max_length = max(len(result.get('daily_returns', [])) for result in individual_results.values())
        
        # Calculate portfolio returns
        weight = 1.0 / len(individual_results)
        
        for i in range(max_length):
            daily_return = 0
            valid_pairs = 0
            
            for pair, result in individual_results.items():
                daily_returns = result.get('daily_returns', [])
                if i < len(daily_returns) and not np.isnan(daily_returns[i]):
                    daily_return += weight * daily_returns[i]
                    valid_pairs += 1
            
            if valid_pairs > 0:
                portfolio_daily_returns.append(daily_return)
            else:
                portfolio_daily_returns.append(0)
        
        # Calculate portfolio equity
        initial_capital = 100000
        portfolio_equity = [initial_capital]
        
        for daily_return in portfolio_daily_returns:
            portfolio_equity.append(portfolio_equity[-1] * (1 + daily_return))
        
        # Calculate portfolio metrics
        total_return = ((portfolio_equity[-1] - initial_capital) / initial_capital) * 100
        
        # Calculate Sharpe ratio
        if len(portfolio_daily_returns) > 0 and np.std(portfolio_daily_returns) > 0:
            sharpe_ratio = (np.mean(portfolio_daily_returns) * 252) / (np.std(portfolio_daily_returns) * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        max_drawdown = self.calculate_max_drawdown(portfolio_equity)
        
        # Calculate cumulative returns
        cumulative_returns = self.calculate_cumulative_returns(portfolio_daily_returns)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'daily_returns': portfolio_daily_returns,
            'equity_curve': portfolio_equity,
            'cumulative_returns': cumulative_returns,
            'individual_results': individual_results,
            'num_pairs': len(individual_results)
        }
    
    def calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def calculate_cumulative_returns(self, daily_returns):
        """Calculate cumulative returns"""
        cumulative_returns = []
        cumulative_return = 0
        
        for daily_return in daily_returns:
            cumulative_return = (1 + cumulative_return) * (1 + daily_return) - 1
            cumulative_returns.append(cumulative_return)
        
        return cumulative_returns
    
    def save_results(self, filename=None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
        
        filepath = os.path.join(self.results_folder, filename)
        
        # Prepare results for JSON serialization
        serializable_results = {}
        
        for pair, result in self.backtest_results.items():
            serializable_results[f"{pair[0]}_{pair[1]}"] = {
                'total_return': result.get('total_return', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0),
                'profit_factor': result.get('profit_factor', 0)
            }
        
        # Save portfolio results
        if self.portfolio_results:
            serializable_results['portfolio'] = {
                'total_return': self.portfolio_results['main'].get('total_return', 0),
                'sharpe_ratio': self.portfolio_results['main'].get('sharpe_ratio', 0),
                'max_drawdown': self.portfolio_results['main'].get('max_drawdown', 0),
                'num_pairs': self.portfolio_results['main'].get('num_pairs', 0)
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {filepath}")
        return filepath
    
    def generate_report(self, filename=None):
        """Generate comprehensive backtest report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.csv"
        
        filepath = os.path.join(self.results_folder, filename)
        
        # Prepare report data
        report_data = []
        
        # Header
        report_data.append([
            'Pair', 'Total_Return', 'Sharpe_Ratio', 'Max_Drawdown', 
            'Total_Trades', 'Win_Rate', 'Profit_Factor', 'Status'
        ])
        
        # Individual pairs
        for pair, result in self.backtest_results.items():
            stock1, stock2 = pair
            
            # Determine status
            total_return = result.get('total_return', 0)
            sharpe_ratio = result.get('sharpe_ratio', 0)
            
            if total_return > 10 and sharpe_ratio > 1:
                status = "EXCELLENT"
            elif total_return > 5 and sharpe_ratio > 0.5:
                status = "GOOD"
            elif total_return > 0:
                status = "POSITIVE"
            else:
                status = "NEGATIVE"
            
            report_data.append([
                f"{stock1}-{stock2}",
                f"{total_return:.2f}",
                f"{sharpe_ratio:.3f}",
                f"{result.get('max_drawdown', 0):.2f}",
                result.get('total_trades', 0),
                f"{result.get('win_rate', 0):.1f}",
                f"{result.get('profit_factor', 0):.2f}",
                status
            ])
        
        # Portfolio summary
        if self.portfolio_results:
            portfolio_result = self.portfolio_results['main']
            report_data.append([
                "PORTFOLIO",
                f"{portfolio_result.get('total_return', 0):.2f}",
                f"{portfolio_result.get('sharpe_ratio', 0):.3f}",
                f"{portfolio_result.get('max_drawdown', 0):.2f}",
                "N/A",
                "N/A",
                "N/A",
                "PORTFOLIO"
            ])
        
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data[1:], columns=report_data[0])
        report_df.to_csv(filepath, index=False)
        
        print(f"📊 Report saved to {filepath}")
        return filepath
    
    def create_visualizations(self, filename=None):
        """Create visualization plots"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_visualizations_{timestamp}.png"
        
        filepath = os.path.join(self.results_folder, filename)
        
        if not self.backtest_results:
            print("❌ No results to visualize")
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Results Summary', fontsize=16, fontweight='bold')
        
        # 1. Returns distribution
        ax1 = axes[0, 0]
        returns = [result.get('total_return', 0) for result in self.backtest_results.values()]
        ax1.hist(returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Total Return (%)')
        ax1.set_ylabel('Number of Pairs')
        ax1.set_title('Distribution of Returns')
        ax1.axvline(x=0, color='red', linestyle='--', label='Zero Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe ratio distribution
        ax2 = axes[0, 1]
        sharpe_ratios = [result.get('sharpe_ratio', 0) for result in self.backtest_results.values()]
        ax2.hist(sharpe_ratios, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Sharpe Ratio')
        ax2.set_ylabel('Number of Pairs')
        ax2.set_title('Distribution of Sharpe Ratios')
        ax2.axvline(x=1, color='red', linestyle='--', label='Target (1.0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Return vs Sharpe scatter
        ax3 = axes[1, 0]
        ax3.scatter(returns, sharpe_ratios, alpha=0.6, s=50)
        ax3.set_xlabel('Total Return (%)')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Return vs Sharpe Ratio')
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # 4. Portfolio equity curve
        ax4 = axes[1, 1]
        if self.portfolio_results:
            portfolio_result = self.portfolio_results['main']
            equity_curve = portfolio_result.get('equity_curve', [])
            if equity_curve:
                ax4.plot(equity_curve, color='blue', linewidth=2)
                ax4.set_xlabel('Time Periods')
                ax4.set_ylabel('Portfolio Value ($)')
                ax4.set_title('Portfolio Equity Curve')
                ax4.grid(True, alpha=0.3)
                
                # Add initial and final values
                ax4.axhline(y=equity_curve[0], color='green', linestyle='--', alpha=0.5, label='Start')
                ax4.axhline(y=equity_curve[-1], color='red', linestyle='--', alpha=0.5, label='End')
                ax4.legend()
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Visualizations saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print backtest summary"""
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        if not self.backtest_results:
            print("❌ No backtest results available")
            return
        
        # Individual pairs summary
        returns = [result.get('total_return', 0) for result in self.backtest_results.values()]
        sharpe_ratios = [result.get('sharpe_ratio', 0) for result in self.backtest_results.values()]
        max_drawdowns = [result.get('max_drawdown', 0) for result in self.backtest_results.values()]
        
        print(f"📈 Individual Pairs ({len(self.backtest_results)} pairs):")
        print(f"   Average Return: {np.mean(returns):+.2f}%")
        print(f"   Best Return: {np.max(returns):+.2f}%")
        print(f"   Worst Return: {np.min(returns):+.2f}%")
        print(f"   Average Sharpe: {np.mean(sharpe_ratios):+.3f}")
        print(f"   Best Sharpe: {np.max(sharpe_ratios):+.3f}")
        print(f"   Average Max DD: {np.mean(max_drawdowns):+.2f}%")
        
        # Portfolio summary
        if self.portfolio_results:
            portfolio_result = self.portfolio_results['main']
            print(f"\n💼 Portfolio:")
            print(f"   Total Return: {portfolio_result.get('total_return', 0):+.2f}%")
            print(f"   Sharpe Ratio: {portfolio_result.get('sharpe_ratio', 0):+.3f}")
            print(f"   Max Drawdown: {portfolio_result.get('max_drawdown', 0):+.2f}%")
            print(f"   Number of Pairs: {portfolio_result.get('num_pairs', 0)}")
        
        # Performance classification
        positive_pairs = sum(1 for r in returns if r > 0)
        excellent_pairs = sum(1 for r, s in zip(returns, sharpe_ratios) if r > 10 and s > 1)
        
        print(f"\n🎯 Performance Classification:")
        print(f"   Positive Pairs: {positive_pairs}/{len(returns)} ({positive_pairs/len(returns)*100:.1f}%)")
        print(f"   Excellent Pairs: {excellent_pairs}/{len(returns)} ({excellent_pairs/len(returns)*100:.1f}%)")
        
        print("="*80)

class ResultComparator:
    """Compare different backtest results"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def add_result(self, name, backtest_runner):
        """Add a result for comparison"""
        self.comparison_results[name] = {
            'individual_results': backtest_runner.backtest_results,
            'portfolio_results': backtest_runner.portfolio_results
        }
    
    def compare_strategies(self):
        """Compare different strategies"""
        if len(self.comparison_results) < 2:
            print("❌ Need at least 2 strategies to compare")
            return
        
        print("\n" + "="*80)
        print("📊 STRATEGY COMPARISON")
        print("="*80)
        
        comparison_data = []
        
        for strategy_name, results in self.comparison_results.items():
            portfolio_result = results['portfolio_results'].get('main', {})
            individual_results = results['individual_results']
            
            # Calculate metrics
            portfolio_return = portfolio_result.get('total_return', 0)
            portfolio_sharpe = portfolio_result.get('sharpe_ratio', 0)
            portfolio_dd = portfolio_result.get('max_drawdown', 0)
            
            individual_returns = [r.get('total_return', 0) for r in individual_results.values()]
            avg_return = np.mean(individual_returns)
            win_rate = sum(1 for r in individual_returns if r > 0) / len(individual_returns) * 100
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Portfolio_Return': portfolio_return,
                'Portfolio_Sharpe': portfolio_sharpe,
                'Portfolio_DD': portfolio_dd,
                'Avg_Individual_Return': avg_return,
                'Win_Rate': win_rate,
                'Num_Pairs': len(individual_results)
            })
        
        # Create comparison table
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Find best strategy
        best_return_strategy = df.loc[df['Portfolio_Return'].idxmax(), 'Strategy']
        best_sharpe_strategy = df.loc[df['Portfolio_Sharpe'].idxmax(), 'Strategy']
        
        print(f"\n🏆 Best Strategy by Return: {best_return_strategy}")
        print(f"🏆 Best Strategy by Sharpe: {best_sharpe_strategy}")
        
        return df
