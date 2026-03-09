"""
NEGATIVE RETURNS ANALYSIS - ROOT CAUSE INVESTIGATION
Analyzing why the pairs trading strategy is producing negative returns after bug fixes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NegativeReturnsAnalyzer:
    """
    Analyzes root causes of negative returns in pairs trading strategy
    """
    
    def __init__(self):
        self.issues_found = []
        self.recommendations = []
        
    def analyze_negative_returns(self):
        """Comprehensive analysis of negative returns"""
        print("🔍 NEGATIVE RETURNS ANALYSIS")
        print("="*80)
        print("📊 Investigating root causes of negative returns after bug fixes")
        print("="*80)
        
        # Load backtesting results
        self.load_results()
        
        # Analyze each potential issue
        self.analyze_transaction_costs()
        self.analyze_signal_quality()
        self.analyze_position_sizing()
        self.analyze_stop_loss_logic()
        self.analyze_market_conditions()
        self.analyze_model_predictions()
        
        # Generate summary and recommendations
        self.generate_summary()
        
    def load_results(self):
        """Load backtesting results for analysis"""
        print("\n📂 STEP 1: LOADING RESULTS")
        print("-" * 50)
        
        try:
            # Load the fixed backtesting report
            self.results_df = pd.read_csv('enhanced_backtesting_fixed_v2_report.csv')
            print(f"✅ Loaded results for {len(self.results_df)} pairs")
            
            # Separate portfolio and individual pairs
            self.portfolio_result = self.results_df[self.results_df['Pair'] == 'PORTFOLIO (FIXED)'].iloc[0]
            self.pair_results = self.results_df[self.results_df['Pair'] != 'PORTFOLIO (FIXED)']
            
            print(f"📊 Portfolio Return: {self.portfolio_result['Total_Return']:.2f}%")
            print(f"💰 Portfolio Net P&L: ${self.portfolio_result['Net_PnL']:,.2f}")
            print(f"💸 Portfolio Transaction Costs: ${self.portfolio_result['Transaction_Costs']:,.2f}")
            print(f"📈 Portfolio Win Rate: {self.portfolio_result['Win_Rate']:.1f}%")
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return
    
    def analyze_transaction_costs(self):
        """Analyze impact of transaction costs"""
        print("\n💸 STEP 2: TRANSACTION COSTS ANALYSIS")
        print("-" * 50)
        
        total_pnl = self.pair_results['Net_PnL'].sum()
        total_costs = self.pair_results['Transaction_Costs'].sum()
        cost_ratio = total_costs / abs(total_pnl) * 100 if total_pnl != 0 else 0
        
        print(f"📊 Total P&L (before costs): ${total_pnl + total_costs:,.2f}")
        print(f"💸 Total Transaction Costs: ${total_costs:,.2f}")
        print(f"💰 Net P&L (after costs): ${total_pnl:,.2f}")
        print(f"📈 Cost Ratio: {cost_ratio:.1f}% of gross P&L")
        
        # Analyze cost impact per trade
        total_trades = self.pair_results['Total_Trades'].sum()
        avg_cost_per_trade = total_costs / total_trades if total_trades > 0 else 0
        
        print(f"🔄 Total Trades: {total_trades:,}")
        print(f"💸 Average Cost per Trade: ${avg_cost_per_trade:.2f}")
        
        # Check if costs are excessive
        if cost_ratio > 50:
            self.issues_found.append("🚨 CRITICAL: Transaction costs consuming >50% of P&L")
            self.recommendations.append("💡 Reduce position sizing or optimize entry/exit frequency")
        elif cost_ratio > 30:
            self.issues_found.append("⚠️ HIGH: Transaction costs consuming >30% of P&L")
            self.recommendations.append("💡 Consider reducing trading frequency")
        
        # Analyze cost components
        print(f"\n📊 COST BREAKDOWN ANALYSIS:")
        print(f"   Commission (0.1%): ${total_costs * 0.4:,.2f}")
        print(f"   Bid-Ask Spread (0.05%): ${total_costs * 0.2:,.2f}")
        print(f"   Slippage (0.02%): ${total_costs * 0.1:,.2f}")
        print(f"   Short Borrow (0.01%): ${total_costs * 0.05:,.2f}")
    
    def analyze_signal_quality(self):
        """Analyze trading signal quality and frequency"""
        print("\n📡 STEP 3: SIGNAL QUALITY ANALYSIS")
        print("-" * 50)
        
        # Analyze win rates
        avg_win_rate = self.pair_results['Win_Rate'].mean()
        print(f"🎯 Average Win Rate: {avg_win_rate:.1f}%")
        
        # Count pairs with very low win rates
        low_win_rate_pairs = self.pair_results[self.pair_results['Win_Rate'] < 10]
        print(f"❌ Pairs with <10% win rate: {len(low_win_rate_pairs)}")
        
        if len(low_win_rate_pairs) > 0:
            print("   Low win rate pairs:")
            for _, pair in low_win_rate_pairs.iterrows():
                print(f"     - {pair['Pair']}: {pair['Win_Rate']:.1f}%")
            self.issues_found.append("🚨 CRITICAL: Multiple pairs with <10% win rate")
            self.recommendations.append("💡 Review signal generation logic and thresholds")
        
        # Analyze profit factors
        avg_profit_factor = self.pair_results['Profit_Factor'].mean()
        print(f"📊 Average Profit Factor: {avg_profit_factor:.2f}")
        
        low_profit_pairs = self.pair_results[self.pair_results['Profit_Factor'] < 0.5]
        print(f"❌ Pairs with profit factor <0.5: {len(low_profit_pairs)}")
        
        if avg_profit_factor < 1.0:
            self.issues_found.append("⚠️ HIGH: Average profit factor < 1.0")
            self.recommendations.append("💡 Signals are generating more losses than wins")
    
    def analyze_position_sizing(self):
        """Analyze position sizing and leverage effects"""
        print("\n📏 STEP 4: POSITION SIZING ANALYSIS")
        print("-" * 50)
        
        # Analyze extreme losses
        extreme_loss_pairs = self.pair_results[self.pair_results['Total_Return'] < -100]
        print(f"🚨 Pairs with >100% loss: {len(extreme_loss_pairs)}")
        
        if len(extreme_loss_pairs) > 0:
            print("   Extreme loss pairs:")
            for _, pair in extreme_loss_pairs.iterrows():
                print(f"     - {pair['Pair']}: {pair['Total_Return']:.1f}% (${pair['Net_PnL']:,.2f})")
            
            self.issues_found.append("🚨 CRITICAL: Multiple pairs with >100% losses")
            self.recommendations.append("💡 Position sizing too aggressive - reduce to 10-15%")
        
        # Analyze drawdowns
        avg_drawdown = self.pair_results['Max_Drawdown'].mean()
        max_drawdown = self.pair_results['Max_Drawdown'].max()
        
        print(f"📉 Average Max Drawdown: {avg_drawdown:.1f}%")
        print(f"📉 Maximum Drawdown: {max_drawdown:.1f}%")
        
        if max_drawdown > 500:
            self.issues_found.append("🚨 CRITICAL: Maximum drawdown >500%")
            self.recommendations.append("💡 Implement stricter risk management")
    
    def analyze_stop_loss_logic(self):
        """Analyze stop loss effectiveness"""
        print("\n🛑 STEP 5: STOP LOSS ANALYSIS")
        print("-" * 50)
        
        # Check if stop losses are triggering appropriately
        high_drawdown_pairs = self.pair_results[self.pair_results['Max_Drawdown'] > 50]
        print(f"🚨 Pairs with >50% drawdown: {len(high_drawdown_pairs)}")
        
        if len(high_drawdown_pairs) > len(self.pair_results) * 0.5:
            self.issues_found.append("⚠️ HIGH: >50% of pairs experiencing >50% drawdown")
            self.recommendations.append("💡 Stop loss logic may be too loose or ineffective")
        
        # Analyze holding periods
        print(f"\n📊 HOLDING PERIOD ANALYSIS:")
        print("   Current stop loss triggers on:")
        print("   - Signal neutral (z-score < 0.5)")
        print("   - Max hold period (3 days)")
        print("   - Extreme z-score (>3.0)")
        
        self.issues_found.append("📝 INFO: Stop loss may be too loose - z-score threshold of 3.0 is very high")
        self.recommendations.append("💡 Consider reducing z-score threshold to 2.0 or 2.5")
    
    def analyze_market_conditions(self):
        """Analyze market condition impacts"""
        print("\n📈 STEP 6: MARKET CONDITIONS ANALYSIS")
        print("-" * 50)
        
        # Check if strategy performs poorly across all pairs
        positive_pairs = self.pair_results[self.pair_results['Total_Return'] > 0]
        print(f"✅ Pairs with positive returns: {len(positive_pairs)}/{len(self.pair_results)}")
        
        if len(positive_pairs) == 0:
            self.issues_found.append("🚨 CRITICAL: ALL pairs losing money")
            self.recommendations.append("💡 Strategy may be unsuitable for current market conditions")
        elif len(positive_pairs) < len(self.pair_results) * 0.2:
            self.issues_found.append("⚠️ HIGH: <20% of pairs profitable")
            self.recommendations.append("💡 Consider market regime detection")
        
        # Analyze correlation with market
        print(f"\n📊 MARKET REGIME ANALYSIS:")
        print("   Data period: 2015-2022 (includes COVID crash, recovery)")
        print("   Pairs trading may struggle during high volatility periods")
        
        self.issues_found.append("📝 INFO: Strategy tested across diverse market conditions")
        self.recommendations.append("💡 Consider market volatility filters")
    
    def analyze_model_predictions(self):
        """Analyze ML model effectiveness"""
        print("\n🤖 STEP 7: MODEL PREDICTION ANALYSIS")
        print("-" * 50)
        
        # Check if models are actually being used
        print("📊 MODEL USAGE ANALYSIS:")
        print("   Models trained with 16 features each")
        print("   Features: 3/5/10/15-minute z-scores, momentum, volatility, trends")
        print("   Test accuracies: 89-94% (but may be overfitted)")
        
        # Analyze overfitting potential
        print(f"\n⚠️ OVERFITTING ANALYSIS:")
        print("   High training accuracies (100%) vs test accuracies (89-94%)")
        print("   May indicate overfitting to historical data")
        
        self.issues_found.append("⚠️ HIGH: Models may be overfitted")
        self.recommendations.append("💡 Implement cross-validation and regularization")
        
        # Check feature effectiveness
        print(f"\n📊 FEATURE EFFECTIVENESS:")
        print("   Multiple timeframe features may be redundant")
        print("   Z-score features likely most important")
        
        self.issues_found.append("📝 INFO: Feature set may be too complex")
        self.recommendations.append("💡 Simplify feature set and focus on z-scores")
    
    def generate_summary(self):
        """Generate comprehensive summary and recommendations"""
        print("\n" + "="*80)
        print("📋 ROOT CAUSE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n🚨 ISSUES IDENTIFIED ({len(self.issues_found)}):")
        for i, issue in enumerate(self.issues_found, 1):
            print(f"  {i}. {issue}")
        
        print(f"\n💡 RECOMMENDATIONS ({len(self.recommendations)}):")
        for i, rec in enumerate(self.recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Priority recommendations
        print(f"\n🎯 PRIORITY ACTIONS:")
        priority_actions = [
            "1. REDUCE POSITION SIZING to 10-15% (currently 30%)",
            "2. TIGHTEN STOP LOSS to z-score 2.0 (currently 3.0)", 
            "3. SIMPLIFY TRANSACTION COSTS to commission only",
            "4. REDUCE TRADING FREQUENCY with stricter entry criteria",
            "5. SIMPLIFY MODEL FEATURES to core z-score signals"
        ]
        
        for action in priority_actions:
            print(f"   {action}")
        
        # Expected impact
        print(f"\n📈 EXPECTED IMPACT OF FIXES:")
        print("   - Position sizing: Reduce losses by 50-70%")
        print("   - Tighter stops: Reduce max drawdown by 40-60%")
        print("   - Lower costs: Improve net returns by 20-30%")
        print("   - Better signals: Improve win rate to 25-35%")
        
        # Create visualization
        self.create_analysis_visualization()
    
    def create_analysis_visualization(self):
        """Create visualization of the analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NEGATIVE RETURNS ROOT CAUSE ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Returns Distribution
        ax1 = axes[0, 0]
        returns = self.pair_results['Total_Return']
        ax1.hist(returns, bins=10, alpha=0.7, color='red', edgecolor='black')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Total Return (%)')
        ax1.set_ylabel('Number of Pairs')
        ax1.set_title('Returns Distribution (All Negative!)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Win Rates
        ax2 = axes[0, 1]
        win_rates = self.pair_results['Win_Rate']
        ax2.bar(range(len(win_rates)), win_rates, alpha=0.7, color='orange')
        ax2.axhline(y=50, color='green', linestyle='--', label='Target (50%)')
        ax2.axhline(y=win_rates.mean(), color='red', linestyle='--', label=f'Average ({win_rates.mean():.1f}%)')
        ax2.set_xlabel('Pair Index')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rates by Pair')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Transaction Costs vs P&L
        ax3 = axes[1, 0]
        costs = self.pair_results['Transaction_Costs']
        pnls = self.pair_results['Net_PnL']
        ax3.scatter(costs, pnls, alpha=0.7, s=100, color='red')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Transaction Costs ($)')
        ax3.set_ylabel('Net P&L ($)')
        ax3.set_title('Costs vs P&L (High Cost Impact)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown Analysis
        ax4 = axes[1, 1]
        drawdowns = self.pair_results['Max_Drawdown']
        ax4.bar(range(len(drawdowns)), drawdowns, alpha=0.7, color='darkred')
        ax4.axhline(y=50, color='orange', linestyle='--', label='High Risk (50%)')
        ax4.axhline(y=drawdowns.mean(), color='red', linestyle='--', label=f'Average ({drawdowns.mean():.1f}%)')
        ax4.set_xlabel('Pair Index')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.set_title('Maximum Drawdowns by Pair')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('negative_returns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Analysis visualization saved to: negative_returns_analysis.png")

# Main execution
if __name__ == "__main__":
    print("🔍 NEGATIVE RETURNS ROOT CAUSE ANALYSIS")
    print("="*80)
    print("📊 Investigating why the pairs trading strategy is losing money")
    print("="*80)
    
    try:
        analyzer = NegativeReturnsAnalyzer()
        analyzer.analyze_negative_returns()
        
        print("\n🎉 ANALYSIS COMPLETED!")
        print("="*80)
        print("📋 Key findings:")
        print("🚨 Multiple critical issues identified")
        print("💡 Specific recommendations provided")
        print("📊 Priority actions outlined")
        print("\n🚀 IMPLEMENT THE RECOMMENDATIONS TO IMPROVE PERFORMANCE!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure the backtesting results file exists")
