"""
MODEL ACCURACY ANALYSIS
Comprehensive accuracy analysis for the optimized pairs trading strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairs_trading_pipeline import PairsTradingPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelAccuracyAnalysis:
    """Comprehensive model accuracy analysis"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pipeline = PairsTradingPipeline(data_folder)
        self.accuracy_metrics = {}
        self.predictions = []
        self.actuals = []
        
    def run_accuracy_analysis(self):
        """Run comprehensive accuracy analysis"""
        print("🎯 MODEL ACCURACY ANALYSIS")
        print("="*80)
        print("📊 Analyzing prediction accuracy and trading performance...")
        
        # Run pipeline to get data
        self.pipeline.step1_data_preprocessing()
        top_pairs = self.pipeline.step2_pair_selection()
        
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
        
        # Generate predictions and actual outcomes
        self.generate_predictions_and_actuals(spread, z_score)
        
        # Calculate accuracy metrics
        self.calculate_accuracy_metrics()
        
        # Display results
        self.display_accuracy_results()
        
        # Generate visualizations
        self.generate_accuracy_visualizations()
        
        return self.accuracy_metrics
    
    def generate_predictions_and_actuals(self, spread, z_score):
        """Generate predictions and actual outcomes"""
        print("\n🔍 GENERATING PREDICTIONS AND ACTUALS...")
        
        # Strategy parameters (optimized)
        entry_threshold = 2.0
        exit_threshold = 0.5
        max_hold_days = 3
        
        predictions = []
        actuals = []
        trade_details = []
        
        position = 0
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
                    if current_z > entry_threshold:
                        position_type = 'SHORT'
                        entry_price = current_spread
                        entry_date = current_date
                        position = -1
                        days_held = 0
                        
                        # Prediction: SHORT (expect spread to go down)
                        predictions.append(-1)
                        
                    elif current_z < -entry_threshold:
                        position_type = 'LONG'
                        entry_price = current_spread
                        entry_date = current_date
                        position = 1
                        days_held = 0
                        
                        # Prediction: LONG (expect spread to go up)
                        predictions.append(1)
            
            else:  # Exit logic
                should_exit = False
                exit_reason = ""
                
                if abs(current_z) < exit_threshold:
                    should_exit = True
                    exit_reason = "Target reached"
                elif days_held >= max_hold_days:
                    should_exit = True
                    exit_reason = "Max hold"
                elif days_held >= 2:
                    if position_type == 'LONG' and current_z < -4.0:
                        should_exit = True
                        exit_reason = "Stop loss"
                    elif position_type == 'SHORT' and current_z > 4.0:
                        should_exit = True
                        exit_reason = "Stop loss"
                    elif position_type == 'LONG' and current_z > -0.3:
                        should_exit = True
                        exit_reason = "Early profit"
                    elif position_type == 'SHORT' and current_z < 0.3:
                        should_exit = True
                        exit_reason = "Early profit"
                
                if should_exit:
                    # Calculate actual outcome
                    if position_type == 'LONG':
                        pnl = (current_spread - entry_price) * 100000 * 0.3 / entry_price
                        # Actual: 1 if profitable, -1 if loss
                        actual = 1 if pnl > 0 else -1
                    else:  # SHORT
                        pnl = (entry_price - current_spread) * 100000 * 0.3 / entry_price
                        # Actual: -1 if profitable, 1 if loss (opposite for short)
                        actual = -1 if pnl > 0 else 1
                    
                    actuals.append(actual)
                    
                    trade_details.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'position_type': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_spread,
                        'entry_z': predictions[-1] if predictions else 0,
                        'actual_outcome': actual,
                        'pnl': pnl,
                        'holding_period': days_held,
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    entry_price = None
                    position_type = None
                    days_held = 0
        
        self.predictions = predictions
        self.actuals = actuals
        self.trade_details = trade_details
        
        print(f"✅ Generated {len(predictions)} predictions and {len(actuals)} actual outcomes")
    
    def calculate_accuracy_metrics(self):
        """Calculate comprehensive accuracy metrics"""
        print("\n📊 CALCULATING ACCURACY METRICS...")
        
        if len(self.predictions) == 0 or len(self.actuals) == 0:
            print("❌ No trades to analyze")
            return
        
        # Ensure same length
        min_length = min(len(self.predictions), len(self.actuals))
        predictions = self.predictions[:min_length]
        actuals = self.actuals[:min_length]
        
        # Basic accuracy
        accuracy = accuracy_score(actuals, predictions)
        
        # Precision, Recall, F1
        precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
        recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
        f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
        
        # Direction accuracy
        correct_directions = sum(1 for pred, actual in zip(predictions, actuals) if pred == actual)
        direction_accuracy = correct_directions / len(predictions) * 100
        
        # Profit accuracy
        profitable_trades = sum(1 for trade in self.trade_details if trade['pnl'] > 0)
        profit_accuracy = profitable_trades / len(self.trade_details) * 100
        
        # Long vs Short accuracy
        long_trades = [trade for trade in self.trade_details if trade['position_type'] == 'LONG']
        short_trades = [trade for trade in self.trade_details if trade['position_type'] == 'SHORT']
        
        long_accuracy = sum(1 for trade in long_trades if trade['pnl'] > 0) / len(long_trades) * 100 if long_trades else 0
        short_accuracy = sum(1 for trade in short_trades if trade['pnl'] > 0) / len(short_trades) * 100 if short_trades else 0
        
        # Z-score accuracy
        high_z_trades = [trade for trade in self.trade_details if abs(trade['entry_z']) > 2.5]
        low_z_trades = [trade for trade in self.trade_details if abs(trade['entry_z']) <= 2.5]
        
        high_z_accuracy = sum(1 for trade in high_z_trades if trade['pnl'] > 0) / len(high_z_trades) * 100 if high_z_trades else 0
        low_z_accuracy = sum(1 for trade in low_z_trades if trade['pnl'] > 0) / len(low_z_trades) * 100 if low_z_trades else 0
        
        # Holding period accuracy
        quick_trades = [trade for trade in self.trade_details if trade['holding_period'] <= 2]
        long_trades_period = [trade for trade in self.trade_details if trade['holding_period'] > 2]
        
        quick_accuracy = sum(1 for trade in quick_trades if trade['pnl'] > 0) / len(quick_trades) * 100 if quick_trades else 0
        long_period_accuracy = sum(1 for trade in long_trades_period if trade['pnl'] > 0) / len(long_trades_period) * 100 if long_trades_period else 0
        
        # Store metrics
        self.accuracy_metrics = {
            'overall_accuracy': accuracy * 100,
            'direction_accuracy': direction_accuracy,
            'profit_accuracy': profit_accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'long_accuracy': long_accuracy,
            'short_accuracy': short_accuracy,
            'high_z_accuracy': high_z_accuracy,
            'low_z_accuracy': low_z_accuracy,
            'quick_trade_accuracy': quick_accuracy,
            'long_period_accuracy': long_period_accuracy,
            'total_predictions': len(predictions),
            'total_actuals': len(actuals),
            'total_trades': len(self.trade_details),
            'profitable_trades': profitable_trades,
            'losing_trades': len(self.trade_details) - profitable_trades
        }
        
        print("✅ Accuracy metrics calculated successfully!")
    
    def display_accuracy_results(self):
        """Display comprehensive accuracy results"""
        metrics = self.accuracy_metrics
        
        print("\n" + "="*80)
        print("📊 MODEL ACCURACY ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\n🎯 OVERALL ACCURACY METRICS")
        print("-" * 50)
        print(f"Overall Accuracy:                {metrics['overall_accuracy']:.2f}%")
        print(f"Direction Accuracy:              {metrics['direction_accuracy']:.2f}%")
        print(f"Profit Accuracy:                 {metrics['profit_accuracy']:.2f}%")
        print(f"Precision:                       {metrics['precision']:.2f}%")
        print(f"Recall:                          {metrics['recall']:.2f}%")
        print(f"F1 Score:                        {metrics['f1_score']:.2f}%")
        
        print(f"\n🔄 POSITION TYPE ACCURACY")
        print("-" * 50)
        print(f"LONG Position Accuracy:           {metrics['long_accuracy']:.2f}%")
        print(f"SHORT Position Accuracy:          {metrics['short_accuracy']:.2f}%")
        
        print(f"\n📈 Z-SCORE ACCURACY")
        print("-" * 50)
        print(f"High Z-Score Accuracy (>2.5):     {metrics['high_z_accuracy']:.2f}%")
        print(f"Low Z-Score Accuracy (≤2.5):      {metrics['low_z_accuracy']:.2f}%")
        
        print(f"\n⏱️ HOLDING PERIOD ACCURACY")
        print("-" * 50)
        print(f"Quick Trade Accuracy (≤2 days):   {metrics['quick_trade_accuracy']:.2f}%")
        print(f"Long Period Accuracy (>2 days):   {metrics['long_period_accuracy']:.2f}%")
        
        print(f"\n📊 TRADE STATISTICS")
        print("-" * 50)
        print(f"Total Predictions:                {metrics['total_predictions']}")
        print(f"Total Actuals:                    {metrics['total_actuals']}")
        print(f"Total Trades:                     {metrics['total_trades']}")
        print(f"Profitable Trades:                {metrics['profitable_trades']}")
        print(f"Losing Trades:                    {metrics['losing_trades']}")
        
        # Accuracy Assessment
        print(f"\n🚀 ACCURACY ASSESSMENT")
        print("-" * 50)
        
        if metrics['overall_accuracy'] >= 80:
            print("🏆 EXCELLENT MODEL ACCURACY (≥80%)")
        elif metrics['overall_accuracy'] >= 70:
            print("✅ GOOD MODEL ACCURACY (≥70%)")
        elif metrics['overall_accuracy'] >= 60:
            print("⚠️  MODERATE MODEL ACCURACY (≥60%)")
        else:
            print("❌ POOR MODEL ACCURACY (<60%)")
        
        if metrics['profit_accuracy'] >= 60:
            print("🏆 EXCELLENT PROFIT ACCURACY (≥60%)")
        elif metrics['profit_accuracy'] >= 50:
            print("✅ GOOD PROFIT ACCURACY (≥50%)")
        elif metrics['profit_accuracy'] >= 40:
            print("⚠️  MODERATE PROFIT ACCURACY (≥40%)")
        else:
            print("❌ POOR PROFIT ACCURACY (<40%)")
        
        # Position Analysis
        if abs(metrics['long_accuracy'] - metrics['short_accuracy']) <= 10:
            print("✅ BALANCED LONG/SHORT PERFORMANCE")
        elif metrics['long_accuracy'] > metrics['short_accuracy']:
            print("⚠️  LONG POSITIONS OUTPERFORM SHORT")
        else:
            print("⚠️  SHORT POSITIONS OUTPERFORM LONG")
        
        # Z-Score Analysis
        if metrics['high_z_accuracy'] > metrics['low_z_accuracy']:
            print("✅ HIGH Z-SCORE SIGNALS MORE RELIABLE")
        else:
            print("⚠️  LOW Z-SCORE SIGNALS MORE RELIABLE")
        
        print("\n" + "="*80)
    
    def generate_accuracy_visualizations(self):
        """Generate accuracy visualization charts"""
        print("\n📊 GENERATING ACCURACY VISUALIZATIONS...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MODEL ACCURACY ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Overall Accuracy Metrics
        ax1 = axes[0, 0]
        metrics_to_plot = ['overall_accuracy', 'direction_accuracy', 'profit_accuracy', 'precision', 'recall', 'f1_score']
        metric_values = [self.accuracy_metrics[m] for m in metrics_to_plot]
        metric_labels = ['Overall', 'Direction', 'Profit', 'Precision', 'Recall', 'F1 Score']
        
        bars = ax1.bar(metric_labels, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5C946E', '#6C4A6D'])
        ax1.set_title('Overall Accuracy Metrics', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Position Type Accuracy
        ax2 = axes[0, 1]
        positions = ['LONG', 'SHORT']
        position_accuracies = [self.accuracy_metrics['long_accuracy'], self.accuracy_metrics['short_accuracy']]
        colors = ['#2E86AB', '#A23B72']
        
        bars = ax2.bar(positions, position_accuracies, color=colors)
        ax2.set_title('Position Type Accuracy', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars, position_accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Z-Score Accuracy
        ax3 = axes[0, 2]
        z_categories = ['High Z (>2.5)', 'Low Z (≤2.5)']
        z_accuracies = [self.accuracy_metrics['high_z_accuracy'], self.accuracy_metrics['low_z_accuracy']]
        colors = ['#F18F01', '#C73E1D']
        
        bars = ax3.bar(z_categories, z_accuracies, color=colors)
        ax3.set_title('Z-Score Signal Accuracy', fontweight='bold')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_ylim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars, z_accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Holding Period Accuracy
        ax4 = axes[1, 0]
        holding_categories = ['Quick (≤2 days)', 'Long (>2 days)']
        holding_accuracies = [self.accuracy_metrics['quick_trade_accuracy'], self.accuracy_metrics['long_period_accuracy']]
        colors = ['#5C946E', '#6C4A6D']
        
        bars = ax4.bar(holding_categories, holding_accuracies, color=colors)
        ax4.set_title('Holding Period Accuracy', fontweight='bold')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_ylim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars, holding_accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Trade Outcome Distribution
        ax5 = axes[1, 1]
        trade_outcomes = ['Profitable', 'Losing']
        trade_counts = [self.accuracy_metrics['profitable_trades'], self.accuracy_metrics['losing_trades']]
        colors = ['#2E86AB', '#A23B72']
        
        bars = ax5.bar(trade_outcomes, trade_counts, color=colors)
        ax5.set_title('Trade Outcome Distribution', fontweight='bold')
        ax5.set_ylabel('Number of Trades')
        
        # Add value labels
        for bar, value in zip(bars, trade_counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + max(trade_counts)*0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Accuracy Gauge Chart
        ax6 = axes[1, 2]
        overall_acc = self.accuracy_metrics['overall_accuracy']
        
        # Create gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r_outer = 1
        r_inner = 0.7
        
        # Background arc
        ax6.fill_between(theta, r_inner, r_outer, color='lightgray', alpha=0.3)
        
        # Accuracy arc
        accuracy_theta = np.linspace(0, np.pi * (overall_acc/100), 100)
        if overall_acc >= 80:
            color = '#2E86AB'
        elif overall_acc >= 70:
            color = '#F18F01'
        elif overall_acc >= 60:
            color = '#C73E1D'
        else:
            color = '#A23B72'
        
        ax6.fill_between(accuracy_theta, r_inner, r_outer, color=color, alpha=0.8)
        
        ax6.set_xlim(-1.2, 1.2)
        ax6.set_ylim(-0.2, 1.2)
        ax6.set_aspect('equal')
        ax6.axis('off')
        ax6.text(0, 0.3, f'{overall_acc:.1f}%', ha='center', va='center', 
                fontsize=24, fontweight='bold')
        ax6.text(0, 0.1, 'Overall Accuracy', ha='center', va='center', fontsize=12)
        ax6.set_title('Model Accuracy Gauge', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_accuracy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Accuracy visualizations generated and saved!")
    
    def export_accuracy_report(self):
        """Export comprehensive accuracy report"""
        # Create detailed report
        report_data = []
        
        # Basic metrics
        report_data.append(['Metric', 'Value', 'Assessment'])
        report_data.append(['Overall Accuracy', f"{self.accuracy_metrics['overall_accuracy']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['overall_accuracy'])])
        report_data.append(['Direction Accuracy', f"{self.accuracy_metrics['direction_accuracy']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['direction_accuracy'])])
        report_data.append(['Profit Accuracy', f"{self.accuracy_metrics['profit_accuracy']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['profit_accuracy'])])
        report_data.append(['Precision', f"{self.accuracy_metrics['precision']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['precision'])])
        report_data.append(['Recall', f"{self.accuracy_metrics['recall']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['recall'])])
        report_data.append(['F1 Score', f"{self.accuracy_metrics['f1_score']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['f1_score'])])
        
        # Detailed metrics
        report_data.append(['', '', ''])
        report_data.append(['Position Type Analysis', '', ''])
        report_data.append(['LONG Accuracy', f"{self.accuracy_metrics['long_accuracy']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['long_accuracy'])])
        report_data.append(['SHORT Accuracy', f"{self.accuracy_metrics['short_accuracy']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['short_accuracy'])])
        
        report_data.append(['', '', ''])
        report_data.append(['Signal Quality Analysis', '', ''])
        report_data.append(['High Z-Score Accuracy', f"{self.accuracy_metrics['high_z_accuracy']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['high_z_accuracy'])])
        report_data.append(['Low Z-Score Accuracy', f"{self.accuracy_metrics['low_z_accuracy']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['low_z_accuracy'])])
        
        report_data.append(['', '', ''])
        report_data.append(['Timing Analysis', '', ''])
        report_data.append(['Quick Trade Accuracy', f"{self.accuracy_metrics['quick_trade_accuracy']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['quick_trade_accuracy'])])
        report_data.append(['Long Period Accuracy', f"{self.accuracy_metrics['long_period_accuracy']:.2f}%", 
                           self.get_accuracy_assessment(self.accuracy_metrics['long_period_accuracy'])])
        
        report_data.append(['', '', ''])
        report_data.append(['Trade Statistics', '', ''])
        report_data.append(['Total Predictions', str(self.accuracy_metrics['total_predictions']), ''])
        report_data.append(['Total Trades', str(self.accuracy_metrics['total_trades']), ''])
        report_data.append(['Profitable Trades', str(self.accuracy_metrics['profitable_trades']), ''])
        report_data.append(['Losing Trades', str(self.accuracy_metrics['losing_trades']), ''])
        
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data[1:], columns=report_data[0])
        report_df.to_csv('model_accuracy_report.csv', index=False)
        
        print("📊 Accuracy report exported to: model_accuracy_report.csv")
        return 'model_accuracy_report.csv'
    
    def get_accuracy_assessment(self, accuracy):
        """Get accuracy assessment"""
        if accuracy >= 80:
            return "EXCELLENT"
        elif accuracy >= 70:
            return "GOOD"
        elif accuracy >= 60:
            return "MODERATE"
        else:
            return "POOR"

# Main execution
if __name__ == "__main__":
    print("🎯 MODEL ACCURACY ANALYSIS")
    print("="*80)
    print("📊 Comprehensive accuracy analysis for pairs trading strategy...")
    
    # Initialize accuracy analysis
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    analyzer = ModelAccuracyAnalysis(data_folder)
    
    # Run accuracy analysis
    accuracy_metrics = analyzer.run_accuracy_analysis()
    
    # Export report
    analyzer.export_accuracy_report()
    
    print("\n🎉 MODEL ACCURACY ANALYSIS COMPLETED!")
    print("="*80)
    print("📁 Files generated:")
    print("  • model_accuracy_report.csv - Detailed accuracy metrics")
    print("  • model_accuracy_analysis.png - Visual accuracy charts")
    print("✅ Model accuracy analysis completed!")
