"""
TIME-OPTIMIZED PAIRS TRADING STRATEGY
Tests different trading times and schedules for optimal performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pairs_trading_pipeline import PairsTradingPipeline
from datetime import time, datetime
import warnings
warnings.filterwarnings('ignore')

class TimeOptimizedStrategy:
    """Time-optimized pairs trading strategy"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pipeline = PairsTradingPipeline(data_folder)
        self.time_results = {}
        self.best_time_config = {}
        
    def test_time_configurations(self, spread, z_score):
        """Test different time configurations"""
        print("🕐 TESTING TIME CONFIGURATIONS...")
        print("-" * 60)
        
        # Define time configurations to test
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
        
        results = []
        
        for config_name, config in time_configs.items():
            print(f"Testing: {config_name}")
            
            # Run strategy with this time configuration
            metrics = self.run_strategy_with_time_filter(
                spread, z_score, config['start_time'], config['end_time'], config['days']
            )
            
            results.append({
                'config': config_name,
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'profit_factor': metrics['profit_factor'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'total_trades': metrics['total_trades'],
                'accuracy': metrics['accuracy']
            })
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Find best configuration
        best_config = results_df.loc[results_df['total_return'].idxmax()]
        
        print(f"\n🏆 BEST TIME CONFIGURATION:")
        print("-" * 60)
        print(f"Configuration: {best_config['config']}")
        print(f"Total Return: {best_config['total_return']:.2f}%")
        print(f"Sharpe Ratio: {best_config['sharpe_ratio']:.3f}")
        print(f"Win Rate: {best_config['win_rate']:.1f}%")
        print(f"Total Trades: {best_config['total_trades']}")
        
        self.time_results = results_df
        self.best_time_config = time_configs[best_config['config']]
        
        return results_df, best_config
    
    def run_strategy_with_time_filter(self, spread, z_score, start_time=None, end_time=None, days=None):
        """Run strategy with time filtering"""
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
                # Still need to update equity and check exits
                if position != 0:
                    days_held += 1
                    
                    # Check if should exit (force exit on time restrictions)
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
                            
                            # Record accuracy
                            if position_type == 'LONG':
                                prediction = 1
                                actual = 1 if pnl > 0 else -1
                            else:
                                prediction = -1
                                actual = -1 if pnl > 0 else 1
                            
                            predictions.append(prediction)
                            actuals.append(actual)
                            
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
                continue
            
            # Normal trading logic for allowed times
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
                    
                    # Record accuracy
                    if position_type == 'LONG':
                        prediction = 1
                        actual = 1 if pnl > 0 else -1
                    else:
                        prediction = -1
                        actual = -1 if pnl > 0 else 1
                    
                    predictions.append(prediction)
                    actuals.append(actual)
                    
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
            
            # Calculate accuracy
            if len(predictions) > 0 and len(actuals) > 0:
                correct_predictions = sum(1 for pred, actual in zip(predictions, actuals) if pred == actual)
                accuracy = correct_predictions / len(predictions) * 100
            else:
                accuracy = 0
        else:
            profit_factor = 0
            win_rate = 0
            accuracy = 0
        
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
            'accuracy': accuracy
        }
    
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
    
    def run_time_optimization(self):
        """Run complete time optimization"""
        print("🕐 TIME-OPTIMIZED PAIRS TRADING STRATEGY")
        print("="*80)
        print("🎯 Testing different trading times and schedules...")
        
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
        
        # Test time configurations
        results_df, best_config = self.test_time_configurations(spread, z_score)
        
        # Run detailed analysis with best configuration
        print(f"\n🎯 DETAILED ANALYSIS WITH BEST CONFIGURATION: {best_config['config']}")
        print("-" * 60)
        
        detailed_results = self.run_detailed_time_analysis(
            spread, z_score, 
            self.best_time_config['start_time'],
            self.best_time_config['end_time'],
            self.best_time_config['days']
        )
        
        # Display comprehensive results
        self.display_time_optimization_results(results_df, best_config, detailed_results)
        
        # Generate visualizations
        self.generate_time_optimization_visualizations(results_df)
        
        return results_df, best_config, detailed_results
    
    def run_detailed_time_analysis(self, spread, z_score, start_time=None, end_time=None, days=None):
        """Run detailed analysis with specific time configuration"""
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
                            
                            # Record accuracy
                            if position_type == 'LONG':
                                prediction = 1
                                actual = 1 if pnl > 0 else -1
                            else:
                                prediction = -1
                                actual = -1 if pnl > 0 else 1
                            
                            predictions.append(prediction)
                            actuals.append(actual)
                            
                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': current_date,
                                'action': position_type,
                                'entry_price': entry_price,
                                'exit_price': current_spread,
                                'pnl': pnl,
                                'holding_period': days_held,
                                'exit_reason': exit_reason,
                                'entry_time': entry_date.time() if entry_date else None,
                                'exit_time': current_date.time()
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
                    
                    # Record accuracy
                    if position_type == 'LONG':
                        prediction = 1
                        actual = 1 if pnl > 0 else -1
                    else:
                        prediction = -1
                        actual = -1 if pnl > 0 else 1
                    
                    predictions.append(prediction)
                    actuals.append(actual)
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'action': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_spread,
                        'pnl': pnl,
                        'holding_period': days_held,
                        'exit_reason': exit_reason,
                        'entry_time': entry_date.time() if entry_date else None,
                        'exit_time': current_date.time()
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
        
        # Calculate detailed metrics
        equity_values = np.array(equity_curve)
        total_return = ((equity_values[-1] - initial_cash) / initial_cash) * 100
        
        years = len(equity_curve) / 252
        cagr = ((equity_values[-1] / initial_cash) ** (1/years) - 1) * 100 if years > 0 else 0
        
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            win_rate = (len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)) * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            expectancy_per_trade = trades_df['pnl'].mean()
            
            # Calculate accuracy
            if len(predictions) > 0 and len(actuals) > 0:
                correct_predictions = sum(1 for pred, actual in zip(predictions, actuals) if pred == actual)
                accuracy = correct_predictions / len(predictions) * 100
            else:
                accuracy = 0
        else:
            profit_factor = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            expectancy_per_trade = 0
            accuracy = 0
        
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
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy_per_trade': expectancy_per_trade,
            'total_trades': len(trades),
            'accuracy': accuracy,
            'trades_df': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'equity_curve': pd.Series(equity_curve, index=spread.index[:len(equity_curve)])
        }
    
    def display_time_optimization_results(self, results_df, best_config, detailed_results):
        """Display comprehensive time optimization results"""
        print("\n" + "="*80)
        print("📊 TIME OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"\n🏆 BEST TIME CONFIGURATION: {best_config['config']}")
        print("-" * 60)
        print(f"Total Return:                     {best_config['total_return']:+.2f}%")
        print(f"Sharpe Ratio:                     {best_config['sharpe_ratio']:+.3f}")
        print(f"Profit Factor:                    {best_config['profit_factor']:.2f}")
        print(f"Max Drawdown:                     {best_config['max_drawdown']:+.2f}%")
        print(f"Win Rate:                         {best_config['win_rate']:.1f}%")
        print(f"Total Trades:                     {best_config['total_trades']}")
        print(f"Accuracy:                         {best_config['accuracy']:.1f}%")
        
        print(f"\n📈 DETAILED METRICS FOR BEST CONFIG:")
        print("-" * 60)
        print(f"CAGR:                             {detailed_results['cagr']:+.2f}%")
        print(f"Avg Win:                          ${detailed_results['avg_win']:+,.2f}")
        print(f"Avg Loss:                         ${detailed_results['avg_loss']:+,.2f}")
        print(f"Expectancy per Trade:             ${detailed_results['expectancy_per_trade']:+.2f}")
        
        print(f"\n📊 ALL TIME CONFIGURATIONS RANKED BY RETURN:")
        print("-" * 60)
        ranked_results = results_df.sort_values('total_return', ascending=False)
        for i, (_, row) in enumerate(ranked_results.iterrows(), 1):
            status = "🏆" if i == 1 else "  " if i <= 3 else "  "
            print(f"{status} {i:2d}. {row['config']:20s}: {row['total_return']:+6.2f}% ({row['total_trades']:3d} trades, {row['accuracy']:.1f}% acc)")
        
        print(f"\n📊 ALL TIME CONFIGURATIONS RANKED BY ACCURACY:")
        print("-" * 60)
        ranked_accuracy = results_df.sort_values('accuracy', ascending=False)
        for i, (_, row) in enumerate(ranked_accuracy.iterrows(), 1):
            status = "🎯" if i == 1 else "  " if i <= 3 else "  "
            print(f"{status} {i:2d}. {row['config']:20s}: {row['accuracy']:5.1f}% ({row['total_trades']:3d} trades, {row['total_return']:+6.2f}% return)")
        
        # Performance analysis
        print(f"\n🚀 TIME OPTIMIZATION ANALYSIS:")
        print("-" * 60)
        
        if best_config['total_return'] > 50:
            print("🏆 EXCELLENT TIME OPTIMIZATION (>50% return)")
        elif best_config['total_return'] > 30:
            print("✅ GOOD TIME OPTIMIZATION (>30% return)")
        elif best_config['total_return'] > 10:
            print("⚠️  MODERATE TIME OPTIMIZATION (>10% return)")
        else:
            print("❌ POOR TIME OPTIMIZATION (<10% return)")
        
        if best_config['accuracy'] > 60:
            print("🎯 EXCELLENT TIMING ACCURACY (>60%)")
        elif best_config['accuracy'] > 50:
            print("✅ GOOD TIMING ACCURACY (>50%)")
        elif best_config['accuracy'] > 40:
            print("⚠️  MODERATE TIMING ACCURACY (>40%)")
        else:
            print("❌ POOR TIMING ACCURACY (<40%)")
        
        # Time-based insights
        print(f"\n💡 TIME-BASED INSIGHTS:")
        print("-" * 60)
        
        if 'Morning' in best_config['config']:
            print("✅ Morning trading shows best performance")
        elif 'Afternoon' in best_config['config']:
            print("✅ Afternoon trading shows best performance")
        elif 'Hour' in best_config['config']:
            print("✅ Specific hour trading shows best performance")
        elif 'Monday' in best_config['config'] or 'Friday' in best_config['config']:
            print("✅ Specific day trading shows best performance")
        else:
            print("✅ Full day trading shows best performance")
        
        print("\n" + "="*80)
    
    def generate_time_optimization_visualizations(self, results_df):
        """Generate time optimization visualizations"""
        print("\n📊 GENERATING TIME OPTIMIZATION VISUALIZATIONS...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TIME OPTIMIZATION ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Returns by Time Configuration
        ax1 = axes[0, 0]
        returns = results_df.sort_values('total_return')
        bars = ax1.barh(returns['config'], returns['total_return'], 
                       color=['#2E86AB' if x > 0 else '#A23B72' for x in returns['total_return']])
        ax1.set_title('Total Return by Time Configuration', fontweight='bold')
        ax1.set_xlabel('Total Return (%)')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, returns['total_return']):
            width = bar.get_width()
            ax1.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2,
                    f'{value:+.1f}%', ha='left' if width > 0 else 'right', va='center', fontweight='bold')
        
        # 2. Accuracy by Time Configuration
        ax2 = axes[0, 1]
        accuracy = results_df.sort_values('accuracy')
        bars = ax2.barh(accuracy['config'], accuracy['accuracy'], 
                       color=['#F18F01' if x > 50 else '#C73E1D' for x in accuracy['accuracy']])
        ax2.set_title('Accuracy by Time Configuration', fontweight='bold')
        ax2.set_xlabel('Accuracy (%)')
        ax2.set_xlim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars, accuracy['accuracy']):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', ha='left', va='center', fontweight='bold')
        
        # 3. Return vs Accuracy Scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(results_df['accuracy'], results_df['total_return'], 
                             s=results_df['total_trades']*10, alpha=0.7,
                             c=results_df['sharpe_ratio'], cmap='RdYlGn')
        ax3.set_xlabel('Accuracy (%)')
        ax3.set_ylabel('Total Return (%)')
        ax3.set_title('Return vs Accuracy (Size = Trades, Color = Sharpe)', fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axvline(x=50, color='black', linestyle='--', alpha=0.3)
        
        # Add labels for best configurations
        for _, row in results_df.iterrows():
            if row['total_return'] > 30 or row['accuracy'] > 55:
                ax3.annotate(row['config'], (row['accuracy'], row['total_return']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add colorbar for Sharpe ratio
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Sharpe Ratio')
        
        # 4. Trade Count by Configuration
        ax4 = axes[1, 1]
        trades = results_df.sort_values('total_trades')
        bars = ax4.barh(trades['config'], trades['total_trades'], color='#5C946E')
        ax4.set_title('Number of Trades by Time Configuration', fontweight='bold')
        ax4.set_xlabel('Number of Trades')
        
        # Add value labels
        for bar, value in zip(bars, trades['total_trades']):
            width = bar.get_width()
            ax4.text(width + max(trades['total_trades'])*0.01, bar.get_y() + bar.get_height()/2,
                    f'{value}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('time_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Time optimization visualizations generated and saved!")
    
    def export_time_optimization_report(self):
        """Export comprehensive time optimization report"""
        # Create detailed report
        report_data = []
        
        report_data.append(['Time Configuration', 'Total Return (%)', 'Sharpe Ratio', 'Profit Factor', 
                           'Max Drawdown (%)', 'Win Rate (%)', 'Total Trades', 'Accuracy (%)'])
        
        for _, row in self.time_results.iterrows():
            report_data.append([
                row['config'],
                f"{row['total_return']:.2f}",
                f"{row['sharpe_ratio']:.3f}",
                f"{row['profit_factor']:.2f}",
                f"{row['max_drawdown']:.2f}",
                f"{row['win_rate']:.1f}",
                str(row['total_trades']),
                f"{row['accuracy']:.1f}"
            ])
        
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data[1:], columns=report_data[0])
        report_df.to_csv('time_optimization_report.csv', index=False)
        
        print("📊 Time optimization report exported to: time_optimization_report.csv")
        return 'time_optimization_report.csv'

# Main execution
if __name__ == "__main__":
    print("🕐 TIME-OPTIMIZED PAIRS TRADING STRATEGY")
    print("="*80)
    print("🎯 Testing different trading times and schedules...")
    
    # Initialize time optimization
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    optimizer = TimeOptimizedStrategy(data_folder)
    
    # Run time optimization
    results_df, best_config, detailed_results = optimizer.run_time_optimization()
    
    # Export report
    optimizer.export_time_optimization_report()
    
    print("\n🎉 TIME OPTIMIZATION COMPLETED!")
    print("="*80)
    print("📁 Files generated:")
    print("  • time_optimization_report.csv - Detailed time analysis")
    print("  • time_optimization_analysis.png - Visual time charts")
    print("✅ Time optimization completed!")
