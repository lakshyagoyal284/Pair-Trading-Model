"""
ENHANCED BACKTESTING SYSTEM WITH CUMULATIVE RETURNS
Professional backtesting with comprehensive cumulative returns analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from logging_system import BacktestLogger, start_backtest_logging, stop_backtest_logging, log_backtest_section, log_backtest_metrics, log_backtest_trade
warnings.filterwarnings('ignore')

class EnhancedBacktestingWithCumulativeReturns:
    """
    Enhanced backtesting system with comprehensive cumulative returns analysis
    """
    
    def __init__(self, data_folder=".."):
        self.data_folder = data_folder
        self.models = {}
        self.backtest_results = {}
        self.portfolio_performance = {}
        self.cumulative_returns_data = {}
        self.logger = None
        
    def run_enhanced_backtesting_with_cumulative_returns(self):
        """Run comprehensive backtesting with cumulative returns analysis"""
        # Initialize logging
        self.logger = BacktestLogger("backtest_logs")
        self.logger.start_logging()
        
        print("🔄 ENHANCED BACKTESTING WITH CUMULATIVE RETURNS")
        print("="*80)
        print("📊 Multi-Timeframe Pairs Trading Backtesting")
        print("📈 Comprehensive Cumulative Returns Analysis")
        print("📝 All output will be saved to log file")
        print("="*80)
        
        try:
            # Step 1: Load trained models
            log_backtest_section("STEP 1: LOAD TRAINED MODELS")
            self.step1_load_trained_models()
            
            # Step 2: Load and prepare test data
            log_backtest_section("STEP 2: LOAD TEST DATA")
            self.step2_load_test_data()
            
            # Step 3: Run individual pair backtests with cumulative returns
            log_backtest_section("STEP 3: INDIVIDUAL PAIR BACKTESTS")
            self.step3_individual_pair_backtests_with_cumulative_returns()
            
            # Step 4: Portfolio-level backtesting with cumulative returns
            log_backtest_section("STEP 4: PORTFOLIO BACKTESTING")
            self.step4_portfolio_backtesting_with_cumulative_returns()
            
            # Step 5: Cumulative returns analysis
            log_backtest_section("STEP 5: CUMULATIVE RETURNS ANALYSIS")
            self.step5_cumulative_returns_analysis()
            
            # Step 6: Performance analysis
            log_backtest_section("STEP 6: PERFORMANCE ANALYSIS")
            self.step6_performance_analysis()
            
            # Step 7: Risk analysis
            log_backtest_section("STEP 7: RISK ANALYSIS")
            self.step7_risk_analysis()
            
            # Step 8: Generate comprehensive reports with cumulative returns
            log_backtest_section("STEP 8: GENERATE REPORTS & VISUALIZATIONS")
            self.step8_generate_comprehensive_reports()
            
            print("\n🎉 ENHANCED BACKTESTING WITH CUMULATIVE RETURNS COMPLETED!")
            print("="*80)
            
        except Exception as e:
            self.logger.log_error(f"Backtesting failed: {str(e)}")
            raise
        finally:
            # Always stop logging
            self.logger.stop_logging()
        
    def step1_load_trained_models(self):
        """Load trained models from enhanced data trainer"""
        print("\n🤖 STEP 1: LOAD TRAINED MODELS")
        print("-" * 50)
        
        try:
            # Load the enhanced data trainer to get models
            from enhanced_data_trainer import EnhancedDataTrainer
            
            trainer = EnhancedDataTrainer()
            
            # Run a quick training to get models
            print("📊 Loading trained models...")
            
            trainer.run_enhanced_training()
            self.models = trainer.models
            
            print(f"✅ Loaded {len(self.models)} trained models")
            
            # Display model summary
            print("\n📈 MODEL SUMMARY:")
            for i, (pair, data) in enumerate(self.models.items(), 1):
                stock1, stock2 = pair
                print(f"  {i}. {stock1} - {stock2}: {data['test_accuracy']:.3f} accuracy")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.create_sample_models()
    
    def create_sample_models(self):
        """Create sample models for demonstration"""
        print("📊 Creating sample models...")
        
        # Sample pairs with their characteristics
        sample_pairs = [
            ('AAREYDRUGS', 'AJRINFRA', 0.952),
            ('ABSLAMC', 'APCL', 0.942),
            ('AGSTRA', 'ANURAS', 0.935),
            ('AARVI', 'ADANITRANS', 0.913),
            ('AAVAS', 'AMRUTANJAN', 0.911)
        ]
        
        for stock1, stock2, accuracy in sample_pairs:
            # Create a simple Random Forest model
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            # Sample hedge ratios
            hedge_ratios = {
                '3minute': np.random.uniform(0.5, 3.0),
                '5minute': np.random.uniform(0.5, 3.0),
                '10minute': np.random.uniform(0.5, 3.0),
                '15minute': np.random.uniform(0.5, 3.0)
            }
            
            self.models[(stock1, stock2)] = {
                'model': model,
                'hedge_ratios': hedge_ratios,
                'test_accuracy': accuracy,
                'features': [f'{tf}_z_score' for tf in ['3minute', '5minute', '10minute', '15minute']] + 
                           [f'{tf}_momentum' for tf in ['3minute', '5minute', '10minute', '15minute']] +
                           [f'{tf}_volatility' for tf in ['3minute', '5minute', '10minute', '15minute']] +
                           [f'{tf}_trend' for tf in ['3minute', '5minute', '10minute', '15minute']]
            }
        
        print(f"✅ Created {len(self.models)} sample models")
    
    def step2_load_test_data(self):
        """Load and prepare test data for backtesting"""
        print("\n📂 STEP 2: LOAD TEST DATA")
        print("-" * 50)
        
        try:
            # Load data from 3minute folder (primary test data)
            import glob
            
            data_folder = os.path.join(self.data_folder, "3minute")
            csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
            
            print(f"📊 Found {len(csv_files)} CSV files")
            
            # Load data for stocks in our models
            all_stocks = set()
            for pair in self.models.keys():
                all_stocks.update(pair)
            
            stock_data = {}
            processed_count = 0
            
            for file in csv_files:
                try:
                    stock_name = os.path.basename(file).replace('.csv', '')
                    
                    if stock_name in all_stocks:
                        df = pd.read_csv(file)
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        
                        # Resample to daily data
                        daily_data = df.resample('D').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                        
                        stock_data[stock_name] = daily_data
                        processed_count += 1
                        
                        if processed_count % 5 == 0:
                            print(f"  Processed {processed_count} stocks")
                
                except Exception as e:
                    continue
            
            if stock_data:
                self.test_data = pd.DataFrame({
                    stock: data['close'] for stock, data in stock_data.items()
                }).dropna()
                
                print(f"✅ Loaded test data for {len(stock_data)} stocks")
                print(f"📅 Date range: {self.test_data.index.min()} to {self.test_data.index.max()}")
                print(f"📊 Total trading days: {len(self.test_data)}")
            else:
                print("❌ No test data loaded")
                self.create_sample_test_data()
                
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            self.create_sample_test_data()
    
    def create_sample_test_data(self):
        """Create sample test data for demonstration"""
        print("📊 Creating sample test data...")
        
        # Get all stocks from models
        all_stocks = set()
        for pair in self.models.keys():
            all_stocks.update(pair)
        
        # Create sample price data
        date_range = pd.date_range('2022-01-01', '2022-12-31', freq='D')
        n_days = len(date_range)
        
        sample_data = {}
        
        for stock in sorted(all_stocks):
            # Generate realistic price series
            np.random.seed(hash(stock) % 1000)  # Consistent random seed per stock
            
            base_price = np.random.uniform(100, 1000)
            returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
            prices = [base_price]
            
            for ret in returns:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1))  # Ensure positive prices
            
            sample_data[stock] = pd.Series(prices[1:], index=date_range)
        
        self.test_data = pd.DataFrame(sample_data)
        print(f"✅ Created sample test data for {len(sample_data)} stocks")
        print(f"📅 Date range: {self.test_data.index.min()} to {self.test_data.index.max()}")
    
    def step3_individual_pair_backtests_with_cumulative_returns(self):
        """Run backtests for individual pairs with cumulative returns tracking"""
        print("\n🔄 STEP 3: INDIVIDUAL PAIR BACKTESTS")
        print("-" * 50)
        
        for i, (pair, model_data) in enumerate(self.models.items(), 1):
            stock1, stock2 = pair
            print(f"\n📈 Backtesting {stock1} - {stock2}...")
            
            try:
                result = self.backtest_single_pair_with_cumulative_returns(pair, model_data)
                self.backtest_results[pair] = result
                
                # Log detailed metrics
                metrics = {
                    "Total Return (%)": result['total_return'],
                    "Cumulative Return (%)": result['cumulative_return'],
                    "Annualized Return (%)": result['annualized_return'],
                    "Sharpe Ratio": result['sharpe_ratio'],
                    "Win Rate (%)": result['win_rate'],
                    "Max Drawdown (%)": result['max_drawdown'],
                    "Total Trades": result['total_trades']
                }
                log_backtest_metrics(metrics, f"{stock1}-{stock2} RESULTS")
                
                # Log individual trades
                if result['trades']:
                    print(f"  📊 Trade Details:")
                    for j, trade in enumerate(result['trades'][:3], 1):  # Log first 3 trades
                        trade_info = {
                            'pair': f"{stock1}-{stock2}",
                            'action': trade['action'],
                            'entry_date': trade['entry_date'],
                            'exit_date': trade['exit_date'],
                            'pnl': trade['pnl'],
                            'cumulative_return': trade.get('cumulative_return', 0),
                            'exit_reason': trade['exit_reason']
                        }
                        log_backtest_trade(trade_info)
                
                print(f"  ✅ Total Return: {result['total_return']:+.2f}%")
                print(f"  📈 Cumulative Return: {result['cumulative_return']:+.2f}%")
                print(f"  📊 Annualized Return: {result['annualized_return']:+.2f}%")
                print(f"  📊 Sharpe Ratio: {result['sharpe_ratio']:+.3f}")
                print(f"  🎯 Win Rate: {result['win_rate']:.1f}%")
                print(f"  📉 Max Drawdown: {result['max_drawdown']:+.2f}%")
                
            except Exception as e:
                self.logger.log_error(f"Backtest failed for {stock1}-{stock2}: {str(e)}")
                continue
        
        print(f"\n✅ Completed backtests for {len(self.backtest_results)} pairs")
    
    def backtest_single_pair_with_cumulative_returns(self, pair, model_data):
        """Backtest a single trading pair with cumulative returns tracking"""
        stock1, stock2 = pair
        
        # Get price data for the pair
        if stock1 not in self.test_data.columns or stock2 not in self.test_data.columns:
            raise ValueError(f"Price data not available for {pair}")
        
        prices1 = self.test_data[stock1].dropna()
        prices2 = self.test_data[stock2].dropna()
        
        # Align data
        combined = pd.DataFrame({'stock1': prices1, 'stock2': prices2}).dropna()
        
        if len(combined) < 50:
            raise ValueError("Insufficient data for backtesting")
        
        # Calculate spread with hedge ratio
        hedge_ratio = model_data['hedge_ratios']['3minute']  # Use 3-minute hedge ratio
        spread = combined['stock1'] - hedge_ratio * combined['stock2']
        
        # Calculate technical indicators
        spread_mean = spread.rolling(20).mean()
        spread_std = spread.rolling(20).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Generate trading signals using the model
        features = pd.DataFrame(index=combined.index)
        
        # Create features (simplified for backtesting)
        features['z_score'] = z_score
        features['momentum'] = spread.pct_change(5)
        features['volatility'] = spread.rolling(10).std()
        features['trend'] = spread.rolling(20).mean() - spread.rolling(50).mean()
        
        # Make predictions (simplified - use z-score thresholds if model not trained)
        try:
            # Try to use the trained model
            feature_data = features.dropna()
            if len(feature_data) > 0 and hasattr(model_data['model'], 'predict'):
                predictions = model_data['model'].predict(feature_data)
                signals = pd.Series(predictions, index=feature_data.index)
            else:
                # Fall back to z-score based signals
                signals = pd.Series(0, index=z_score.index)
                signals[z_score > 2.0] = 1  # Short spread
                signals[z_score < -2.0] = -1  # Long spread
                signals[(z_score > -0.5) & (z_score < 0.5)] = 0  # Neutral
        except:
            # Fall back to z-score based signals
            signals = pd.Series(0, index=z_score.index)
            signals[z_score > 2.0] = 1
            signals[z_score < -2.0] = -1
            signals[(z_score > -0.5) & (z_score < 0.5)] = 0
        
        # Backtesting parameters
        initial_cash = 100000
        commission = 0.001
        position_size = 0.3
        max_hold_days = 5
        
        # Backtesting logic with cumulative returns tracking
        cash = initial_cash
        position = 0
        entry_price = None
        entry_date = None
        trades = []
        equity_curve = []
        daily_returns = []
        cumulative_returns = []
        
        for i in range(len(spread)):
            current_date = spread.index[i]
            current_spread = spread.iloc[i]
            current_signal = signals.iloc[i] if i < len(signals) else 0
            stock1_price = combined['stock1'].iloc[i]
            stock2_price = combined['stock2'].iloc[i]
            
            # Skip if no signal
            if pd.isna(current_signal):
                current_equity = cash
                if position != 0:
                    unrealized_pnl = (current_spread - entry_price) * position
                    current_equity += unrealized_pnl
                equity_curve.append(current_equity)
                
                # Calculate daily return
                if len(equity_curve) > 1:
                    daily_return = (current_equity - equity_curve[-2]) / equity_curve[-2]
                    daily_returns.append(daily_return)
                    
                    # Calculate cumulative return
                    if len(cumulative_returns) == 0:
                        cumulative_returns.append(daily_return)
                    else:
                        cumulative_returns.append(cumulative_returns[-1] + daily_return)
                else:
                    daily_returns.append(0)
                    cumulative_returns.append(0)
                continue
            
            # Exit logic
            if position != 0:
                days_held = (current_date - entry_date).days
                
                should_exit = False
                exit_reason = ""
                
                if abs(current_signal) < 0.5:  # Signal changed to neutral
                    should_exit = True
                    exit_reason = "Signal neutral"
                elif days_held >= max_hold_days:
                    should_exit = True
                    exit_reason = "Max hold period"
                elif position > 0 and current_spread < entry_price * 0.95:  # Stop loss
                    should_exit = True
                    exit_reason = "Stop loss"
                elif position < 0 and current_spread > entry_price * 1.05:  # Stop loss
                    should_exit = True
                    exit_reason = "Stop loss"
                
                if should_exit:
                    # Calculate P&L
                    pnl = (current_spread - entry_price) * abs(position) * initial_cash * position_size / entry_price
                    cash += pnl
                    cash -= abs(pnl) * commission
                    
                    # Calculate cumulative return for this trade
                    trade_return = pnl / (initial_cash * position_size)
                    if len(cumulative_returns) > 0:
                        trade_cumulative_return = cumulative_returns[-1] + trade_return
                    else:
                        trade_cumulative_return = trade_return
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'action': 'LONG' if position > 0 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_spread,
                        'pnl': pnl,
                        'holding_period': days_held,
                        'exit_reason': exit_reason,
                        'cumulative_return': trade_cumulative_return,
                        'trade_return': trade_return
                    })
                    
                    position = 0
                    entry_price = None
                    entry_date = None
            
            # Entry logic
            if position == 0 and abs(current_signal) > 0.5:
                if current_signal > 0:  # Short spread signal
                    position = -1
                    entry_price = current_spread
                    entry_date = current_date
                elif current_signal < 0:  # Long spread signal
                    position = 1
                    entry_price = current_spread
                    entry_date = current_date
            
            # Calculate current equity
            current_equity = cash
            if position != 0:
                unrealized_pnl = (current_spread - entry_price) * position * initial_cash * position_size / entry_price
                current_equity += unrealized_pnl
            
            equity_curve.append(current_equity)
            
            # Calculate daily return
            if len(equity_curve) > 1:
                daily_return = (current_equity - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)
                
                # Calculate cumulative return
                if len(cumulative_returns) == 0:
                    cumulative_returns.append(daily_return)
                else:
                    cumulative_returns.append(cumulative_returns[-1] + daily_return)
            else:
                daily_returns.append(0)
                cumulative_returns.append(0)
        
        # Calculate performance metrics
        total_return = ((equity_curve[-1] - initial_cash) / initial_cash) * 100
        cumulative_return = cumulative_returns[-1] * 100 if cumulative_returns else 0
        
        # Calculate annualized return
        years = len(equity_curve) / 252
        annualized_return = ((equity_curve[-1] / initial_cash) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate Sharpe ratio
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        
        # Calculate trade statistics
        total_trades = len(trades)
        if total_trades > 0:
            trades_df = pd.DataFrame(trades)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = (len(winning_trades) / total_trades) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Store cumulative returns data
        self.cumulative_returns_data[pair] = {
            'dates': spread.index,
            'cumulative_returns': cumulative_returns,
            'equity_curve': equity_curve,
            'daily_returns': daily_returns
        }
        
        return {
            'total_return': total_return,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
            'trades': trades,
            'daily_returns': daily_returns,
            'cumulative_returns_series': cumulative_returns
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
    
    def step4_portfolio_backtesting_with_cumulative_returns(self):
        """Run portfolio-level backtesting with cumulative returns"""
        print("\n💼 STEP 4: PORTFOLIO BACKTESTING")
        print("-" * 50)
        
        if not self.backtest_results:
            print("❌ No individual backtest results available")
            return
        
        # Combine all pairs into a portfolio
        portfolio_equity = []
        portfolio_trades = []
        portfolio_cumulative_returns = []
        
        # Equal weight allocation
        weight_per_pair = 1.0 / len(self.backtest_results)
        initial_cash = 100000
        
        print(f"📊 Creating portfolio with {len(self.backtest_results)} pairs")
        print(f"💰 Weight per pair: {weight_per_pair:.2%}")
        
        # Log portfolio configuration
        portfolio_config = {
            "Number of Pairs": len(self.backtest_results),
            "Weight per Pair": f"{weight_per_pair:.2%}",
            "Initial Capital": initial_cash,
            "Portfolio Type": "Equal Weight"
        }
        log_backtest_metrics(portfolio_config, "PORTFOLIO CONFIGURATION")
        
        # Find the longest equity curve
        max_length = max(len(result['equity_curve']) for result in self.backtest_results.values())
        
        for i in range(max_length):
            daily_return = 0
            
            for pair, result in self.backtest_results.items():
                if i < len(result['equity_curve']):
                    pair_equity = result['equity_curve'][i]
                    pair_return = (pair_equity - initial_cash) / initial_cash
                    daily_return += pair_return * weight_per_pair
            
            portfolio_value = initial_cash * (1 + daily_return)
            portfolio_equity.append(portfolio_value)
            
            # Calculate portfolio cumulative return
            if i == 0:
                portfolio_cumulative_returns.append(daily_return)
            else:
                portfolio_cumulative_returns.append(portfolio_cumulative_returns[-1] + daily_return)
        
        # Aggregate all trades
        for pair, result in self.backtest_results.items():
            for trade in result['trades']:
                trade['pair'] = f"{pair[0]}-{pair[1]}"
                portfolio_trades.append(trade)
        
        # Calculate portfolio metrics
        self.portfolio_performance = self.calculate_portfolio_metrics(
            portfolio_equity, portfolio_trades, portfolio_cumulative_returns, initial_cash
        )
        
        # Store portfolio cumulative returns
        self.cumulative_returns_data['PORTFOLIO'] = {
            'dates': list(range(len(portfolio_equity))),
            'cumulative_returns': portfolio_cumulative_returns,
            'equity_curve': portfolio_equity,
            'daily_returns': [(portfolio_equity[i] - portfolio_equity[i-1]) / portfolio_equity[i-1] if i > 0 else 0 for i in range(len(portfolio_equity))]
        }
        
        # Log portfolio performance
        portfolio_metrics = {
            "Portfolio Return (%)": self.portfolio_performance['total_return'],
            "Portfolio Cumulative Return (%)": self.portfolio_performance['cumulative_return'],
            "Portfolio Annualized Return (%)": self.portfolio_performance['annualized_return'],
            "Portfolio Sharpe": self.portfolio_performance['sharpe_ratio'],
            "Portfolio Win Rate (%)": self.portfolio_performance['win_rate'],
            "Portfolio Max DD (%)": self.portfolio_performance['max_drawdown'],
            "Total Portfolio Trades": self.portfolio_performance['total_trades']
        }
        log_backtest_metrics(portfolio_metrics, "PORTFOLIO PERFORMANCE")
        
        print(f"✅ Portfolio backtesting completed")
        print(f"📈 Portfolio Total Return: {self.portfolio_performance['total_return']:+.2f}%")
        print(f"📈 Portfolio Cumulative Return: {self.portfolio_performance['cumulative_return']:+.2f}%")
        print(f"📊 Portfolio Annualized Return: {self.portfolio_performance['annualized_return']:+.2f}%")
        print(f"📊 Portfolio Sharpe Ratio: {self.portfolio_performance['sharpe_ratio']:+.3f}")
        print(f"🎯 Portfolio Win Rate: {self.portfolio_performance['win_rate']:.1f}%")
        print(f"📉 Portfolio Max Drawdown: {self.portfolio_performance['max_drawdown']:+.2f}%")
    
    def calculate_portfolio_metrics(self, equity_curve, trades, cumulative_returns, initial_cash):
        """Calculate comprehensive portfolio metrics"""
        # Calculate returns
        total_return = ((equity_curve[-1] - initial_cash) / initial_cash) * 100
        cumulative_return = cumulative_returns[-1] * 100 if cumulative_returns else 0
        
        # Calculate annualized return
        years = len(equity_curve) / 252
        annualized_return = ((equity_curve[-1] / initial_cash) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate Sharpe ratio
        daily_returns = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1] if i > 0 else 0 for i in range(len(equity_curve))]
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        
        # Calculate trade statistics
        total_trades = len(trades)
        if total_trades > 0:
            trades_df = pd.DataFrame(trades)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = (len(winning_trades) / total_trades) * 100
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
            'trades': trades,
            'daily_returns': daily_returns,
            'cumulative_returns_series': cumulative_returns
        }
    
    def step5_cumulative_returns_analysis(self):
        """Analyze cumulative returns across all pairs and portfolio"""
        print("\n📈 STEP 5: CUMULATIVE RETURNS ANALYSIS")
        print("-" * 50)
        
        print("📊 CUMULATIVE RETURNS SUMMARY:")
        print("-" * 60)
        
        # Analyze individual pairs
        for pair, data in self.cumulative_returns_data.items():
            if pair != 'PORTFOLIO':
                stock1, stock2 = pair
                final_cumulative_return = data['cumulative_returns'][-1] * 100 if data['cumulative_returns'] else 0
                max_cumulative_return = max(data['cumulative_returns']) * 100 if data['cumulative_returns'] else 0
                min_cumulative_return = min(data['cumulative_returns']) * 100 if data['cumulative_returns'] else 0
                
                print(f"  {stock1}-{stock2}:")
                print(f"    Final Cumulative Return: {final_cumulative_return:+.2f}%")
                print(f"    Maximum Cumulative Return: {max_cumulative_return:+.2f}%")
                print(f"    Minimum Cumulative Return: {min_cumulative_return:+.2f}%")
                print()
        
        # Portfolio analysis
        if 'PORTFOLIO' in self.cumulative_returns_data:
            portfolio_data = self.cumulative_returns_data['PORTFOLIO']
            final_cumulative_return = portfolio_data['cumulative_returns'][-1] * 100 if portfolio_data['cumulative_returns'] else 0
            max_cumulative_return = max(portfolio_data['cumulative_returns']) * 100 if portfolio_data['cumulative_returns'] else 0
            min_cumulative_return = min(portfolio_data['cumulative_returns']) * 100 if portfolio_data['cumulative_returns'] else 0
            
            print("💼 PORTFOLIO:")
            print(f"    Final Cumulative Return: {final_cumulative_return:+.2f}%")
            print(f"    Maximum Cumulative Return: {max_cumulative_return:+.2f}%")
            print(f"    Minimum Cumulative Return: {min_cumulative_return:+.2f}%")
        
        print("-" * 60)
    
    def step6_performance_analysis(self):
        """Analyze overall performance with cumulative returns focus"""
        print("\n📈 STEP 6: PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        if not self.backtest_results:
            print("❌ No backtest results to analyze")
            return
        
        # Individual pair performance summary
        print("📊 INDIVIDUAL PAIR PERFORMANCE")
        print("-" * 40)
        
        sorted_results = sorted(
            self.backtest_results.items(),
            key=lambda x: x[1]['cumulative_return'],
            reverse=True
        )
        
        for i, (pair, result) in enumerate(sorted_results, 1):
            stock1, stock2 = pair
            print(f"{i:2d}. {stock1}-{stock2}: "
                  f"Cumulative Return: {result['cumulative_return']:+6.2f}%, "
                  f"Total Return: {result['total_return']:+6.2f}%, "
                  f"Sharpe: {result['sharpe_ratio']:+5.3f}, "
                  f"Win Rate: {result['win_rate']:5.1f}%")
        
        # Portfolio vs individual comparison
        if self.portfolio_performance:
            print(f"\n💼 PORTFOLIO PERFORMANCE")
            print("-" * 40)
            print(f"Portfolio Cumulative Return:     {self.portfolio_performance['cumulative_return']:+.2f}%")
            print(f"Portfolio Total Return:           {self.portfolio_performance['total_return']:+.2f}%")
            print(f"Portfolio Annualized Return:       {self.portfolio_performance['annualized_return']:+.2f}%")
            print(f"Portfolio Sharpe:                 {self.portfolio_performance['sharpe_ratio']:+.3f}")
            print(f"Portfolio Win Rate:               {self.portfolio_performance['win_rate']:.1f}%")
            print(f"Portfolio Max DD:                 {self.portfolio_performance['max_drawdown']:+.2f}%")
            
            # Calculate average individual performance
            avg_cumulative_return = np.mean([r['cumulative_return'] for r in self.backtest_results.values()])
            avg_total_return = np.mean([r['total_return'] for r in self.backtest_results.values()])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in self.backtest_results.values()])
            avg_win_rate = np.mean([r['win_rate'] for r in self.backtest_results.values()])
            
            print(f"\n📊 AVERAGE INDIVIDUAL PERFORMANCE")
            print("-" * 40)
            print(f"Avg Cumulative Return:     {avg_cumulative_return:+.2f}%")
            print(f"Avg Total Return:           {avg_total_return:+.2f}%")
            print(f"Avg Sharpe:                 {avg_sharpe:+.3f}")
            print(f"Avg Win Rate:               {avg_win_rate:.1f}%")
            
            # Portfolio benefits
            print(f"\n🎯 PORTFOLIO BENEFITS")
            print("-" * 40)
            print(f"Cumulative Return Bonus:    {self.portfolio_performance['cumulative_return'] - avg_cumulative_return:+.2f}%")
            print(f"Total Return Bonus:         {self.portfolio_performance['total_return'] - avg_total_return:+.2f}%")
            print(f"Risk Reduction:             {avg_sharpe - self.portfolio_performance['sharpe_ratio']:+.3f} Sharpe")
    
    def step7_risk_analysis(self):
        """Analyze risk metrics with cumulative returns focus"""
        print("\n⚠️  STEP 7: RISK ANALYSIS")
        print("-" * 50)
        
        if not self.backtest_results:
            print("❌ No backtest results for risk analysis")
            return
        
        # Calculate risk metrics for all pairs
        all_returns = []
        all_drawdowns = []
        all_cumulative_returns = []
        
        for pair, result in self.backtest_results.items():
            equity_values = np.array(result['equity_curve'])
            returns = pd.Series(equity_values).pct_change().dropna()
            all_returns.extend(returns)
            
            max_dd = self.calculate_max_drawdown(equity_values)
            all_drawdowns.append(max_dd)
            
            if result['cumulative_returns_series']:
                all_cumulative_returns.append(result['cumulative_returns_series'][-1])
        
        # Portfolio risk metrics
        if self.portfolio_performance:
            portfolio_returns = pd.Series(self.portfolio_performance['daily_returns'])
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            portfolio_var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
            
            print("💼 PORTFOLIO RISK METRICS")
            print("-" * 40)
            print(f"Annual Volatility:     {portfolio_volatility:.2%}")
            print(f"95% VaR (annual):      {portfolio_var_95:.2%}")
            print(f"Max Drawdown:          {self.portfolio_performance['max_drawdown']:.2f}%")
            print(f"Calmar Ratio:          {self.portfolio_performance['total_return']/abs(self.portfolio_performance['max_drawdown']):.3f}")
            print(f"Cumulative Return Vol: {np.std(all_cumulative_returns)*100:.2f}%")
        
        # Individual pair risk analysis
        avg_drawdown = np.mean(all_drawdowns)
        worst_drawdown = np.max(all_drawdowns)
        avg_cumulative_return = np.mean(all_cumulative_returns)
        cumulative_volatility = np.std(all_cumulative_returns) * 100
        
        print(f"\n📊 INDIVIDUAL PAIR RISK")
        print("-" * 40)
        print(f"Average Max Drawdown:  {avg_drawdown:.2f}%")
        print(f"Worst Max Drawdown:    {worst_drawdown:.2f}%")
        print(f"Drawdown Consistency:  {'High' if np.std(all_drawdowns) < 5 else 'Low'}")
        print(f"Avg Cumulative Return: {avg_cumulative_return:+.2f}%")
        print(f"Cumulative Volatility: {cumulative_volatility:.2f}%")
        
        # Correlation analysis (if multiple pairs)
        if len(self.backtest_results) > 1:
            print(f"\n🔗 CORRELATION ANALYSIS")
            print("-" * 40)
            
            # Calculate correlation between pair returns
            pair_returns = {}
            for pair, result in self.backtest_results.items():
                equity_values = np.array(result['equity_curve'])
                returns = pd.Series(equity_values).pct_change().dropna()
                pair_returns[pair] = returns
            
            # Create correlation matrix
            returns_df = pd.DataFrame(pair_returns)
            correlation_matrix = returns_df.corr()
            
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            print(f"Average Pair Correlation: {avg_correlation:.3f}")
            print(f"Diversification Benefit:  {'High' if avg_correlation < 0.5 else 'Moderate' if avg_correlation < 0.7 else 'Low'}")
    
    def step8_generate_comprehensive_reports(self):
        """Generate comprehensive reports with cumulative returns"""
        print("\n📊 STEP 8: GENERATE REPORTS & VISUALIZATIONS")
        print("-" * 50)
        
        # Generate CSV report with cumulative returns
        self.generate_comprehensive_csv_report()
        
        # Generate visualizations with cumulative returns
        self.generate_cumulative_returns_visualizations()
        
        # Generate cumulative returns analysis report
        self.generate_cumulative_returns_analysis_report()
        
        print("✅ Comprehensive reports and visualizations generated!")
    
    def generate_comprehensive_csv_report(self):
        """Generate comprehensive CSV report with cumulative returns"""
        report_data = []
        
        # Header
        report_data.append([
            'Pair', 'Total_Return', 'Cumulative_Return', 'Annualized_Return', 'Sharpe_Ratio', 'Max_Drawdown', 
            'Total_Trades', 'Win_Rate', 'Profit_Factor', 'Max_Cumulative_Return', 'Min_Cumulative_Return', 'Status'
        ])
        
        # Add individual pair results
        for pair, result in self.backtest_results.items():
            stock1, stock2 = pair
            
            # Get cumulative returns metrics
            cumulative_returns = result['cumulative_returns_series']
            max_cumulative = max(cumulative_returns) * 100 if cumulative_returns else 0
            min_cumulative = min(cumulative_returns) * 100 if cumulative_returns else 0
            
            status = "EXCELLENT" if result['cumulative_return'] > 20 and result['sharpe_ratio'] > 1.0 else \
                    "GOOD" if result['cumulative_return'] > 10 and result['sharpe_ratio'] > 0.5 else \
                    "FAIR" if result['cumulative_return'] > 0 else "POOR"
            
            report_data.append([
                f"{stock1}-{stock2}",
                f"{result['total_return']:.2f}",
                f"{result['cumulative_return']:.2f}",
                f"{result['annualized_return']:.2f}",
                f"{result['sharpe_ratio']:.3f}",
                f"{result['max_drawdown']:.2f}",
                result['total_trades'],
                f"{result['win_rate']:.1f}",
                f"{result['profit_factor']:.2f}",
                f"{max_cumulative:.2f}",
                f"{min_cumulative:.2f}",
                status
            ])
        
        # Add portfolio
        if self.portfolio_performance:
            portfolio_cumulative = self.portfolio_performance['cumulative_returns_series']
            max_cumulative = max(portfolio_cumulative) * 100 if portfolio_cumulative else 0
            min_cumulative = min(portfolio_cumulative) * 100 if portfolio_cumulative else 0
            
            report_data.append([
                "PORTFOLIO",
                f"{self.portfolio_performance['total_return']:.2f}",
                f"{self.portfolio_performance['cumulative_return']:.2f}",
                f"{self.portfolio_performance['annualized_return']:.2f}",
                f"{self.portfolio_performance['sharpe_ratio']:.3f}",
                f"{self.portfolio_performance['max_drawdown']:.2f}",
                self.portfolio_performance['total_trades'],
                f"{self.portfolio_performance['win_rate']:.1f}",
                f"{self.portfolio_performance['profit_factor']:.2f}",
                f"{max_cumulative:.2f}",
                f"{min_cumulative:.2f}",
                "PORTFOLIO"
            ])
        
        # Create DataFrame and save
        report_df = pd.DataFrame(report_data[1:], columns=report_data[0])
        report_df.to_csv('enhanced_backtesting_cumulative_returns_report.csv', index=False)
        
        print("📊 Comprehensive backtesting report exported to: enhanced_backtesting_cumulative_returns_report.csv")
    
    def generate_cumulative_returns_visualizations(self):
        """Generate visualizations with cumulative returns focus"""
        print("📊 Generating cumulative returns visualizations...")
        
        if not self.cumulative_returns_data:
            print("❌ No cumulative returns data to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('ENHANCED BACKTESTING WITH CUMULATIVE RETURNS ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Cumulative Returns Comparison
        ax1 = axes[0, 0]
        for pair, data in list(self.cumulative_returns_data.items())[:4]:
            if pair != 'PORTFOLIO':
                stock1, stock2 = pair
                cumulative_pct = [r * 100 for r in data['cumulative_returns']]
                ax1.plot(data['dates'], cumulative_pct, label=f'{stock1}-{stock2}', alpha=0.7)
        
        if 'PORTFOLIO' in self.cumulative_returns_data:
            portfolio_data = self.cumulative_returns_data['PORTFOLIO']
            portfolio_cumulative_pct = [r * 100 for r in portfolio_data['cumulative_returns']]
            ax1.plot(portfolio_data['dates'], portfolio_cumulative_pct, 'k--', linewidth=3, label='PORTFOLIO')
        
        ax1.set_title('Cumulative Returns Comparison', fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Equity Curves
        ax2 = axes[0, 1]
        for pair, data in list(self.cumulative_returns_data.items())[:4]:
            if pair != 'PORTFOLIO':
                stock1, stock2 = pair
                ax2.plot(data['dates'], data['equity_curve'], label=f'{stock1}-{stock2}', alpha=0.7)
        
        if 'PORTFOLIO' in self.cumulative_returns_data:
            portfolio_data = self.cumulative_returns_data['PORTFOLIO']
            ax2.plot(portfolio_data['dates'], portfolio_data['equity_curve'], 'k--', linewidth=3, label='PORTFOLIO')
        
        ax2.set_title('Equity Curves', fontweight='bold')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Return Distribution
        ax3 = axes[0, 2]
        total_returns = [result['total_return'] for result in self.backtest_results.values()]
        cumulative_returns = [result['cumulative_return'] for result in self.backtest_results.values()]
        
        ax3.scatter(total_returns, cumulative_returns, alpha=0.7, s=100)
        if self.portfolio_performance:
            ax3.scatter(self.portfolio_performance['total_return'], 
                       self.portfolio_performance['cumulative_return'], 
                       color='red', s=200, marker='*', label='Portfolio')
        
        ax3.set_xlabel('Total Return (%)')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.set_title('Total vs Cumulative Returns', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Returns Distribution
        ax4 = axes[1, 0]
        cumulative_returns_list = [result['cumulative_return'] for result in self.backtest_results.values()]
        ax4.hist(cumulative_returns_list, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        if self.portfolio_performance:
            ax4.axvline(self.portfolio_performance['cumulative_return'], color='red', 
                        linestyle='--', linewidth=2, label='Portfolio')
        ax4.set_xlabel('Cumulative Return (%)')
        ax4.set_ylabel('Number of Pairs')
        ax4.set_title('Cumulative Returns Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # 5. Maximum Cumulative Returns
        ax5 = axes[1, 1]
        max_cumulative_returns = []
        pair_names = []
        
        for pair, data in self.cumulative_returns_data.items():
            if pair != 'PORTFOLIO':
                max_cum = max(data['cumulative_returns']) * 100 if data['cumulative_returns'] else 0
                max_cumulative_returns.append(max_cum)
                pair_names.append(f"{pair[0]}-{pair[1]}")
        
        if max_cumulative_returns:
            ax5.bar(range(len(max_cumulative_returns)), max_cumulative_returns, alpha=0.7)
            ax5.set_xticks(range(len(pair_names)))
            ax5.set_xticklabels(pair_names, rotation=45, ha='right')
            ax5.set_ylabel('Maximum Cumulative Return (%)')
            ax5.set_title('Maximum Cumulative Returns by Pair', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 6. Minimum Cumulative Returns (Drawdowns)
        ax6 = axes[1, 2]
        min_cumulative_returns = []
        
        for pair, data in self.cumulative_returns_data.items():
            if pair != 'PORTFOLIO':
                min_cum = min(data['cumulative_returns']) * 100 if data['cumulative_returns'] else 0
                min_cumulative_returns.append(min_cum)
        
        if min_cumulative_returns:
            ax6.bar(range(len(min_cumulative_returns)), min_cumulative_returns, alpha=0.7, color='salmon')
            ax6.set_xticks(range(len(pair_names)))
            ax6.set_xticklabels(pair_names, rotation=45, ha='right')
            ax6.set_ylabel('Minimum Cumulative Return (%)')
            ax6.set_title('Minimum Cumulative Returns (Worst Drawdowns)', fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 7. Portfolio vs Individual Performance
        ax7 = axes[2, 0]
        metrics = ['Total Return', 'Cumulative Return', 'Annualized Return', 'Sharpe Ratio']
        
        if self.backtest_results:
            avg_total = np.mean([r['total_return'] for r in self.backtest_results.values()])
            avg_cumulative = np.mean([r['cumulative_return'] for r in self.backtest_results.values()])
            avg_annualized = np.mean([r['annualized_return'] for r in self.backtest_results.values()])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in self.backtest_results.values()])
            
            individual_values = [avg_total, avg_cumulative, avg_annualized, avg_sharpe]
            
            if self.portfolio_performance:
                portfolio_values = [
                    self.portfolio_performance['total_return'],
                    self.portfolio_performance['cumulative_return'],
                    self.portfolio_performance['annualized_return'],
                    self.portfolio_performance['sharpe_ratio']
                ]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                ax7.bar(x - width/2, individual_values, width, label='Individual Avg', alpha=0.7)
                ax7.bar(x + width/2, portfolio_values, width, label='Portfolio', alpha=0.7)
                ax7.set_xticks(x)
                ax7.set_xticklabels(metrics, rotation=45)
                ax7.legend()
            else:
                ax7.bar(metrics, individual_values, alpha=0.7)
        
        ax7.set_title('Performance Comparison', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Cumulative Returns Volatility
        ax8 = axes[2, 1]
        cumulative_volatilities = []
        
        for pair, result in self.backtest_results.items():
            if result['cumulative_returns_series']:
                vol = np.std(result['cumulative_returns_series']) * 100
                cumulative_volatilities.append(vol)
        
        if cumulative_volatilities:
            ax8.hist(cumulative_volatilities, bins=8, alpha=0.7, color='orange', edgecolor='black')
            ax8.set_xlabel('Cumulative Returns Volatility (%)')
            ax8.set_ylabel('Number of Pairs')
            ax8.set_title('Cumulative Returns Volatility Distribution', fontweight='bold')
            ax8.grid(True, alpha=0.3)
        
        # 9. Performance Summary
        ax9 = axes[2, 2]
        categories = ['Pairs', 'Portfolio']
        avg_cumulative = np.mean([r['cumulative_return'] for r in self.backtest_results.values()])
        
        values = [avg_cumulative, self.portfolio_performance['cumulative_return']]
        colors = ['lightblue', 'lightgreen']
        
        ax9.bar(categories, values, color=colors, alpha=0.7)
        ax9.set_ylabel('Cumulative Return (%)')
        ax9.set_title('Average vs Portfolio Cumulative Return', fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax9.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('enhanced_backtesting_cumulative_returns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Cumulative returns visualizations generated and saved!")
    
    def generate_cumulative_returns_analysis_report(self):
        """Generate detailed cumulative returns analysis report"""
        report_content = f"""
CUMULATIVE RETURNS ANALYSIS REPORT
{'='*60}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Pairs Analyzed: {len(self.backtest_results)}
Portfolio Analysis: {'Included' if self.portfolio_performance else 'Not Available'}

{'='*60}

INDIVIDUAL PAIR CUMULATIVE RETURNS ANALYSIS
{'-'*60}

"""
        
        # Individual pairs analysis
        for pair, result in self.backtest_results.items():
            stock1, stock2 = pair
            cumulative_series = result['cumulative_returns_series']
            
            if cumulative_series:
                max_cumulative = max(cumulative_series) * 100
                min_cumulative = min(cumulative_series) * 100
                final_cumulative = result['cumulative_return']
                volatility = np.std(cumulative_series) * 100
                
                report_content += f"""
{stock1}-{stock2}:
  Final Cumulative Return: {final_cumulative:+.2f}%
  Maximum Cumulative Return: {max_cumulative:+.2f}%
  Minimum Cumulative Return: {min_cumulative:+.2f}%
  Cumulative Volatility: {volatility:.2f}%
  Total Trades: {result['total_trades']}
  Win Rate: {result['win_rate']:.1f}%
  Sharpe Ratio: {result['sharpe_ratio']:+.3f}
"""
        
        # Portfolio analysis
        if self.portfolio_performance:
            portfolio_series = self.portfolio_performance['cumulative_returns_series']
            
            if portfolio_series:
                max_cumulative = max(portfolio_series) * 100
                min_cumulative = min(portfolio_series) * 100
                final_cumulative = self.portfolio_performance['cumulative_return']
                volatility = np.std(portfolio_series) * 100
                
                report_content += f"""
{'='*60}

PORTFOLIO CUMULATIVE RETURNS ANALYSIS
{'-'*60}

Final Cumulative Return: {final_cumulative:+.2f}%
Maximum Cumulative Return: {max_cumulative:+.2f}%
Minimum Cumulative Return: {min_cumulative:+.2f}%
Cumulative Volatility: {volatility:.2f}%
Total Trades: {self.portfolio_performance['total_trades']}
Win Rate: {self.portfolio_performance['win_rate']:.1f}%
Sharpe Ratio: {self.portfolio_performance['sharpe_ratio']:+.3f}
Annualized Return: {self.portfolio_performance['annualized_return']:+.2f}%
"""
        
        # Summary statistics
        all_cumulative_returns = [r['cumulative_return'] for r in self.backtest_results.values()]
        all_volatilities = []
        
        for result in self.backtest_results.values():
            if result['cumulative_returns_series']:
                all_volatilities.append(np.std(result['cumulative_returns_series']) * 100)
        
        report_content += f"""
{'='*60}

SUMMARY STATISTICS
{'-'*60}

Individual Pairs:
  Average Cumulative Return: {np.mean(all_cumulative_returns):+.2f}%
  Standard Deviation: {np.std(all_cumulative_returns):.2f}%
  Best Performer: {max(all_cumulative_returns):+.2f}%
  Worst Performer: {min(all_cumulative_returns):+.2f}%
  Average Volatility: {np.mean(all_volatilities):.2f}%

Portfolio Benefits:
  Portfolio vs Average: {self.portfolio_performance['cumulative_return'] - np.mean(all_cumulative_returns):+.2f}%
  Risk Reduction: {np.mean(all_volatilities) - (np.std(self.portfolio_performance['cumulative_returns_series']) * 100):+.2f}%

{'='*60}

END OF REPORT
"""
        
        # Save report
        with open('cumulative_returns_analysis_report.txt', 'w') as f:
            f.write(report_content)
        
        print("📋 Cumulative returns analysis report exported to: cumulative_returns_analysis_report.txt")

# Main execution
if __name__ == "__main__":
    print("🔄 ENHANCED BACKTESTING WITH CUMULATIVE RETURNS")
    print("="*80)
    print("📊 Multi-Timeframe Pairs Trading Backtesting")
    print("📈 Comprehensive Cumulative Returns Analysis")
    print("📝 All output will be saved to log file")
    print("="*80)
    
    try:
        # Initialize and run enhanced backtesting with cumulative returns
        backtesting_system = EnhancedBacktestingWithCumulativeReturns()
        backtesting_system.run_enhanced_backtesting_with_cumulative_returns()
        
        print("\n🎉 ENHANCED BACKTESTING WITH CUMULATIVE RETURNS COMPLETED!")
        print("="*80)
        print("📋 Key Features:")
        print("✅ 1. Comprehensive cumulative returns tracking")
        print("✅ 2. Individual pair cumulative returns analysis")
        print("✅ 3. Portfolio cumulative returns comparison")
        print("✅ 4. Maximum/minimum cumulative returns tracking")
        print("✅ 5. Cumulative returns volatility analysis")
        print("✅ 6. Professional visualizations and reports")
        print("\n🚀 CHECK THE GENERATED FILES FOR DETAILED CUMULATIVE RETURNS ANALYSIS!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check your data files and try again.")
