import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PairsTradingPipeline:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.price_data = None
        self.returns_data = None
        self.cointegrated_pairs = []
        self.ml_model = None
        self.backtest_results = None
        
    def step1_data_preprocessing(self):
        """Step 1: Data Preprocessing"""
        print("Step 1: Data Preprocessing...")
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files")
        
        # Read and process each file
        all_data = {}
        
        for i, csv_file in enumerate(csv_files[:50]):  # Limit to first 50 stocks for manageable processing
            if i % 10 == 0:
                print(f"Processing file {i+1}/50: {csv_file}")
            
            try:
                # Read CSV
                file_path = os.path.join(self.data_folder, csv_file)
                df = pd.read_csv(file_path)
                
                # Parse dates and extract date part
                df['date'] = pd.to_datetime(df['date']).dt.date
                
                # Get daily closing prices (last entry per day)
                daily_data = df.groupby('date')['close'].last().reset_index()
                daily_data.set_index('date', inplace=True)
                
                # Store with ticker name (remove .csv extension)
                ticker = csv_file.replace('.csv', '')
                all_data[ticker] = daily_data['close']
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        # Create DataFrame with all stocks
        self.price_data = pd.DataFrame(all_data)
        
        # Forward-fill missing values
        self.price_data = self.price_data.fillna(method='ffill')
        
        # Remove stocks with too many missing values
        min_data_points = len(self.price_data) * 0.8  # At least 80% data
        valid_stocks = self.price_data.columns[self.price_data.count() >= min_data_points]
        self.price_data = self.price_data[valid_stocks]
        
        print(f"Successfully loaded {len(self.price_data.columns)} stocks")
        print(f"Date range: {self.price_data.index.min()} to {self.price_data.index.max()}")
        
        # Calculate daily percentage returns
        self.returns_data = self.price_data.pct_change().dropna()
        
        print("Data preprocessing completed!")
        return self.price_data, self.returns_data
    
    def step2_pair_selection(self, n_components=10, clustering_method='dbscan'):
        """Step 2: Pair Selection using Unsupervised Learning"""
        print("\nStep 2: Pair Selection (Unsupervised Learning)...")
        
        if self.returns_data is None:
            raise ValueError("Please run step1_data_preprocessing first")
        
        print(f"Using {len(self.returns_data.columns)} stocks for analysis")
        
        # Standardize returns
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(self.returns_data.T)  # Transpose for stocks as samples
        
        # Apply PCA for dimensionality reduction
        print("Applying PCA...")
        pca = PCA(n_components=n_components)
        returns_pca = pca.fit_transform(returns_scaled)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Apply clustering
        print(f"Applying {clustering_method.upper()} clustering...")
        
        if clustering_method.lower() == 'dbscan':
            clustering = DBSCAN(eps=0.5, min_samples=2)
        else:  # KMeans
            clustering = KMeans(n_clusters=8, random_state=42)
        
        cluster_labels = clustering.fit_predict(returns_pca)
        
        # Create clusters dictionary
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.returns_data.columns[i])
        
        print(f"Found {len(clusters)} clusters:")
        for label, stocks in clusters.items():
            print(f"  Cluster {label}: {len(stocks)} stocks")
        
        # Test pairs within each cluster for cointegration
        print("\nTesting pairs for cointegration...")
        cointegration_results = []
        
        for cluster_id, stocks_in_cluster in clusters.items():
            if len(stocks_in_cluster) < 2:
                continue
                
            print(f"Testing {len(list(combinations(stocks_in_cluster, 2)))} pairs in cluster {cluster_id}")
            
            # Test all combinations within the cluster
            for stock1, stock2 in combinations(stocks_in_cluster, 2):
                try:
                    # Get price series for the pair
                    pair_data = self.price_data[[stock1, stock2]].dropna()
                    
                    if len(pair_data) < 100:  # Need sufficient data points
                        continue
                    
                    # Test for cointegration using Engle-Granger method
                    spread = pair_data[stock1] - pair_data[stock2]
                    
                    # Perform ADF test on spread
                    adf_result = adfuller(spread, autolag='AIC')
                    p_value = adf_result[1]
                    test_statistic = adf_result[0]
                    
                    # Store results
                    cointegration_results.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'cluster': cluster_id,
                        'p_value': p_value,
                        'test_statistic': test_statistic,
                        'is_cointegrated': p_value < 0.05
                    })
                    
                except Exception as e:
                    continue
        
        # Convert to DataFrame and sort by p-value
        cointegration_df = pd.DataFrame(cointegration_results)
        
        if len(cointegration_df) > 0:
            cointegration_df = cointegration_df.sort_values('p_value')
            
            # Filter for cointegrated pairs (p < 0.05)
            cointegrated_pairs = cointegration_df[cointegration_df['is_cointegrated']]
            
            print(f"\nFound {len(cointegrated_pairs)} cointegrated pairs (p < 0.05)")
            
            # Get top 5 pairs
            top_5_pairs = cointegrated_pairs.head(5)
            
            print("\nTop 5 Most Cointegrated Pairs:")
            print("=" * 80)
            for i, (_, pair) in enumerate(top_5_pairs.iterrows(), 1):
                print(f"{i}. {pair['stock1']} - {pair['stock2']}")
                print(f"   P-value: {pair['p_value']:.6f}")
                print(f"   Test Statistic: {pair['test_statistic']:.6f}")
                print(f"   Cluster: {pair['cluster']}")
                print()
            
            self.cointegrated_pairs = top_5_pairs
            return top_5_pairs
        else:
            print("No cointegrated pairs found!")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def step3_signal_generation(self, top_pair_index=0):
        """Step 3: Signal Generation using Supervised Learning"""
        print("\nStep 3: Signal Generation (Supervised Learning)...")
        
        if len(self.cointegrated_pairs) == 0:
            raise ValueError("Please run step2_pair_selection first")
        
        # Get the best cointegrated pair
        best_pair = self.cointegrated_pairs.iloc[top_pair_index]
        stock1, stock2 = best_pair['stock1'], best_pair['stock2']
        
        print(f"Using pair: {stock1} - {stock2}")
        print(f"P-value: {best_pair['p_value']:.6f}")
        
        # Get price data for the pair
        pair_prices = self.price_data[[stock1, stock2]].copy()
        
        # Calculate spread
        pair_prices['spread'] = pair_prices[stock1] - pair_prices[stock2]
        
        # Calculate features
        # 1. Rolling Z-score of spread (20-day window)
        pair_prices['spread_mean'] = pair_prices['spread'].rolling(window=20).mean()
        pair_prices['spread_std'] = pair_prices['spread'].rolling(window=20).std()
        pair_prices['z_score'] = (pair_prices['spread'] - pair_prices['spread_mean']) / pair_prices['spread_std']
        
        # 2. RSI of spread
        pair_prices['rsi'] = self.calculate_rsi(pair_prices['spread'])
        
        # 3. 20-day rolling volatility of spread
        pair_prices['volatility'] = pair_prices['spread'].rolling(window=20).std()
        
        # Create target variable: Will spread revert next day?
        # Target: 1 for Go Long (spread will decrease), -1 for Go Short (spread will increase), 0 for Hold
        pair_prices['next_day_spread'] = pair_prices['spread'].shift(-1)
        pair_prices['spread_change'] = pair_prices['next_day_spread'] - pair_prices['spread']
        
        # Define target based on Z-score threshold and expected reversion
        z_threshold = 2.0
        conditions = [
            (pair_prices['z_score'] > z_threshold) & (pair_prices['spread_change'] < 0),  # Overbought, expect decrease
            (pair_prices['z_score'] < -z_threshold) & (pair_prices['spread_change'] > 0),  # Oversold, expect increase
        ]
        choices = [1, -1]  # 1 for Long, -1 for Short
        pair_prices['target'] = np.select(conditions, choices, default=0)
        
        # Remove NaN values
        feature_data = pair_prices[['z_score', 'rsi', 'volatility', 'target']].dropna()
        
        print(f"Generated {len(feature_data)} training samples")
        print(f"Target distribution:")
        print(feature_data['target'].value_counts())
        
        # Split data (80% train, 20% test)
        split_idx = int(len(feature_data) * 0.8)
        train_data = feature_data.iloc[:split_idx]
        test_data = feature_data.iloc[split_idx:]
        
        X_train = train_data[['z_score', 'rsi', 'volatility']]
        y_train = train_data['target']
        X_test = test_data[['z_score', 'rsi', 'volatility']]
        y_test = test_data['target']
        
        # Train Random Forest classifier
        print("Training Random Forest classifier...")
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.ml_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.ml_model.predict(X_test)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return {
            'model': self.ml_model,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'test_data': test_data,
            'predictions': y_pred
        }
    
    def step4_backtesting(self, signal_results, transaction_cost=0.001):
        """Step 4: Backtesting Engine"""
        print("\nStep 4: Backtesting Engine...")
        
        if self.ml_model is None:
            raise ValueError("Please run step3_signal_generation first")
        
        # Get the best pair from previous step
        best_pair = self.cointegrated_pairs.iloc[0]
        stock1, stock2 = best_pair['stock1'], best_pair['stock2']
        
        print(f"Backtesting pair: {stock1} - {stock2}")
        print(f"Transaction cost: {transaction_cost*100}% per trade")
        
        # Get test data and predictions
        test_data = signal_results['test_data']
        predictions = signal_results['predictions']
        
        # Get price data for test period
        test_dates = test_data.index
        price_test_data = self.price_data.loc[test_dates, [stock1, stock2]]
        
        # Initialize backtracking variables
        initial_capital = 100000
        capital = initial_capital
        position = 0  # 0: no position, 1: long spread, -1: short spread
        positions = []
        returns = []
        equity_curve = [initial_capital]
        
        # Track trades
        trades = []
        
        for i in range(len(predictions)):
            current_signal = predictions[i]
            current_price1 = price_test_data[stock1].iloc[i]
            current_price2 = price_test_data[stock2].iloc[i]
            current_spread = current_price1 - current_price2
            
            # Calculate position value
            if position == 1:  # Long spread
                position_value = current_spread
            elif position == -1:  # Short spread
                position_value = -current_spread
            else:
                position_value = 0
            
            # Execute trades based on signals
            if current_signal == 1 and position != 1:  # Go Long
                if position == -1:  # Close short position first
                    pnl = -current_spread + trades[-1]['spread'] if trades else 0
                    capital += pnl - transaction_cost * abs(pnl)
                
                position = 1
                trades.append({
                    'date': test_dates[i],
                    'action': 'LONG',
                    'spread': current_spread,
                    'price1': current_price1,
                    'price2': current_price2
                })
                capital -= transaction_cost * abs(current_spread)
                
            elif current_signal == -1 and position != -1:  # Go Short
                if position == 1:  # Close long position first
                    pnl = current_spread - trades[-1]['spread'] if trades else 0
                    capital += pnl - transaction_cost * abs(pnl)
                
                position = -1
                trades.append({
                    'date': test_dates[i],
                    'action': 'SHORT',
                    'spread': current_spread,
                    'price1': current_price1,
                    'price2': current_price2
                })
                capital -= transaction_cost * abs(current_spread)
            
            elif current_signal == 0 and position != 0:  # Close position
                if position == 1:
                    pnl = current_spread - trades[-1]['spread'] if trades else 0
                else:
                    pnl = -current_spread + trades[-1]['spread'] if trades else 0
                
                capital += pnl - transaction_cost * abs(pnl)
                position = 0
                trades.append({
                    'date': test_dates[i],
                    'action': 'CLOSE',
                    'spread': current_spread,
                    'pnl': pnl
                })
            
            positions.append(position)
            
            # Calculate daily return
            if i > 0:
                daily_return = (capital - equity_curve[-1]) / equity_curve[-1]
                returns.append(daily_return)
            
            equity_curve.append(capital)
        
        # Calculate performance metrics
        total_return = (capital - initial_capital) / initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        
        returns_array = np.array(returns)
        sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        
        # Calculate maximum drawdown
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)

        # Calculate average holding period
        holding_periods = []
        open_trade = None

        for trade in trades:
            if trade['action'] == 'LONG' or trade['action'] == 'SHORT':
                open_trade = trade
            elif trade['action'] == 'CLOSE' and open_trade:
                duration = (trade['date'] - open_trade['date']).days
                if duration > 0: # Only consider trades that were held for at least one day
                    holding_periods.append(duration)
                open_trade = None
        
        avg_holding_period = np.mean(holding_periods) if len(holding_periods) > 0 else 0

        # Store results
        self.backtest_results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'equity_curve': equity_curve,
            'returns': returns,
            'trades': trades,
            'avg_holding_period': avg_holding_period
        }
        
        # Print results in formatted table
        print("\n" + "="*80)
        print("BACKTESTING RESULTS")
        print("="*80)
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital: ${capital:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Total Trades: {len(trades)}")
        print(f"Average Holding Period: {avg_holding_period:.2f} days")
        print("="*60)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, equity_curve[1:], label='Equity Curve', linewidth=2)
        plt.title(f'Pairs Trading Backtest: {stock1} - {stock2}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"backtest_{stock1}_{stock2}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Equity curve saved as: {plot_filename}")
        
        return self.backtest_results
    
    def display_formatted_results(self, signal_results):
        """Display results in formatted table format"""
        print("\n" + "="*80)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*80)
        
        # Extract metrics from signal results and backtest results
        accuracy = signal_results['accuracy']
        
        # Get precision from classification report (weighted average for trading signals)
        from sklearn.metrics import precision_score
        X_test = signal_results['test_data'][['z_score', 'rsi', 'volatility']]
        y_test = signal_results['test_data']['target']
        y_pred = signal_results['predictions']
        
        # Calculate precision for trading signals (classes 1 and -1)
        trading_mask = (y_test != 0)  # Only consider trading signals
        if trading_mask.sum() > 0:
            precision_trading = precision_score(y_test[trading_mask], y_pred[trading_mask], average='weighted')
        else:
            precision_trading = 0
        
        # Get backtest metrics
        cagr = self.backtest_results['annualized_return'] if self.backtest_results else 0
        avg_holding_period = self.backtest_results['avg_holding_period'] if self.backtest_results else 0
        
        # Create formatted tables
        print("\nRETURN METRICS")
        print("-" * 40)
        print(f"{'Metric':<20} {'Value':<20}")
        print(f"{'-'*20} {'-'*20}")
        print(f"{'Total Return':<20} {self.backtest_results['total_return']:.2%}{'':<10}")
        print(f"{'CAGR':<20} {cagr:.2%}{'':<10}")
        print(f"{'Sharpe Ratio':<20} {self.backtest_results['sharpe_ratio']:.3f}{'':<10}")
        
        print("\nRISK METRICS")
        print("-" * 40)
        print(f"{'Metric':<20} {'Value':<20}")
        print(f"{'-'*20} {'-'*20}")
        print(f"{'Max Drawdown':<20} {self.backtest_results['max_drawdown']:.2%}{'':<10}")
        print(f"{'Volatility (Daily)':<20} {np.std(self.backtest_results['returns']):.4f}{'':<10}")
        
        print("\nTIMING METRICS")
        print("-" * 40)
        print(f"{'Metric':<20} {'Value':<20}")
        print(f"{'-'*20} {'-'*20}")
        print(f"{'Accuracy':<20} {accuracy:.3f}{'':<10}")
        print(f"{'Precision':<20} {precision_trading:.3f}{'':<10}")
        print(f"{'Avg Holding Period':<20} {avg_holding_period:.1f} days{'':<5}")
        print(f"{'Total Trades':<20} {self.backtest_results['total_trades']}{'':<10}")
        
        print("\n" + "="*80)

# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    pipeline = PairsTradingPipeline(data_folder)
    
    # Execute Step 1
    price_data, returns_data = pipeline.step1_data_preprocessing()
    
    # Execute Step 2
    top_pairs = pipeline.step2_pair_selection(n_components=10, clustering_method='dbscan')
    
    # Execute Step 3
    signal_results = pipeline.step3_signal_generation(top_pair_index=0)
    
    # Execute Step 4
    backtest_results = pipeline.step4_backtesting(signal_results, transaction_cost=0.001)
    
    # Display formatted results
    pipeline.display_formatted_results(signal_results)
    
    print("\n" + "="*80)
    print("COMPLETE PAIRS TRADING PIPELINE - ALL STEPS COMPLETED")
    print("="*80)
