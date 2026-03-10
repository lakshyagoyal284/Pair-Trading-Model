"""
DATA PROCESSING MODULES
Handles data loading, cleaning, and preparation for backtesting
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Handles loading and processing of stock data"""
    
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data_cache = {}
        self.available_stocks = []
    
    def load_all_stocks(self, subfolder='3minute'):
        """Load all stocks from specified subfolder"""
        print(f"📊 Loading stocks from {subfolder} folder...")
        
        # Get all CSV files
        csv_path = os.path.join(self.data_folder, subfolder, '*.csv')
        csv_files = glob.glob(csv_path)
        
        if not csv_files:
            print(f"❌ No CSV files found in {csv_path}")
            return {}
        
        print(f"📁 Found {len(csv_files)} CSV files")
        
        # Load all stocks
        loaded_stocks = {}
        failed_stocks = []
        
        for csv_file in csv_files:
            try:
                stock_name = os.path.basename(csv_file).replace('.csv', '')
                df = self.load_single_stock(csv_file)
                
                if df is not None and len(df) > 100:
                    loaded_stocks[stock_name] = df
                    if len(loaded_stocks) % 10 == 0:
                        print(f"📊 Loaded {len(loaded_stocks)} stocks...")
                else:
                    failed_stocks.append(stock_name)
                    
            except Exception as e:
                print(f"⚠️  Failed to load {stock_name}: {e}")
                failed_stocks.append(stock_name)
        
        print(f"✅ Successfully loaded {len(loaded_stocks)} stocks")
        if failed_stocks:
            print(f"⚠️  Failed to load {len(failed_stocks)} stocks: {failed_stocks[:5]}...")
        
        self.available_stocks = list(loaded_stocks.keys())
        self.data_cache.update(loaded_stocks)
        
        return loaded_stocks
    
    def load_single_stock(self, file_path):
        """Load a single stock CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Basic validation
            if len(df) < 50:
                return None
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure required columns exist
            required_cols = ['date', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return None
            
            # Parse date
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Basic cleaning
            df = df[df['open'] > 0]
            df = df[df['high'] > 0]
            df = df[df['low'] > 0]
            df = df[df['close'] > 0]
            
            # Add basic features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            return df
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def get_stock_data(self, stock_name):
        """Get data for a specific stock"""
        if stock_name in self.data_cache:
            return self.data_cache[stock_name]
        else:
            # Try to load from file
            file_path = os.path.join(self.data_folder, '3minute', f'{stock_name}.csv')
            if os.path.exists(file_path):
                df = self.load_single_stock(file_path)
                if df is not None:
                    self.data_cache[stock_name] = df
                return df
            return None
    
    def create_pairs(self, max_pairs=None):
        """Create all possible pairs from available stocks"""
        if not self.available_stocks:
            return []
        
        pairs = []
        n_stocks = len(self.available_stocks)
        
        for i in range(n_stocks):
            for j in range(i + 1, n_stocks):
                pairs.append((self.available_stocks[i], self.available_stocks[j]))
                
                if max_pairs and len(pairs) >= max_pairs:
                    break
            
            if max_pairs and len(pairs) >= max_pairs:
                break
        
        print(f"📊 Created {len(pairs)} pairs from {n_stocks} stocks")
        return pairs
    
    def filter_pairs_by_correlation(self, pairs, min_correlation=0.3, max_correlation=0.9, period=252):
        """Filter pairs based on correlation"""
        filtered_pairs = []
        
        for stock1, stock2 in pairs:
            data1 = self.get_stock_data(stock1)
            data2 = self.get_stock_data(stock2)
            
            if data1 is None or data2 is None:
                continue
            
            # Align data
            common_index = data1.index.intersection(data2.index)
            if len(common_index) < period:
                continue
            
            returns1 = data1.loc[common_index, 'returns'].dropna()
            returns2 = data2.loc[common_index, 'returns'].dropna()
            
            if len(returns1) < period:
                continue
            
            # Calculate correlation
            correlation = returns1.corr(returns2)
            
            if min_correlation <= abs(correlation) <= max_correlation:
                filtered_pairs.append((stock1, stock2, abs(correlation)))
        
        # Sort by correlation
        filtered_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"📊 Filtered to {len(filtered_pairs)} pairs with correlation {min_correlation}-{max_correlation}")
        return [(p[0], p[1]) for p in filtered_pairs]
    
    def get_data_summary(self):
        """Get summary of loaded data"""
        if not self.data_cache:
            return "No data loaded"
        
        summary = {
            'total_stocks': len(self.data_cache),
            'date_range': {},
            'data_quality': {}
        }
        
        all_dates = []
        quality_issues = []
        
        for stock_name, df in self.data_cache.items():
            all_dates.extend(df.index)
            
            # Check for quality issues
            missing_data_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            zero_returns_pct = (df['returns'] == 0).sum() / len(df) * 100
            
            if missing_data_pct > 5:
                quality_issues.append(f"{stock_name}: {missing_data_pct:.1f}% missing data")
            
            if zero_returns_pct > 20:
                quality_issues.append(f"{stock_name}: {zero_returns_pct:.1f}% zero returns")
        
        if all_dates:
            summary['date_range']['start'] = min(all_dates)
            summary['date_range']['end'] = max(all_dates)
            summary['date_range']['total_days'] = (max(all_dates) - min(all_dates)).days
        
        summary['data_quality']['issues'] = quality_issues
        
        return summary

class DataValidator:
    """Validates data quality and consistency"""
    
    @staticmethod
    def validate_stock_data(df, stock_name):
        """Validate a single stock's data"""
        issues = []
        
        if df is None or len(df) == 0:
            return [f"{stock_name}: No data"]
        
        # Check for missing values
        missing_data = df.isnull().sum()
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                missing_pct = missing_count / len(df) * 100
                issues.append(f"{stock_name}: {missing_pct:.1f}% missing {col}")
        
        # Check for price consistency
        if 'open' in df.columns and 'close' in df.columns:
            # High should be >= low, close should be between high and low
            if 'high' in df.columns and 'low' in df.columns:
                inconsistent_prices = (df['high'] < df['low']).sum()
                if inconsistent_prices > 0:
                    issues.append(f"{stock_name}: {inconsistent_prices} inconsistent high/low prices")
        
        # Check for extreme returns
        if 'returns' in df.columns:
            extreme_returns = (np.abs(df['returns']) > 0.5).sum()  # > 50% returns
            if extreme_returns > 0:
                issues.append(f"{stock_name}: {extreme_returns} extreme returns (>50%)")
        
        # Check for duplicate dates
        duplicate_dates = df.index.duplicated().sum()
        if duplicate_dates > 0:
            issues.append(f"{stock_name}: {duplicate_dates} duplicate dates")
        
        return issues
    
    @staticmethod
    def validate_pair_data(stock1_data, stock2_data, stock1_name, stock2_name):
        """Validate data for a pair of stocks"""
        issues = []
        
        # Check data overlap
        common_dates = stock1_data.index.intersection(stock2_data.index)
        if len(common_dates) < 100:
            issues.append(f"Insufficient overlap: {len(common_dates)} common dates")
        
        # Check for data gaps
        if len(common_dates) > 0:
            expected_dates = pd.date_range(start=common_dates.min(), end=common_dates.max(), freq='D')
            missing_dates = len(expected_dates) - len(common_dates)
            if missing_dates > expected_dates * 0.1:  # More than 10% missing
                issues.append(f"Too many missing dates: {missing_dates} missing")
        
        return issues

class DataPreprocessor:
    """Preprocesses data for backtesting"""
    
    @staticmethod
    def clean_data(df):
        """Clean and standardize data"""
        if df is None:
            return None
        
        # Remove outliers (extreme price movements)
        if 'returns' in df.columns:
            # Cap returns at +/- 20% per period
            df['returns'] = df['returns'].clip(lower=-0.2, upper=0.2)
        
        # Forward fill missing values (limited)
        df = df.fillna(method='ffill', limit=5)
        
        # Remove remaining NaN rows
        df = df.dropna()
        
        return df
    
    @staticmethod
    def align_data(stock1_data, stock2_data):
        """Align two stock datasets"""
        if stock1_data is None or stock2_data is None:
            return None, None
        
        # Find common dates
        common_dates = stock1_data.index.intersection(stock2_data.index)
        
        if len(common_dates) < 100:
            return None, None
        
        # Align data
        aligned1 = stock1_data.loc[common_dates]
        aligned2 = stock2_data.loc[common_dates]
        
        return aligned1, aligned2
    
    @staticmethod
    def resample_data(df, frequency='D'):
        """Resample data to different frequency"""
        if df is None:
            return None
        
        if frequency == 'D':
            # Daily resampling
            return df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        elif frequency == 'H':
            # Hourly resampling
            return df.resample('H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        else:
            return df

# Utility functions
def load_data_with_validation(data_folder, subfolder='3minute'):
    """Load data with comprehensive validation"""
    loader = DataLoader(data_folder)
    validator = DataValidator()
    
    # Load all stocks
    stocks_data = loader.load_all_stocks(subfolder)
    
    # Validate each stock
    all_issues = []
    for stock_name, df in stocks_data.items():
        issues = validator.validate_stock_data(df, stock_name)
        all_issues.extend(issues)
    
    # Print validation summary
    if all_issues:
        print(f"⚠️  Found {len(all_issues)} data quality issues:")
        for issue in all_issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(all_issues) > 10:
            print(f"  ... and {len(all_issues) - 10} more issues")
    else:
        print("✅ All data passed validation")
    
    return stocks_data, loader

def create_robust_pairs(stocks_data, min_correlation=0.3, max_pairs=50):
    """Create robust pairs with filtering"""
    loader = DataLoader("")
    loader.data_cache = stocks_data
    loader.available_stocks = list(stocks_data.keys())
    
    # Create all possible pairs
    all_pairs = loader.create_pairs()
    
    # Filter by correlation
    filtered_pairs = loader.filter_pairs_by_correlation(
        all_pairs, min_correlation=min_correlation, max_correlation=0.9
    )
    
    # Limit number of pairs
    if max_pairs and len(filtered_pairs) > max_pairs:
        filtered_pairs = filtered_pairs[:max_pairs]
    
    return filtered_pairs
