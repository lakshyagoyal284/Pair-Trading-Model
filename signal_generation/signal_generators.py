"""
SIGNAL GENERATION MODULES
Contains all signal generation strategies and algorithms
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

class MeanReversionSignals:
    """Mean reversion signal generation for pairs trading"""
    
    def __init__(self, entry_z=2.0, exit_z=0.5, window=20):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.window = window
    
    def generate_signals(self, spread):
        """Generate mean reversion signals from spread"""
        # Calculate rolling statistics
        spread_mean = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        
        # Calculate z-scores
        z_scores = (spread - spread_mean) / spread_std
        
        # Generate signals
        signals = pd.Series(0, index=spread.index)
        
        # Long spread when z-score is too low (oversold)
        signals[z_scores < -self.entry_z] = 1
        
        # Short spread when z-score is too high (overbought)
        signals[z_scores > self.entry_z] = -1
        
        # Exit when spread returns to normal
        signals[z_scores.abs() < self.exit_z] = 0
        
        return signals, z_scores
    
    def get_signal_strength(self, z_scores):
        """Get signal strength based on z-score magnitude"""
        return np.abs(z_scores)

class MomentumSignals:
    """Momentum-based signal generation"""
    
    def __init__(self, momentum_period=10, threshold=0.02):
        self.momentum_period = momentum_period
        self.threshold = threshold
    
    def generate_signals(self, spread):
        """Generate momentum signals from spread"""
        # Calculate momentum
        momentum = spread.pct_change(self.momentum_period)
        
        # Generate signals
        signals = pd.Series(0, index=spread.index)
        
        # Long spread when momentum is positive
        signals[momentum > self.threshold] = 1
        
        # Short spread when momentum is negative
        signals[momentum < -self.threshold] = -1
        
        return signals, momentum

class MLSignals:
    """Machine learning based signal generation"""
    
    def __init__(self, model_type='rf', lookback_period=20):
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.model = None
        self.scaler = StandardScaler()
        
        # Initialize model
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            self.model = LinearRegression()
    
    def prepare_features(self, spread):
        """Prepare features for ML model"""
        features = pd.DataFrame(index=spread.index)
        
        # Technical indicators
        features['z_score'] = (spread - spread.rolling(self.lookback_period).mean()) / spread.rolling(self.lookback_period).std()
        features['momentum'] = spread.pct_change(5)
        features['volatility'] = spread.rolling(self.lookback_period).std()
        features['trend'] = spread.rolling(self.lookback_period).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Lag features
        for lag in range(1, 6):
            features[f'lag_{lag}'] = spread.shift(lag)
        
        return features.dropna()
    
    def train(self, spread, returns):
        """Train ML model"""
        features = self.prepare_features(spread)
        
        # Align features with returns
        features = features.reindex(returns.index).dropna()
        returns = returns.reindex(features.index)
        
        if len(features) < 100:
            return False
        
        # Prepare target variable
        if self.model_type == 'rf':
            # Classification: 1 for positive return, 0 for negative
            target = (returns > 0).astype(int)
            self.model.fit(features, target)
        else:
            # Regression: predict returns
            self.model.fit(features, returns)
        
        return True
    
    def generate_signals(self, spread):
        """Generate ML-based signals"""
        features = self.prepare_features(spread)
        
        if self.model is None or len(features) == 0:
            return pd.Series(0, index=spread.index), None
        
        # Predict
        if self.model_type == 'rf':
            predictions = self.model.predict_proba(features)[:, 1]  # Probability of positive return
        else:
            predictions = self.model.predict(features)
        
        # Generate signals
        signals = pd.Series(0, index=features.index)
        
        if self.model_type == 'rf':
            signals[predictions > 0.6] = 1  # Long when probability > 60%
            signals[predictions < 0.4] = -1  # Short when probability < 40%
        else:
            signals[predictions > 0.01] = 1  # Long when predicted return > 1%
            signals[predictions < -0.01] = -1  # Short when predicted return < -1%
        
        return signals, predictions

class HybridSignals:
    """Hybrid signal generation combining multiple strategies"""
    
    def __init__(self, mean_reversion_weight=0.4, momentum_weight=0.3, ml_weight=0.3):
        self.mean_reversion_weight = mean_reversion_weight
        self.momentum_weight = momentum_weight
        self.ml_weight = ml_weight
        
        # Initialize signal generators
        self.mean_reversion = MeanReversionSignals()
        self.momentum = MomentumSignals()
        self.ml = MLSignals()
    
    def generate_signals(self, spread):
        """Generate hybrid signals"""
        # Get signals from all strategies
        mr_signals, mr_z = self.mean_reversion.generate_signals(spread)
        mom_signals, mom_mom = self.momentum.generate_signals(spread)
        ml_signals, ml_pred = self.ml.generate_signals(spread)
        
        # Combine signals with weights
        combined_signals = pd.Series(0, index=spread.index)
        
        for idx in spread.index:
            signal_value = 0
            
            if idx in mr_signals.index:
                signal_value += mr_signals[idx] * self.mean_reversion_weight
            
            if idx in mom_signals.index:
                signal_value += mom_signals[idx] * self.momentum_weight
            
            if idx in ml_signals.index:
                signal_value += ml_signals[idx] * self.ml_weight
            
            # Convert to discrete signal
            if signal_value > 0.5:
                combined_signals[idx] = 1
            elif signal_value < -0.5:
                combined_signals[idx] = -1
            else:
                combined_signals[idx] = 0
        
        return combined_signals, {
            'mean_reversion': mr_signals,
            'momentum': mom_signals,
            'ml': ml_signals,
            'z_scores': mr_z,
            'momentum_values': mom_mom,
            'ml_predictions': ml_pred
        }

class AdaptiveSignals:
    """Adaptive signal generation that adjusts parameters based on market conditions"""
    
    def __init__(self, base_entry_z=2.0, base_exit_z=0.5, adaptation_period=50):
        self.base_entry_z = base_entry_z
        self.base_exit_z = base_exit_z
        self.adaptation_period = adaptation_period
        self.current_entry_z = base_entry_z
        self.current_exit_z = base_exit_z
    
    def adapt_parameters(self, spread, recent_performance):
        """Adapt parameters based on recent performance"""
        if len(recent_performance) < 10:
            return
        
        # Calculate recent win rate
        recent_win_rate = np.mean([p > 0 for p in recent_performance[-20:]])
        
        # Adjust thresholds based on performance
        if recent_win_rate < 0.4:  # Poor performance, make signals more selective
            self.current_entry_z = self.base_entry_z * 1.2
            self.current_exit_z = self.base_exit_z * 0.8
        elif recent_win_rate > 0.6:  # Good performance, make signals less selective
            self.current_entry_z = self.base_entry_z * 0.8
            self.current_exit_z = self.base_exit_z * 1.2
        else:  # Normal performance, use base parameters
            self.current_entry_z = self.base_entry_z
            self.current_exit_z = self.base_exit_z
    
    def generate_signals(self, spread, recent_performance=None):
        """Generate adaptive signals"""
        # Adapt parameters if performance data is provided
        if recent_performance is not None:
            self.adapt_parameters(spread, recent_performance)
        
        # Generate signals with current parameters
        spread_mean = spread.rolling(self.adaptation_period).mean()
        spread_std = spread.rolling(self.adaptation_period).std()
        z_scores = (spread - spread_mean) / spread_std
        
        signals = pd.Series(0, index=spread.index)
        signals[z_scores < -self.current_entry_z] = 1
        signals[z_scores > self.current_entry_z] = -1
        signals[z_scores.abs() < self.current_exit_z] = 0
        
        return signals, z_scores

# Utility functions
def calculate_hedge_ratio(stock1_prices, stock2_prices, method='ols'):
    """Calculate hedge ratio between two stocks"""
    if method == 'ols':
        # Ordinary Least Squares
        X = stock2_prices.values.reshape(-1, 1)
        y = stock1_prices.values
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]
    
    elif method == 'correlation':
        # Correlation-based
        correlation = np.corrcoef(stock1_prices, stock2_prices)[0, 1]
        std1 = stock1_prices.std()
        std2 = stock2_prices.std()
        return correlation * (std1 / std2)
    
    elif method == 'fixed':
        # Fixed ratio
        return 1.0
    
    else:
        raise ValueError(f"Unknown hedge ratio method: {method}")

def calculate_spread(stock1_prices, stock2_prices, hedge_ratio):
    """Calculate spread between two stocks"""
    return stock1_prices - hedge_ratio * stock2_prices

def validate_signals(signals, min_trades=10):
    """Validate generated signals"""
    if len(signals) == 0:
        return False, "No signals generated"
    
    # Count trades
    trades = (signals.diff() != 0).sum()
    
    if trades < min_trades:
        return False, f"Too few trades: {trades} < {min_trades}"
    
    # Check signal distribution
    signal_counts = signals.value_counts()
    if len(signal_counts) < 2:
        return False, "Signals only in one direction"
    
    return True, f"Valid signals: {trades} trades, {len(signal_counts)} directions"
