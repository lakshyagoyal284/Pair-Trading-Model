import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from pairs_trading_pipeline import PairsTradingPipeline
from train_model import ModelTrainer

class ModelValidator:
    def __init__(self, model_path='pairs_trading_model.pkl'):
        self.model_path = model_path
        self.trainer = ModelTrainer("c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute")
        self.min_accuracy_threshold = 0.80  # 80% minimum accuracy requirement
        
    def load_and_validate_model(self):
        """Load model and validate accuracy"""
        print("🔍 MODEL VALIDATION")
        print("="*50)
        
        # Load the trained model
        if not self.trainer.load_model(self.model_path):
            print("❌ Failed to load model")
            return False
        
        # Get model metadata
        metadata = self.trainer.model_metadata
        current_accuracy = metadata['accuracy']
        
        print(f"📊 Model Information:")
        print(f"Trading Pair: {metadata['pair']}")
        print(f"Current Accuracy: {current_accuracy:.3f} ({current_accuracy*100:.1f}%)")
        print(f"Required Accuracy: {self.min_accuracy_threshold:.3f} ({self.min_accuracy_threshold*100:.1f}%)")
        
        # Check if accuracy meets requirement
        if current_accuracy >= self.min_accuracy_threshold:
            print(f"✅ ACCURACY REQUIREMENT MET!")
            print(f"Model accuracy ({current_accuracy:.1f}%) exceeds minimum ({self.min_accuracy_threshold*100:.1f}%)")
            return True
        else:
            print(f"❌ ACCURACY REQUIREMENT NOT MET!")
            print(f"Model accuracy ({current_accuracy:.1f}%) is below minimum ({self.min_accuracy_threshold*100:.1f}%)")
            return False
    
    def detailed_validation(self):
        """Perform detailed validation with cross-validation"""
        print("\n🔬 DETAILED VALIDATION")
        print("="*50)
        
        if not self.trainer.load_model(self.model_path):
            return False
        
        # Re-train with cross-validation to get robust metrics
        print("Performing detailed validation...")
        
        # Initialize pipeline for validation
        pipeline = PairsTradingPipeline("c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute")
        
        # Step 1: Data Preprocessing
        price_data, returns_data = pipeline.step1_data_preprocessing()
        
        # Step 2: Pair Selection (use same pair as trained model)
        top_pairs = pipeline.step2_pair_selection(n_components=10, clustering_method='dbscan')
        
        if len(top_pairs) == 0:
            print("❌ No cointegrated pairs found")
            return False
        
        # Step 3: Signal Generation with detailed metrics
        best_pair = top_pairs.iloc[0]
        stock1, stock2 = best_pair['stock1'], best_pair['stock2']
        
        print(f"Validating pair: {stock1} - {stock2}")
        
        # Get price data for the pair
        pair_prices = price_data[[stock1, stock2]].copy()
        
        # Calculate spread and features
        pair_prices['spread'] = pair_prices[stock1] - pair_prices[stock2]
        pair_prices['spread_mean'] = pair_prices['spread'].rolling(window=20).mean()
        pair_prices['spread_std'] = pair_prices['spread'].rolling(window=20).std()
        pair_prices['z_score'] = (pair_prices['spread'] - pair_prices['spread_mean']) / pair_prices['spread_std']
        pair_prices['rsi'] = self.calculate_rsi(pair_prices['spread'])
        pair_prices['volatility'] = pair_prices['spread'].rolling(window=20).std()
        
        # Create target variable
        pair_prices['next_day_spread'] = pair_prices['spread'].shift(-1)
        pair_prices['spread_change'] = pair_prices['next_day_spread'] - pair_prices['spread']
        
        z_threshold = 2.0
        conditions = [
            (pair_prices['z_score'] > z_threshold) & (pair_prices['spread_change'] < 0),
            (pair_prices['z_score'] < -z_threshold) & (pair_prices['spread_change'] > 0),
        ]
        choices = [1, -1]
        pair_prices['target'] = np.select(conditions, choices, default=0)
        
        # Remove NaN values
        feature_data = pair_prices[['z_score', 'rsi', 'volatility', 'target']].dropna()
        
        # Split data for validation
        split_idx = int(len(feature_data) * 0.8)
        train_data = feature_data.iloc[:split_idx]
        test_data = feature_data.iloc[split_idx:]
        
        X_train = train_data[['z_score', 'rsi', 'volatility']]
        y_train = train_data['target']
        X_test = test_data[['z_score', 'rsi', 'volatility']]
        y_test = test_data['target']
        
        # Train model (fresh training for validation)
        validation_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        validation_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = validation_model.predict(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate trading-specific metrics (only for trading signals)
        trading_mask = (y_test != 0)  # Only consider trading signals
        if trading_mask.sum() > 0:
            trading_accuracy = accuracy_score(y_test[trading_mask], y_pred[trading_mask])
            trading_precision = precision_score(y_test[trading_mask], y_pred[trading_mask], average='weighted')
        else:
            trading_accuracy = 0
            trading_precision = 0
        
        print(f"\n📊 VALIDATION METRICS:")
        print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Trading Signal Accuracy: {trading_accuracy:.3f} ({trading_accuracy*100:.1f}%)")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        print(f"\n🎯 ACCURACY STATUS:")
        if accuracy >= self.min_accuracy_threshold:
            print(f"✅ PASSED: {accuracy*100:.1f}% >= {self.min_accuracy_threshold*100:.1f}%")
            return True
        else:
            print(f"❌ FAILED: {accuracy*100:.1f}% < {self.min_accuracy_threshold*100:.1f}%")
            return False
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# Main validation execution
if __name__ == "__main__":
    validator = ModelValidator()
    
    # Quick validation
    quick_validation_passed = validator.load_and_validate_model()
    
    # Detailed validation
    detailed_validation_passed = validator.detailed_validation()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Quick Validation: {'✅ PASSED' if quick_validation_passed else '❌ FAILED'}")
    print(f"Detailed Validation: {'✅ PASSED' if detailed_validation_passed else '❌ FAILED'}")
    
    if quick_validation_passed and detailed_validation_passed:
        print("\n🎉 MODEL VALIDATION SUCCESSFUL!")
        print("Model accuracy meets the 80% requirement.")
    else:
        print("\n⚠️  MODEL VALIDATION FAILED!")
        print("Model accuracy does not meet the 80% requirement.")
        print("Consider retraining with different parameters or more data.")
