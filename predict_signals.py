import pandas as pd
import numpy as np
import pickle
from train_model import ModelTrainer

class SignalPredictor:
    def __init__(self, model_path='pairs_trading_model.pkl'):
        self.model_path = model_path
        self.trainer = ModelTrainer("c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute")
        self.model_loaded = False
    
    def load_model(self):
        """Load the trained model"""
        success = self.trainer.load_model(self.model_path)
        self.model_loaded = success
        return success
    
    def get_latest_signal(self):
        """Get the latest trading signal"""
        if not self.model_loaded:
            if not self.load_model():
                return None
        
        return self.trainer.predict_signals()
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.model_loaded:
            if not self.load_model():
                return None
        
        metadata = self.trainer.model_metadata
        return {
            'pair': metadata['pair'],
            'p_value': metadata['p_value'],
            'accuracy': metadata['accuracy'],
            'feature_importance': metadata['feature_importance']
        }

# Usage example
if __name__ == "__main__":
    print("🔮 PAIRS TRADING SIGNAL PREDICTOR")
    print("="*50)
    
    # Initialize predictor
    predictor = SignalPredictor()
    
    # Load model
    if predictor.load_model():
        print("✅ Model loaded successfully!")
        
        # Get model information
        model_info = predictor.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"Trading Pair: {model_info['pair']}")
        print(f"Accuracy: {model_info['accuracy']:.3f}")
        print(f"P-value: {model_info['p_value']:.6f}")
        
        # Get latest signal
        signal = predictor.get_latest_signal()
        
        if signal:
            print(f"\n🎯 CURRENT TRADING SIGNAL:")
            print(f"Signal: {signal['signal_name']}")
            print(f"Confidence: {signal['confidence']:.3f}")
            print(f"Z-Score: {signal['features']['z_score']:.3f}")
            print(f"RSI: {signal['features']['rsi']:.3f}")
            print(f"Volatility: {signal['features']['volatility']:.3f}")
        else:
            print("❌ Unable to generate signal")
    else:
        print("❌ Failed to load model")
