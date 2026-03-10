import pandas as pd
import numpy as np
import os
import pickle
from pairs_trading_pipeline import PairsTradingPipeline

class ModelTrainer:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.pipeline = None
        self.trained_model = None
        self.model_metadata = {}
    
    def train_complete_pipeline(self, save_model=True):
        """Train the complete pairs trading pipeline"""
        print("="*80)
        print("TRAINING PAIRS TRADING MODEL")
        print("="*80)
        
        # Initialize pipeline
        self.pipeline = PairsTradingPipeline(self.data_folder)
        
        # Step 1: Data Preprocessing
        print("\n📊 STEP 1: DATA PREPROCESSING")
        price_data, returns_data = self.pipeline.step1_data_preprocessing()
        
        # Step 2: Pair Selection
        print("\n🔍 STEP 2: PAIR SELECTION")
        top_pairs = self.pipeline.step2_pair_selection(n_components=10, clustering_method='dbscan')
        
        if len(top_pairs) == 0:
            print("❌ No cointegrated pairs found. Cannot proceed with training.")
            return None
        
        # Step 3: Signal Generation (Model Training)
        print("\n🤖 STEP 3: MODEL TRAINING")
        signal_results = self.pipeline.step3_signal_generation(top_pair_index=0)
        
        # Step 4: Backtesting (Optional for validation)
        print("\n📈 STEP 4: MODEL VALIDATION")
        backtest_results = self.pipeline.step4_backtesting(signal_results, transaction_cost=0.001)
        
        # Store trained model and metadata
        self.trained_model = signal_results['model']
        self.model_metadata = {
            'pair': f"{top_pairs.iloc[0]['stock1']}-{top_pairs.iloc[0]['stock2']}",
            'p_value': top_pairs.iloc[0]['p_value'],
            'accuracy': signal_results['accuracy'],
            'feature_importance': signal_results['feature_importance'],
            'training_samples': len(signal_results['test_data']) + len(signal_results['test_data']) * 4,  # Approximate
            'backtest_metrics': backtest_results
        }
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        # Display training summary
        self.display_training_summary()
        
        return {
            'pipeline': self.pipeline,
            'model': self.trained_model,
            'metadata': self.model_metadata
        }
    
    def save_model(self, model_path='pairs_trading_model.pkl'):
        """Save the trained model and pipeline"""
        try:
            model_data = {
                'model': self.trained_model,
                'pipeline': self.pipeline,
                'metadata': self.model_metadata
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved successfully as: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_path='pairs_trading_model.pkl'):
        """Load a previously trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.trained_model = model_data['model']
            self.pipeline = model_data['pipeline']
            self.model_metadata = model_data['metadata']
            
            print(f"✅ Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def display_training_summary(self):
        """Display training summary"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        print(f"🎯 Trading Pair: {self.model_metadata['pair']}")
        print(f"📊 P-value: {self.model_metadata['p_value']:.6f}")
        print(f"🤖 Model Accuracy: {self.model_metadata['accuracy']:.3f}")
        print(f"📚 Training Samples: {self.model_metadata['training_samples']}")
        
        print("\n🔥 Feature Importance:")
        for _, row in self.model_metadata['feature_importance'].iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        if 'backtest_metrics' in self.model_metadata:
            metrics = self.model_metadata['backtest_metrics']
            print(f"\n📈 Backtest Performance:")
            print(f"   Total Return: {metrics['total_return']:.2%}")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"   Total Trades: {metrics['total_trades']}")
        
        print("="*80)
    
    def predict_signals(self, new_data=None):
        """Make predictions using the trained model"""
        if self.trained_model is None:
            print("❌ No trained model available. Please train the model first.")
            return None
        
        if new_data is None:
            # Use the most recent data from pipeline
            if self.pipeline and hasattr(self.pipeline, 'price_data'):
                # Get the most recent data for the trained pair
                pair_name = self.model_metadata['pair']
                stock1, stock2 = pair_name.split('-')
                
                recent_data = self.pipeline.price_data[[stock1, stock2]].tail(50)
                
                # Calculate features (same as in training)
                recent_data['spread'] = recent_data[stock1] - recent_data[stock2]
                recent_data['spread_mean'] = recent_data['spread'].rolling(window=20).mean()
                recent_data['spread_std'] = recent_data['spread'].rolling(window=20).std()
                recent_data['z_score'] = (recent_data['spread'] - recent_data['spread_mean']) / recent_data['spread_std']
                recent_data['rsi'] = self.calculate_rsi(recent_data['spread'])
                recent_data['volatility'] = recent_data['spread'].rolling(window=20).std()
                
                # Get latest features
                latest_features = recent_data[['z_score', 'rsi', 'volatility']].iloc[-1:].values
                
                # Make prediction
                prediction = self.trained_model.predict(latest_features)[0]
                prediction_proba = self.trained_model.predict_proba(latest_features)[0]
                
                signal_map = {1: "LONG", -1: "SHORT", 0: "HOLD"}
                
                print(f"\n🎯 LATEST TRADING SIGNAL")
                print(f"Pair: {pair_name}")
                print(f"Signal: {signal_map[prediction]}")
                print(f"Confidence: {max(prediction_proba):.3f}")
                print(f"Z-Score: {latest_features[0][0]:.3f}")
                print(f"RSI: {latest_features[0][1]:.3f}")
                print(f"Volatility: {latest_features[0][2]:.3f}")
                
                return {
                    'signal': prediction,
                    'signal_name': signal_map[prediction],
                    'confidence': max(prediction_proba),
                    'features': {
                        'z_score': latest_features[0][0],
                        'rsi': latest_features[0][1],
                        'volatility': latest_features[0][2]
                    }
                }
            else:
                print("❌ No recent data available for prediction.")
                return None
        else:
            print("❌ Custom data prediction not implemented yet.")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# Main training execution
if __name__ == "__main__":
    # Initialize trainer
    data_folder = "c:/Users/laksh/Desktop/PAIR BASED TRADE 2022 DATA/3minute"
    trainer = ModelTrainer(data_folder)
    
    # Train the model
    training_results = trainer.train_complete_pipeline(save_model=True)
    
    if training_results:
        # Make a prediction with the trained model
        trainer.predict_signals()
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("Model saved as 'pairs_trading_model.pkl'")
        print("You can now load and use this model for trading signals.")
    else:
        print("\n❌ TRAINING FAILED!")
        print("Please check the data and try again.")
