# 📁 PAIRS TRADING PROJECT - FOLDER STRUCTURE

## **🗂️ ORGANIZED FOLDER STRUCTURE**

### **📊 DATA PROCESSING**
```
data_processing/
├── data_loader.py          # Data loading, validation, and preprocessing
├── signal_generators.py     # Signal generation strategies
└── README.md             # Data processing documentation
```

**Purpose**: Handle all data-related operations including loading, cleaning, validation, and preparation for backtesting.

**Key Features**:
- Load stocks from CSV files
- Data validation and quality checks
- Create and filter pairs
- Resampling and alignment

---

### **🎯 SIGNAL GENERATION**
```
signal_generation/
├── signal_generators.py     # All signal generation strategies
├── mean_reversion.py       # Mean reversion signals
├── momentum_signals.py     # Momentum-based signals
├── ml_signals.py          # Machine learning signals
└── README.md             # Signal generation documentation
```

**Purpose**: Generate trading signals using various strategies.

**Signal Types**:
- Mean Reversion (Z-score based)
- Momentum (Trend following)
- Machine Learning (Random Forest, Linear Regression)
- Hybrid (Combination of strategies)
- Adaptive (Self-adjusting parameters)

---

### **🧪 BACKTESTING SYSTEMS**
```
backtesting_systems/
├── optimized_pairs_trading_system.py    # Main optimized system
├── enhanced_backtesting_system.py       # Enhanced version
├── professional_backtesting_final.py    # Professional version
├── balanced_unbiased_strategy.py       # Balanced strategy
├── optimized_unbiased_strategy.py      # Optimized unbiased
├── redesigned_unbiased_strategy.py     # Redesigned version
├── comprehensive_time_backtest.py      # Comprehensive testing
├── time_optimized_strategy.py          # Time-optimized version
├── pairs_trading_pipeline.py          # Complete pipeline
├── train_model.py                     # Model training
├── predict_signals.py                 # Signal prediction
├── model_validation.py                # Model validation
├── pairs_trading_model.pkl           # Trained model
├── flowchart_explanation.txt          # Flowchart documentation
└── README.md                        # Backtesting documentation
```

**Purpose**: Complete backtesting systems with different strategies and approaches.

**Key Features**:
- Multiple trading strategies
- Walk-forward validation
- Portfolio-level backtesting
- Risk management
- Performance analysis

---

### **📈 TESTING RESULTS**
```
testing_results/
├── backtest_runner.py       # Main backtest runner
├── result_comparator.py     # Compare different strategies
├── performance_analyzer.py  # Detailed performance analysis
├── result_validator.py      # Validate backtest results
└── README.md              # Testing documentation
```

**Purpose**: Execute backtests and manage results.

**Key Features**:
- Run individual and portfolio backtests
- Aggregate results
- Generate reports and visualizations
- Compare different strategies

---

### **📊 REPORTS**
```
reports/
├── *.csv                   # Performance reports
├── *.png                   # Visualizations and charts
├── *.txt                   # Analysis reports
└── README.md              # Reports documentation
```

**Purpose**: Store all generated reports and visualizations.

**Report Types**:
- Performance metrics (CSV)
- Equity curves and charts (PNG)
- Detailed analysis reports (TXT)
- Backtest summaries

---

### **📝 LOGS**
```
logs/
├── backtest_logs_*/       # Backtest execution logs
├── error_logs/           # Error logs
├── performance_logs/      # Performance logs
└── README.md            # Logs documentation
```

**Purpose**: Store all execution logs for debugging and analysis.

**Log Types**:
- Backtest execution logs
- Error logs
- Performance metrics logs
- Debug information

---

## **🚀 USAGE GUIDE**

### **1. DATA PROCESSING**
```python
from data_processing.data_loader import load_data_with_validation
from data_processing.data_loader import create_robust_pairs

# Load and validate data
stocks_data, loader = load_data_with_validation("data_folder")

# Create robust pairs
pairs = create_robust_pairs(stocks_data, min_correlation=0.3, max_pairs=50)
```

### **2. SIGNAL GENERATION**
```python
from signal_generation.signal_generators import MeanReversionSignals, HybridSignals

# Create signal generator
signal_gen = MeanReversionSignals(entry_z=2.0, exit_z=0.5)

# Generate signals
signals, z_scores = signal_gen.generate_signals(spread)
```

### **3. BACKTESTING**
```python
from backtesting_systems.optimized_pairs_trading_system import RealisticPairsTradingSystem

# Initialize backtesting system
system = RealisticPairsTradingSystem("data_folder")

# Run backtest
system.run_realistic_backtesting()
```

### **4. RESULTS MANAGEMENT**
```python
from testing_results.backtest_runner import BacktestRunner

# Initialize runner
runner = BacktestRunner()

# Run backtest
result = runner.run_portfolio_backtest(strategy, pairs, data, params)

# Generate report
runner.generate_report()
runner.create_visualizations()
```

---

## **📋 FILE ORGANIZATION BENEFITS**

### **✅ ADVANTAGES**:
1. **Modular Design**: Each component is separate and focused
2. **Easy Maintenance**: Clear separation of concerns
3. **Scalability**: Easy to add new strategies or features
4. **Reusability**: Components can be reused across different projects
5. **Debugging**: Easy to isolate and fix issues
6. **Documentation**: Each folder has its own documentation

### **🎯 WORKFLOW**:
1. **Data Processing** → Load and prepare data
2. **Signal Generation** → Generate trading signals
3. **Backtesting** → Execute backtests
4. **Results** → Analyze and compare results
5. **Reports** → Generate final reports and visualizations

### **📊 DATA FLOW**:
```
Raw Data → Data Processing → Signal Generation → Backtesting → Results → Reports
```

---

## **🔧 GETTING STARTED**

### **1. Install Dependencies**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### **2. Run Complete Pipeline**:
```python
# 1. Load data
from data_processing.data_loader import load_data_with_validation
stocks_data, loader = load_data_with_validation(".")

# 2. Create pairs
pairs = loader.create_pairs(max_pairs=20)

# 3. Run backtest
from backtesting_systems.optimized_pairs_trading_system import RealisticPairsTradingSystem
system = RealisticPairsTradingSystem(".")
system.run_realistic_backtesting()

# 4. Generate reports
from testing_results.backtest_runner import BacktestRunner
runner = BacktestRunner()
runner.generate_report()
```

### **3. Check Results**:
- Reports: `reports/` folder
- Logs: `logs/` folder
- Visualizations: `reports/*.png`

---

## **📝 NOTES**

- **Data Folder**: Contains raw CSV files (3minute, 5minute, 10minute, 15minute)
- **Main Entry Point**: `backtesting_systems/optimized_pairs_trading_system.py`
- **Configuration**: Each system has its own parameters
- **Extensibility**: Easy to add new strategies to any folder

---

**🎉 ORGANIZATION COMPLETE!**

**📁 All files are now properly segregated into specialized folders for better organization and maintainability!**
