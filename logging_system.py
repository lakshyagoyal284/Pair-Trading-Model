"""
LOGGING SYSTEM FOR PAIRS TRADING BACKTESTING
Automatically captures all terminal output and saves to files
"""

import sys
import os
from datetime import datetime
import logging
import io
from contextlib import redirect_stdout, redirect_stderr

class BacktestLogger:
    """
    Enhanced logging system that captures all terminal output
    during backtesting and saves to timestamped files
    """
    
    def __init__(self, log_folder="logs"):
        self.log_folder = log_folder
        self.log_file = None
        self.stdout_backup = None
        self.stderr_backup = None
        self.log_buffer = io.StringIO()
        
        # Create logs folder if it doesn't exist
        os.makedirs(log_folder, exist_ok=True)
        
        # Setup file logging
        self.setup_file_logging()
        
    def setup_file_logging(self):
        """Setup file-based logging with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_folder, f"backtest_log_{timestamp}.txt")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def start_logging(self):
        """Start capturing all terminal output"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create log header
        header = f"""
{'='*80}
PAIRS TRADING BACKTESTING LOG
Session Started: {timestamp}
Log File: {self.log_file}
{'='*80}

"""
        
        # Write header to file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)
        
        # Backup original stdout and stderr
        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr
        
        # Create custom output writers
        self.stdout_writer = OutputWriter(self.stdout_backup, self.log_file)
        self.stderr_writer = OutputWriter(self.stderr_backup, self.log_file, is_error=True)
        
        # Redirect stdout and stderr
        sys.stdout = self.stdout_writer
        sys.stderr = self.stderr_writer
        
        print(f"📝 Logging started: All output will be saved to {self.log_file}")
        
    def stop_logging(self):
        """Stop capturing and restore original output"""
        if self.stdout_backup and self.stderr_backup:
            # Restore original stdout and stderr
            sys.stdout = self.stdout_backup
            sys.stderr = self.stderr_backup
            
            # Write log footer
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            footer = f"""

{'='*80}
BACKTESTING SESSION COMPLETED
Session Ended: {timestamp}
Log File: {self.log_file}
{'='*80}
"""
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(footer)
            
            print(f"✅ Logging stopped: Log saved to {self.log_file}")
            
    def log_section(self, section_name):
        """Log a section header"""
        separator = f"\n{'-'*60}\n{section_name.upper()}\n{'-'*60}\n"
        print(separator)
        
    def log_metrics(self, metrics_dict, title="PERFORMANCE METRICS"):
        """Log performance metrics in a formatted way"""
        print(f"\n📊 {title}")
        print("-" * 50)
        
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:+.2f}")
            elif isinstance(value, str):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        print("-" * 50)
        
    def log_trade(self, trade_info):
        """Log individual trade information"""
        print(f"📈 TRADE EXECUTED:")
        print(f"  Pair: {trade_info.get('pair', 'N/A')}")
        print(f"  Action: {trade_info.get('action', 'N/A')}")
        print(f"  Entry: {trade_info.get('entry_date', 'N/A')}")
        print(f"  Exit: {trade_info.get('exit_date', 'N/A')}")
        print(f"  P&L: ${trade_info.get('pnl', 0):+,.2f}")
        print(f"  Reason: {trade_info.get('exit_reason', 'N/A')}")
        
    def log_error(self, error_message):
        """Log error messages"""
        print(f"❌ ERROR: {error_message}")
        
    def log_warning(self, warning_message):
        """Log warning messages"""
        print(f"⚠️  WARNING: {warning_message}")
        
    def log_success(self, success_message):
        """Log success messages"""
        print(f"✅ SUCCESS: {success_message}")

class OutputWriter:
    """
    Custom output writer that writes to both terminal and file
    """
    
    def __init__(self, original_stream, log_file, is_error=False):
        self.original_stream = original_stream
        self.log_file = log_file
        self.is_error = is_error
        
    def write(self, text):
        """Write to both original stream and log file"""
        # Write to original stream (terminal)
        self.original_stream.write(text)
        self.original_stream.flush()
        
        # Write to log file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                if self.is_error:
                    f.write(f"[ERROR] {text}")
                else:
                    f.write(text)
        except Exception as e:
            # If file writing fails, at least terminal output works
            pass
            
    def flush(self):
        """Flush both streams"""
        self.original_stream.flush()

class BacktestDecorator:
    """
    Decorator class to automatically add logging to any backtesting function
    """
    
    def __init__(self, log_folder="logs"):
        self.log_folder = log_folder
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Initialize logger
            logger = BacktestLogger(self.log_folder)
            
            try:
                # Start logging
                logger.start_logging()
                
                # Log function start
                logger.log_section(f"STARTING {func.__name__.upper()}")
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Log function completion
                logger.log_success(f"{func.__name__} completed successfully")
                
                return result
                
            except Exception as e:
                # Log error
                logger.log_error(f"Error in {func.__name__}: {str(e)}")
                raise
                
            finally:
                # Always stop logging
                logger.stop_logging()
                
        return wrapper

# Global logger instance
_global_logger = None

def get_logger():
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = BacktestLogger()
    return _global_logger

def start_backtest_logging(log_folder="logs"):
    """Start global backtest logging"""
    global _global_logger
    _global_logger = BacktestLogger(log_folder)
    _global_logger.start_logging()
    return _global_logger

def stop_backtest_logging():
    """Stop global backtest logging"""
    global _global_logger
    if _global_logger:
        _global_logger.stop_logging()

def log_backtest_section(section_name):
    """Log a section header"""
    logger = get_logger()
    logger.log_section(section_name)

def log_backtest_metrics(metrics_dict, title="PERFORMANCE METRICS"):
    """Log performance metrics"""
    logger = get_logger()
    logger.log_metrics(metrics_dict, title)

def log_backtest_trade(trade_info):
    """Log individual trade"""
    logger = get_logger()
    logger.log_trade(trade_info)

# Decorator for automatic logging
def log_backtest(log_folder="logs"):
    """Decorator to automatically log backtesting functions"""
    return BacktestDecorator(log_folder)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 TESTING LOGGING SYSTEM")
    
    # Test basic logging
    logger = BacktestLogger()
    
    logger.start_logging()
    
    logger.log_section("TESTING SECTION")
    logger.log_success("This is a success message")
    logger.log_warning("This is a warning message")
    logger.log_error("This is an error message")
    
    # Test metrics logging
    metrics = {
        "Total Return": 25.5,
        "Sharpe Ratio": 1.2,
        "Max Drawdown": -10.3,
        "Win Rate": 65.0
    }
    logger.log_metrics(metrics, "TEST METRICS")
    
    # Test trade logging
    trade = {
        "pair": "AAPL-MSFT",
        "action": "LONG",
        "entry_date": "2022-01-01",
        "exit_date": "2022-01-03",
        "pnl": 1250.50,
        "exit_reason": "Target reached"
    }
    logger.log_trade(trade)
    
    logger.stop_logging()
    
    print("✅ Logging system test completed!")
