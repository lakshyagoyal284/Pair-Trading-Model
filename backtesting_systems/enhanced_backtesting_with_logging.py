"""
ENHANCED BACKTESTING SYSTEM WITH AUTOMATIC LOGGING
This version automatically captures all terminal output and saves to timestamped files
"""

# Import the enhanced backtesting system
from enhanced_backtesting_system import EnhancedBacktestingSystem
from logging_system import BacktestLogger, start_backtest_logging, stop_backtest_logging

class EnhancedBacktestingWithLogging:
    """
    Wrapper class that adds automatic logging to the enhanced backtesting system
    """
    
    def __init__(self, data_folder="..", log_folder="backtest_logs"):
        self.data_folder = data_folder
        self.log_folder = log_folder
        self.backtesting_system = EnhancedBacktestingSystem(data_folder)
        
    def run_with_logging(self):
        """Run backtesting with automatic logging"""
        print("🚀 ENHANCED BACKTESTING WITH AUTOMATIC LOGGING")
        print("="*80)
        print("📝 All terminal output will be automatically saved to log files")
        print("📁 Log files will be stored in: backtest_logs/")
        print("="*80)
        
        # Initialize logger
        logger = BacktestLogger(self.log_folder)
        logger.start_logging()
        
        try:
            # Run the enhanced backtesting system
            self.backtesting_system.run_enhanced_backtesting()
            
            # Log completion summary
            logger.log_success("Enhanced backtesting completed successfully")
            
            # Create summary of log files
            self.create_log_summary()
            
        except Exception as e:
            logger.log_error(f"Backtesting failed: {str(e)}")
            raise
        finally:
            # Always stop logging
            logger.stop_logging()
            
            print(f"\n📝 LOGGING COMPLETED!")
            print(f"📁 Log files saved to: {self.log_folder}/")
            print(f"📊 Check the latest log file for complete backtesting details")
    
    def create_log_summary(self):
        """Create a summary of all log files"""
        import os
        from datetime import datetime
        
        log_files = []
        if os.path.exists(self.log_folder):
            log_files = [f for f in os.listdir(self.log_folder) if f.endswith('.txt')]
            log_files.sort(reverse=True)  # Most recent first
        
        summary_file = os.path.join(self.log_folder, "log_summary.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("BACKTESTING LOG FILES SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Log Files: {len(log_files)}\n")
            f.write("="*50 + "\n\n")
            
            for i, log_file in enumerate(log_files, 1):
                file_path = os.path.join(self.log_folder, log_file)
                file_size = os.path.getsize(file_path)
                f.write(f"{i}. {log_file} ({file_size:,} bytes)\n")
        
        print(f"📋 Log summary created: {summary_file}")

# Main execution
if __name__ == "__main__":
    print("🚀 ENHANCED BACKTESTING WITH AUTOMATIC LOGGING")
    print("="*80)
    print("📝 This version automatically captures all terminal output")
    print("📁 Log files are saved with timestamps in 'backtest_logs/' folder")
    print("="*80)
    
    try:
        # Initialize and run with logging
        backtesting_with_logging = EnhancedBacktestingWithLogging()
        backtesting_with_logging.run_with_logging()
        
        print("\n🎉 BACKTESTING WITH LOGGING COMPLETED!")
        print("="*80)
        print("📋 Key Features:")
        print("✅ 1. All terminal output automatically captured")
        print("✅ 2. Timestamped log files created")
        print("✅ 3. Detailed trade-by-trade logging")
        print("✅ 4. Performance metrics logged")
        print("✅ 5. Error tracking and debugging info")
        print("✅ 6. Log file summary created")
        print("\n🚀 CHECK THE 'backtest_logs/' FOLDER FOR DETAILED OUTPUT!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check your data files and try again.")
