import os
import sys
import logging
from data.downloader import DataDownloader
from strategies.smc_strategy import SMCStrategy
from backtesting.backtester import Backtester
from config.settings import SYMBOLS, TIMEFRAMES

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the trading system"""
    logger.info("Starting trading system")
    
    # Step 1: Download data if needed
    data_dir = "trading_system/data/raw"
    downloader = DataDownloader(data_dir=data_dir)
    downloader.download_data()
    
    # Step 2: Run backtests
    results = {}
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            logger.info(f"Running backtest for {symbol} {timeframe}")
            
            # Create path to data file
            data_path = os.path.join(data_dir, f"{symbol.replace('=', '_')}_{timeframe}.csv")
            
            # Check if data file exists
            if not os.path.exists(data_path):
                logger.warning(f"Data file not found: {data_path}")
                continue
            
            # Run backtest
            backtester = Backtester(
                strategy_class=SMCStrategy,
                data_path=data_path,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Run with default parameters
            result = backtester.run(
                plot=True,  # Set to False for headless servers
                order_block_lookback=5,
                sweep_lookback=10,
                min_block_size=0.5,
                risk_pct=1.0,
                risk_reward=3.0
            )
            
            # Store results
            if result:
                results[(symbol, timeframe)] = result
    
    # Step 3: Print summary of results
    logger.info("\n=== SUMMARY OF RESULTS ===")
    for (symbol, timeframe), result in results.items():
        logger.info(f"{symbol} {timeframe}: PnL: {result['pnl_pct']:.2f}%, Win Rate: {result['win_rate']:.2f}%, Sharpe: {result['sharpe']:.2f}")
    
    logger.info("Trading system execution complete")

if __name__ == "__main__":
    main()