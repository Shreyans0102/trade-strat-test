import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import sys
import logging

# Add parent directory to path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SYMBOLS, TIMEFRAMES, START_DATE, END_DATE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataDownloader:
    def __init__(self, data_dir="trading_system/data/raw"):
        """Initialize with directory for storing data"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_data(self):
        """Download historical data for all symbols and timeframes"""
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                logger.info(f"Downloading {symbol} {timeframe} data...")
                try:
                    data = yf.download(
                        tickers=symbol,
                        start=START_DATE,
                        end=END_DATE,
                        interval=timeframe
                    )
                    
                    if not data.empty:
                        filename = f"{symbol.replace('=', '_')}_{timeframe}.csv"
                        filepath = os.path.join(self.data_dir, filename)
                        data.to_csv(filepath)
                        logger.info(f"Saved to {filepath}")
                    else:
                        logger.warning(f"No data available for {symbol} {timeframe}")
                except Exception as e:
                    logger.error(f"Error downloading {symbol} {timeframe}: {e}")
        
        return True

if __name__ == "__main__":
    # When run as script, download all data
    downloader = DataDownloader()
    downloader.download_data()