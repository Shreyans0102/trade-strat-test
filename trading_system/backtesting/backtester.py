import backtrader as bt
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging

# Add parent directory to path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import INITIAL_CAPITAL, START_DATE, END_DATE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, strategy_class, data_path, symbol, timeframe='1d'):
        """Initialize backtester with strategy and data"""
        self.strategy_class = strategy_class
        self.data_path = data_path
        self.symbol = symbol
        self.timeframe = timeframe
        
    def run(self, plot=True, **strategy_params):
        """Run backtest with given parameters"""
        # Create cerebro instance
        cerebro = bt.Cerebro()
        
        # Add the strategy with parameters
        cerebro.addstrategy(self.strategy_class, **strategy_params)
        
        # Load data
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found: {self.data_path}")
            return None
            
        logger.info(f"Loading data from: {self.data_path}")
        data = bt.feeds.YahooFinanceCSVData(
            dataname=self.data_path,
            fromdate=pd.to_datetime(START_DATE),
            todate=pd.to_datetime(END_DATE),
            dtformat='%Y-%m-%d',
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1
        )
        
        # Add data to cerebro
        cerebro.adddata(data)
        
        # Set starting cash
        cerebro.broker.setcash(INITIAL_CAPITAL)
        
        # Set commission - 0.1% per trade
        cerebro.broker.setcommission(commission=0.001)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Log starting conditions
        logger.info(f"Starting backtest with {INITIAL_CAPITAL:.2f} initial capital")
        
        # Run the backtest
        results = cerebro.run()
        strategy = results[0]
        
        # Calculate results
        final_value = cerebro.broker.getvalue()
        pnl = final_value - INITIAL_CAPITAL
        pnl_pct = (pnl / INITIAL_CAPITAL) * 100
        
        # Get analyzer results
        sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_dd = strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        ret = strategy.analyzers.returns.get_analysis().get('rtot', 0) * 100
        
        # Trade stats
        trade_analysis = strategy.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get('total', 0)
        win_rate = (trade_analysis.get('won', 0) / max(1, total_trades)) * 100
        
        # Log results
        logger.info(f"Backtest completed. Final value: {final_value:.2f}")
        logger.info(f"P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
        logger.info(f"Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"Max Drawdown: {max_dd:.2f}%")
        logger.info(f"Total Return: {ret:.2f}%")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        
        # Plot if requested
        if plot:
            cerebro.plot(style='candlestick', barup='green', bardown='red', 
                        volup='green', voldown='red', plotdist=1)
        
        # Return results for further analysis
        return {
            'final_value': final_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_return': ret,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'strategy': strategy
        }