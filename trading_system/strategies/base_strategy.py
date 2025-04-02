import backtrader as bt
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(bt.Strategy):
    """Base strategy class that all strategies should inherit from"""
    
    params = (
        ('risk_pct', 1.0),       # Risk per trade as percentage of portfolio
        ('risk_reward', 3.0),    # Risk-reward ratio for take profit
    )
    
    def __init__(self):
        """Initialize strategy components"""
        # Common indicators
        self.sma50 = bt.indicators.SMA(self.data.close, period=50)
        self.sma200 = bt.indicators.SMA(self.data.close, period=200)
        
        # Easy access to data
        self.close = self.data.close
        self.high = self.data.high
        self.low = self.data.low
        self.open = self.data.open
        self.volume = self.data.volume
        
        # Track open orders
        self.orders = {}
        self.stop_orders = []
        self.target_orders = []
        
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        """Track order status"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - no action required
            return

        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

    def next(self):
        """Main strategy logic - to be implemented by subclasses"""
        pass
    
    def enter_long(self, size=None, stop_price=None):
        """Enter a long position with proper risk management"""
        # Calculate position size based on risk
        if size is None:
            price = self.close[0]
            
            # If no stop price provided, use a default
            if stop_price is None:
                stop_price = self.low[-5]  # Simple default
            
            risk_amount = price - stop_price
            if risk_amount <= 0:
                self.log("Invalid stop price for long position")
                return
                
            # Calculate position size
            cash = self.broker.getcash()
            risk_cash = cash * (self.p.risk_pct / 100)
            size = risk_cash / risk_amount
            
            # Round size to standard lot
            size = max(0.01, round(size, 2))
            
        # Calculate take profit based on risk
        target_price = price + (risk_amount * self.p.risk_reward)
        
        # Place orders
        self.log(f'LONG ORDER PLACED, Price: {price:.2f}, Stop: {stop_price:.2f}, Target: {target_price:.2f}, Size: {size:.2f}')
        buy_order = self.buy(size=size)
        stop_order = self.sell(exectype=bt.Order.Stop, price=stop_price, size=size)
        target_order = self.sell(exectype=bt.Order.Limit, price=target_price, size=size)
        
        # Store orders for management
        self.orders[buy_order.ref] = {
            'type': 'entry',
            'direction': 'long',
            'price': price,
            'stop': stop_order,
            'target': target_order
        }
        
    def enter_short(self, size=None, stop_price=None):
        """Enter a short position with proper risk management"""
        # Calculate position size based on risk
        if size is None:
            price = self.close[0]
            
            # If no stop price provided, use a default
            if stop_price is None:
                stop_price = self.high[-5]  # Simple default
            
            risk_amount = stop_price - price
            if risk_amount <= 0:
                self.log("Invalid stop price for short position")
                return
                
            # Calculate position size
            cash = self.broker.getcash()
            risk_cash = cash * (self.p.risk_pct / 100)
            size = risk_cash / risk_amount
            
            # Round size to standard lot
            size = max(0.01, round(size, 2))
            
        # Calculate take profit based on risk
        target_price = price - (risk_amount * self.p.risk_reward)
        
        # Place orders
        self.log(f'SHORT ORDER PLACED, Price: {price:.2f}, Stop: {stop_price:.2f}, Target: {target_price:.2f}, Size: {size:.2f}')
        sell_order = self.sell(size=size)
        stop_order = self.buy(exectype=bt.Order.Stop, price=stop_price, size=size)
        target_order = self.buy(exectype=bt.Order.Limit, price=target_price, size=size)
        
        # Store orders for management
        self.orders[sell_order.ref] = {
            'type': 'entry',
            'direction': 'short',
            'price': price,
            'stop': stop_order,
            'target': target_order
        }