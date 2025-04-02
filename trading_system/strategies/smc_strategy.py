import backtrader as bt
import numpy as np
from .base_strategy import BaseStrategy

class SMCStrategy(BaseStrategy):
    """
    Implementation of an SMC strategy using liquidity sweeps and order blocks
    """
    params = (
        ('order_block_lookback', 5),   # How far back to look for order blocks
        ('sweep_lookback', 10),        # How far back to look for liquidity sweeps
        ('min_block_size', 0.5),       # Minimum relative size of order block
    )
    
    def __init__(self):
        """Initialize SMC strategy components"""
        # Initialize the parent class first
        super(SMCStrategy, self).__init__()
        
        # Data structures for tracking SMC components
        self.highs = []  # Recent highs for market structure
        self.lows = []   # Recent lows for market structure
        self.order_blocks = []  # Identified order blocks
        
    def next(self):
        """Main strategy logic executed for each bar"""
        # Store price data for analysis
        self.highs.append(self.high[0])
        self.lows.append(self.low[0])
        
        # Limit the arrays to conserve memory
        if len(self.highs) > 200:
            self.highs.pop(0)
        if len(self.lows) > 200:
            self.lows.pop(0)
        
        # Skip if not enough data
        if len(self.data) < 50:
            return
            
        # Identify order blocks
        self.identify_order_blocks()
        
        # Generate signals only if we don't have a position
        if not self.position:
            # Bullish setup: uptrend + liquidity sweep + near order block
            if self.sma50[0] > self.sma200[0]:  # Basic uptrend
                if self.detect_liquidity_sweep('long') and self.is_near_orderblock('bullish'):
                    self.enter_long()
                
            # Bearish setup: downtrend + liquidity sweep + near order block
            elif self.sma50[0] < self.sma200[0]:  # Basic downtrend
                if self.detect_liquidity_sweep('short') and self.is_near_orderblock('bearish'):
                    self.enter_short()
    
    def identify_order_blocks(self):
        """Identify potential order blocks"""
        # Need at least 3 bars of data
        if len(self.data) < 3:
            return
            
        # Identify bearish candle (potential bullish order block)
        if (self.close[-2] < self.open[-2] and 
            (self.open[-2] - self.close[-2]) / (self.high[-2] - self.low[-2]) > self.p.min_block_size):
            # If followed by bullish price action
            if self.close[-1] > self.open[-1] and self.close[0] > self.close[-1]:
                # Add as potential bullish order block
                self.order_blocks.append({
                    'type': 'bullish',
                    'top': self.high[-2],
                    'bottom': self.low[-2],
                    'bar': len(self.data) - 2
                })
        
        # Identify bullish candle (potential bearish order block)
        if (self.close[-2] > self.open[-2] and 
            (self.close[-2] - self.open[-2]) / (self.high[-2] - self.low[-2]) > self.p.min_block_size):
            # If followed by bearish price action
            if self.close[-1] < self.open[-1] and self.close[0] < self.close[-1]:
                # Add as potential bearish order block
                self.order_blocks.append({
                    'type': 'bearish',
                    'top': self.high[-2],
                    'bottom': self.low[-2],
                    'bar': len(self.data) - 2
                })
                
        # Keep only recent order blocks to save memory
        if len(self.order_blocks) > 20:
            self.order_blocks.pop(0)
    
    def detect_liquidity_sweep(self, direction):
        """
        Detect if price has swept liquidity (broken key levels) and then reversed
        """
        # Need enough price history
        if len(self.highs) < self.p.sweep_lookback or len(self.lows) < self.p.sweep_lookback:
            return False
            
        if direction == 'long':
            # Find the lowest recent low (excluding current bar)
            recent_low = min(self.lows[-self.p.sweep_lookback:-1])
            # Check if price broke below then reversed up
            return (self.low[0] < recent_low and self.close[0] > recent_low)
                
        elif direction == 'short':
            # Find the highest recent high (excluding current bar)
            recent_high = max(self.highs[-self.p.sweep_lookback:-1])
            # Check if price broke above then reversed down
            return (self.high[0] > recent_high and self.close[0] < recent_high)
            
        return False
        
    def is_near_orderblock(self, ob_type):
        """Check if current price is near a relevant order block"""
        # Check if we have any order blocks
        if not self.order_blocks:
            return False
            
        # Get relevant order blocks
        relevant_obs = [ob for ob in self.order_blocks if ob['type'] == ob_type]
        if not relevant_obs:
            return False
            
        # Check if price is near/inside any of them
        for ob in relevant_obs:
            if ob_type == 'bullish' and self.low[0] >= ob['bottom'] and self.low[0] <= ob['top']:
                return True
            if ob_type == 'bearish' and self.high[0] <= ob['top'] and self.high[0] >= ob['bottom']:
                return True
                
        return False