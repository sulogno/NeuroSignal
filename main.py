from binance.client import Client
from dotenv import load_dotenv
import pandas as pd
import ta
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import linregress

class EnhancedSignalGenerator:   
    # Optimized parameters
    RISK_REWARD_RATIO = 2.8  # Increased for better profit potential
    BASE_SL_MULTIPLIER = 1.5  # Base stop loss multiplier
    TP_MULTIPLIER = 3.2       # Wider take profit
    MIN_ATR_RATIO = 0.01      # Minimum volatility threshold
    SIGNAL_FILE = "signals.json"
    WINDOW_SIZE = 5           # Larger window for smoother signals
    MIN_TREND_STRENGTH = 20   # ADX threshold for valid trends
    MIN_VOLUME_RATIO = 1.2    # Strong volume confirmation
    MAX_TRADES_PER_DAY = 5    # Prevent overtrading
    
    # Weighting system for indicators (adjusted based on importance)
    INDICATOR_WEIGHTS = {
        'ema_cross': 3.2,
        'adx': 2.8,
        'macd': 2.2,
        'breakout': 2.2,
        'rsi': 1.8,
        'bb': 1.8,
        'candlestick': 1.2,
        'volume': 1.2,
        'volatility': 1.2,
        'trend_strength': 1.8,
        'time_filter': 0.8,
        'liquidity': 1.5
    }
    
    TIMEFRAMES = {
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR
    }

    def __init__(self, symbol='BTCUSDT', score_threshold=0.6, debug=False, window_size=5, verbose=False):
        load_dotenv()
        self.client = self._create_client()
        self.symbol = symbol
        self.score_threshold = score_threshold
        self.debug = debug
        self.indicator_data = {}
        self.max_possible_score = sum(self.INDICATOR_WEIGHTS.values())
        self.WINDOW_SIZE = window_size
        self.verbose = verbose
        self.trade_journal = []
        self.daily_trade_count = 0
        self.last_trade_day = None
        self.performance_tracker = PerformanceTracker()

    def _create_client(self):
        """Initialize Binance API client"""
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET_KEY")
        return Client(api_key, api_secret)

    def reset_daily_counts(self):
        """Reset trade counts at midnight"""
        current_day = datetime.now().date()
        if self.last_trade_day != current_day:
            self.daily_trade_count = 0
            self.last_trade_day = current_day

    def calculate_position_size(self, balance, atr, entry, sl):
        """Volatility-adjusted position sizing"""
        risk_pct = 0.01  # 1% risk per trade
        risk_amount = balance * risk_pct
        risk_per_unit = abs(entry - sl)
        position_size = risk_amount / risk_per_unit
        return min(position_size, balance * 0.1)  # Cap at 10% of balance

    def detect_potential_reversal(self, df):
        """Detect possible reversal using MACD divergence"""
        if len(df) < 15:
            return "NONE"

        macd_hist = df['macd_hist'][-5:]
        close = df['close'][-5:]

        price_slope = linregress(range(5), close).slope
        macd_slope = linregress(range(5), macd_hist).slope

        if price_slope < 0 and macd_slope > 0:
            return "BULLISH"
        elif price_slope > 0 and macd_slope < 0:
            return "BEARISH"
        else:
            return "NONE"

    

    def is_optimal_trading_hour(self, timestamp):
        """Filter for high liquidity periods"""
        hour = timestamp.hour
        # London/New York overlap + Asian session open
        optimal_hours = {2, 3, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19}

        return hour in optimal_hours

    def detect_volatility_breakout(self, window):
        """Identify high-probability breakout setups"""
        latest = window.iloc[-1]
        prev = window.iloc[-2]
        
        return (
            (latest['bb_width_pct'] > 0.045) and  # Expanding volatility
            (latest['volume'] > prev['volume'] * 1.9) and
            (abs(latest['close'] - latest['open']) > latest['atr'] * 0.75)
        )

    def confirm_trend(self, window):
        """Require 3/4 indicators to agree on trend"""
        latest = window.iloc[-1]
        conditions = {
            'ema': latest['ema9'] > latest['ema21'] > latest['ema50'],
            'macd': latest['macd'] > latest['macd_signal'],
            'supertrend': latest['supertrend_direction'] == "up",
            'adx': (latest['adx'] > 25) and (latest['plus_di'] > latest['minus_di'])
        }
        return sum(conditions.values()) >= 3

    def get_optimal_entry(self, window):
        """Wait for pullback in strong trends"""
        latest = window.iloc[-1]
        
        if latest['adx'] > 30:  # Strong trend
            if latest['plus_di'] > latest['minus_di']:  # Uptrend
                return latest['close'] <= latest['ema21']  # Buy dip
            else:  # Downtrend
                return latest['close'] >= latest['ema21']  # Sell rally
        return True  # No filter in weak trends

    def has_sufficient_liquidity(self, window):
        """Check order book depth (simplified)"""
        latest = window.iloc[-1]
        return latest['volume'] > window['volume'].rolling(20).mean().iloc[-1] * 1.3


    def calculate_indicators(self, df):
        if df is None or df.empty:
            print("❌ Skipping indicator calculation due to empty DataFrame")
            return None

        """Calculate all technical indicators with enhanced features"""
        df = self.detect_candlestick_patterns(df)

        # Momentum indicators
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        # Trend indicators
        df['ema9'] = ta.trend.EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema21'] = ta.trend.EMAIndicator(close=df['close'], window=21).ema_indicator()
        df['ema50'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['ema100'] = ta.trend.EMAIndicator(close=df['close'], window=100).ema_indicator()
        df['ema200'] = ta.trend.EMAIndicator(close=df['close'], window=200).ema_indicator()

        macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        # Volatility indicators
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2.0)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_width_pct'] = df['bb_width'] / df['close']
        df['bb_percent'] = bb.bollinger_pband()

        # Support/Resistance
        df['recent_high'] = df['high'].rolling(window=20).max()
        df['recent_low'] = df['low'].rolling(window=20).min()

        # Volume
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']

        # Breakout
        df['breakout_up'] = (df['close'] > df['recent_high'].shift(1)) & (df['volume'] > (1.6 * df['volume_ma20']))
        df['breakout_down'] = (df['close'] < df['recent_low'].shift(1)) & (df['volume'] > (1.6 * df['volume_ma20']))

        # ADX
        adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx.adx()
        df['plus_di'] = adx.adx_pos()
        df['minus_di'] = adx.adx_neg()

        # ATR
        atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = df['atr'] / df['close']

        # VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()

        # Supertrend
        df = self.compute_supertrend(df)

        # Final cleanup
        df.dropna(inplace=True)

        return df

    def calculate_stop_loss(self, entry_price, atr, trend_strength, direction):
        """Adaptive stop loss based on trend strength"""
        if abs(trend_strength) > 1.2:  # Strong trend
            multiplier = self.BASE_SL_MULTIPLIER * 1.8
        else:
            multiplier = self.BASE_SL_MULTIPLIER * 0.8

        if direction == "BUY":
            return entry_price - (atr * multiplier)
        else:
            return entry_price + (atr * multiplier)

    def calculate_take_profit(self, entry_price, atr, trend_strength, direction):
        sl = self.calculate_stop_loss(entry_price, atr, trend_strength, direction)
        if direction == "BUY":
            return entry_price + self.RISK_REWARD_RATIO * (entry_price - sl)
        else:
            return entry_price - self.RISK_REWARD_RATIO * (sl - entry_price)

    


    def compute_supertrend(self, df, period=10, multiplier=3.0):
        """Custom Supertrend implementation"""
        hl2 = (df['high'] + df['low']) / 2
        atr = df['atr']

        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)

        supertrend = [True]  # First row is bullish
        direction = []

        for i in range(1, len(df)):
            curr_close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i - 1]
            prev_upper = upperband.iloc[i - 1]
            prev_lower = lowerband.iloc[i - 1]
            prev_st = supertrend[-1]

            if prev_st:
                if curr_close <= lowerband.iloc[i]:
                    supertrend.append(False)
                else:
                    upperband.iloc[i] = min(upperband.iloc[i], prev_upper)
                    supertrend.append(True)
            else:
                if curr_close >= upperband.iloc[i]:
                    supertrend.append(True)
                else:
                    lowerband.iloc[i] = max(lowerband.iloc[i], prev_lower)
                    supertrend.append(False)

            direction.append("up" if supertrend[-1] else "down")

        direction = [None] + direction  # First row has no direction
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        return df

    def detect_candlestick_patterns(self, df):
        """Enhanced candlestick pattern detection with validation"""

        # Defensive check to avoid processing bad or empty DataFrames
        if df is None or df.empty or 'close' not in df.columns or 'open' not in df.columns:
            print("❌ Candlestick pattern detection skipped due to invalid or empty DataFrame")
            return df

        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)

        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)

        df['hammer'] = (
            (df['close'] > df['open']) &
            ((df['close'] - df['open']) > 0.7 * (df['high'] - df['low'])) &
            ((df['high'] - df['close']) < 0.2 * (df['high'] - df['low']))
        ).astype(int)

        df['shooting_star'] = (
            (df['close'] < df['open']) &
            ((df['open'] - df['close']) > 0.7 * (df['high'] - df['low'])) &
            ((df['high'] - df['open']) < 0.2 * (df['high'] - df['low']))
        ).astype(int)

        return df

    def confirm_candlestick(self, candle):
        """Enhanced candlestick pattern detection with volume confirmation"""
        open_ = candle['open']
        close = candle['close']
        high = candle['high']
        low = candle['low']
        body = abs(close - open_)
        wick = high - low
        body_ratio = body / wick if wick > 0 else 0
        
        # Volume confirmation
        volume_ok = candle['volume'] > candle['volume_ma20'] * 1.6
        
        # Basic patterns
        if body_ratio < 0.25:
            return "DOJI"
        elif candle['bullish_engulfing'] and volume_ok:
            return "BULLISH_ENGULF"
        elif candle['bearish_engulfing'] and volume_ok:
            return "BEARISH_ENGULF"
        elif candle['hammer'] and volume_ok:
            return "HAMMER"
        elif candle['shooting_star'] and volume_ok:
            return "SHOOTING_STAR"
        return "NONE"
    
    def calculate_trend_strength(self, prices):
        """Calculate trend strength using linear regression"""
        x = np.arange(len(prices))
        slope, _, _, _, _ = linregress(x, prices)
        return slope * 100  # Scale for better interpretation

    def _get_smoothed_value(self, df, column, window_size=5):
        """Get smoothed value using rolling average"""
        return df[column].rolling(window=window_size).mean().iloc[-1]

    def _get_condition_flags(self, window):
        """Calculate condition flags with enhanced trend detection"""
        latest = window.iloc[-1]
        
        # Smoothed values
        rsi_smoothed = self._get_smoothed_value(window, 'rsi')
        volume_ratio_smoothed = self._get_smoothed_value(window, 'volume_ratio')
        bb_width_pct_smoothed = self._get_smoothed_value(window, 'bb_width_pct')
        bb_percent_smoothed = self._get_smoothed_value(window, 'bb_percent')
        adx_smoothed = self._get_smoothed_value(window, 'adx')
        
        # Trend strength calculation
        trend_strength = self.calculate_trend_strength(window['close'].tail(10))
        
        # Candlestick pattern
        candlestick_pattern = self.confirm_candlestick(latest)
        
        # EMA cross
        ema_cross_buy = all(row['ema9'] > row['ema21'] > row['ema50'] for _, row in window.iterrows())
        ema_cross_sell = all(row['ema9'] < row['ema21'] < row['ema50'] for _, row in window.iterrows())
        
        # MACD with histogram confirmation
        macd_buy = (latest['macd'] > latest['macd_signal']) and (window['macd_hist'].iloc[-1] > window['macd_hist'].iloc[-2])
        macd_sell = (latest['macd'] < latest['macd_signal']) and (window['macd_hist'].iloc[-1] < window['macd_hist'].iloc[-2])
        
        # Breakout conditions
        breakout_up = latest['breakout_up']
        breakout_down = latest['breakout_down']
        
        # Supertrend confirmation
        supertrend_buy = all(row['supertrend_direction'] == "up" for _, row in window.iterrows())
        supertrend_sell = all(row['supertrend_direction'] == "down" for _, row in window.iterrows())

        # Time filter
        optimal_time = self.is_optimal_trading_hour(latest.name)
        
        # Liquidity check
        sufficient_liquidity = self.has_sufficient_liquidity(window)
        
        # Volatility breakout
        volatility_breakout = self.detect_volatility_breakout(window)

        #reversal
        reversal = self.detect_potential_reversal(window)

        
        return {
            'rsi_buy': 24 < rsi_smoothed < 32,
            'reversal_bullish': reversal == "BULLISH",
            'reversal_bearish': reversal == "BEARISH",
            'rsi_sell': 68 < rsi_smoothed < 76,
            'ema_buy': ema_cross_buy,
            'ema_sell': ema_cross_sell,
            'macd_buy': macd_buy,
            'macd_sell': macd_sell,
            'bb_buy': bb_percent_smoothed < 0.08,
            'bb_sell': bb_percent_smoothed > 0.92,
            'adx_strong': adx_smoothed > self.MIN_TREND_STRENGTH,
            'candlestick_buy': candlestick_pattern in ["BULLISH_ENGULF", "HAMMER"],
            'candlestick_sell': candlestick_pattern in ["BEARISH_ENGULF", "SHOOTING_STAR"],
            'volume_ok': volume_ratio_smoothed > self.MIN_VOLUME_RATIO,
            'volatility_ok': bb_width_pct_smoothed > 0.035,
            'trend_confirmation': latest['close'] > latest['ema100'],
            'breakout_buy': breakout_up,
            'breakout_sell': breakout_down,
            'candlestick_pattern': candlestick_pattern,
            'trend_up': latest['plus_di'] > latest['minus_di'],
            'trend_down': latest['minus_di'] > latest['plus_di'],
            'supertrend_buy': supertrend_buy,
            'supertrend_sell': supertrend_sell,
            'trend_strength': trend_strength,
            'optimal_time': optimal_time,
            'sufficient_liquidity': sufficient_liquidity,
            'volatility_breakout': volatility_breakout
        }

    def get_signal(self, window):
        """Enhanced signal generation with multi-filter confirmation"""
        self.reset_daily_counts()
        
        if self.daily_trade_count >= self.MAX_TRADES_PER_DAY:
            if self.debug:
                print(" - Daily trade limit reached")
            return "HOLD", 0, 0, None, None

        flags = self._get_condition_flags(window)
        latest = window.iloc[-1]

        # Choppy market filter
        if not flags['volatility_ok'] or latest['atr_pct'] < self.MIN_ATR_RATIO:
            if self.debug:
                print(" - Low volatility market: Avoiding trade")
            return "HOLD", 0, 0, None, None

        # Neutral zone filter
        if 42 < latest['rsi'] < 58 and abs(latest['macd_hist']) < 0.08:
            if self.debug:
                print(" - Neutral RSI and MACD: Avoiding trade")
            return "HOLD", 0, 0, None, None
        
        

        # Calculate scores
        buy_score = 0
        sell_score = 0






        # BUY conditions with stricter requirements
        if flags['rsi_buy']:
            buy_score += self.INDICATOR_WEIGHTS['rsi']
        if flags['ema_buy'] and flags['supertrend_buy']:
            buy_score += self.INDICATOR_WEIGHTS['ema_cross'] * 1.6
        if flags['macd_buy'] and flags['trend_up']:
            buy_score += self.INDICATOR_WEIGHTS['macd'] * 1.3
        if flags['bb_buy'] and flags['volatility_ok']:
            buy_score += self.INDICATOR_WEIGHTS['bb']
        if flags['adx_strong'] and flags['trend_up']:
            buy_score += self.INDICATOR_WEIGHTS['adx'] * 2.2
        if flags['candlestick_buy'] and flags['volume_ok']:
            buy_score += self.INDICATOR_WEIGHTS['candlestick'] * 1.6
        if flags['volume_ok']:
            buy_score += self.INDICATOR_WEIGHTS['volume']
        if flags['volatility_ok']:
            buy_score += self.INDICATOR_WEIGHTS['volatility']
        if flags['breakout_buy'] and flags['volume_ok']:
            buy_score += self.INDICATOR_WEIGHTS['breakout'] * 2.2
        if flags['trend_strength'] > 0.9:  # Strong upward trend
            buy_score += self.INDICATOR_WEIGHTS['trend_strength']
        if flags['optimal_time']:
            buy_score += self.INDICATOR_WEIGHTS['time_filter']
        if flags['sufficient_liquidity']:
            buy_score += self.INDICATOR_WEIGHTS['liquidity']
        if flags['volatility_breakout']:
            buy_score += 1.5  # Bonus for breakout

        # SELL conditions with stricter requirements
        if flags['rsi_sell']:
            sell_score += self.INDICATOR_WEIGHTS['rsi']
        if flags['ema_sell'] and flags['supertrend_sell']:
            sell_score += self.INDICATOR_WEIGHTS['ema_cross'] * 1.6
        if flags['macd_sell'] and flags['trend_down']:
            sell_score += self.INDICATOR_WEIGHTS['macd'] * 1.3
        if flags['bb_sell'] and flags['volatility_ok']:
            sell_score += self.INDICATOR_WEIGHTS['bb']
        if flags['adx_strong'] and flags['trend_down']:
            sell_score += self.INDICATOR_WEIGHTS['adx'] * 2.2
        if flags['candlestick_sell'] and flags['volume_ok']:
            sell_score += self.INDICATOR_WEIGHTS['candlestick'] * 1.6
        if flags['volume_ok']:
            sell_score += self.INDICATOR_WEIGHTS['volume']
        if flags['volatility_ok']:
            sell_score += self.INDICATOR_WEIGHTS['volatility']
        if flags['breakout_sell'] and flags['volume_ok']:
            sell_score += self.INDICATOR_WEIGHTS['breakout'] * 2.2
        if flags['trend_strength'] < -1.2:  # Strong downward trend
            sell_score += self.INDICATOR_WEIGHTS['trend_strength']
        if flags['optimal_time']:
            sell_score += self.INDICATOR_WEIGHTS['time_filter']
        if flags['sufficient_liquidity']:
            sell_score += self.INDICATOR_WEIGHTS['liquidity']
        if flags['volatility_breakout']:
            sell_score += 1.5  # Bonus for breakout

        # Trend bonus - only if trend is strong
        if flags['trend_confirmation'] and flags['adx_strong']:
            buy_score += 2.2
        else:
            sell_score += 2.2

        # Calculate confidence
        confidence = int((max(buy_score, sell_score) / self.max_possible_score * 100))
        
        # Volume boost for high confidence signals
        if flags['volume_ok'] and confidence > 75:
            confidence = min(100, int(confidence * 1.18))

        # Determine signal
        signal = "HOLD"
        score = 0
        
        # Require multiple confirmations
        trend_confirms_buy = int(flags['ema_buy']) + int(flags['macd_buy']) + int(flags['trend_up'])
        trend_confirms_sell = int(flags['ema_sell']) + int(flags['macd_sell']) + int(flags['trend_down'])

        if flags['reversal_bullish']:
            buy_score += 1.2  # Bonus for reversal
        if flags['reversal_bearish']:
            sell_score += 1.2    
        if flags['reversal_bearish'] and flags['candlestick_sell']:
            sell_score += 0.8  # small synergy bonus
        if flags['reversal_bullish'] and flags['candlestick_buy']:
             buy_score += 0.8

        
        # Buy signal requirements
        if (buy_score >= self.score_threshold * self.max_possible_score and 
            flags['adx_strong'] and 
            trend_confirms_buy >= 2 and
            flags['volume_ok'] and
            self.get_optimal_entry(window)):
            
            signal = "BUY"
            score = buy_score
            sl = self.calculate_stop_loss(latest['close'], latest['atr'], flags['trend_strength'], "BUY")
            tp = self.calculate_take_profit(latest['close'], latest['atr'], flags['trend_strength'], "BUY")

            
        # Sell signal requirements
        elif (sell_score >= self.score_threshold * self.max_possible_score and 
            flags['adx_strong'] and 
            trend_confirms_sell >= 2 and
            flags['volume_ok'] and
            self.get_optimal_entry(window)):
            
            signal = "SELL"
            score = sell_score
            sl = self.calculate_stop_loss(latest['close'], latest['atr'], flags['trend_strength'], "SELL")
            tp = self.calculate_take_profit(latest['close'], latest['atr'], flags['trend_strength'], "SELL")

        else:
            sl, tp = None, None

        # Debug output
        if self.debug and signal != "HOLD":
            print("\n🧠 DEBUG: Signal Generation Details")
            print(f" - Threshold Score: {self.score_threshold * self.max_possible_score:.2f}")
            print(f" - Max Score Possible: {self.max_possible_score}")
            print(f" - BUY Score: {buy_score:.2f} | SELL Score: {sell_score:.2f}")
            print(f" - ADX: {flags['adx_strong']} | Trend Confirms: BUY={trend_confirms_buy}, SELL={trend_confirms_sell}")
            print(f" - Volume OK: {flags['volume_ok']} | Volatility OK: {flags['volatility_ok']}")
            print(f" - Optimal Time: {flags['optimal_time']} | Liquidity: {flags['sufficient_liquidity']}")
            print(f" - Final Confidence: {confidence}%")
            print(f" - SL: {sl if sl else 'N/A'} | TP: {tp if tp else 'N/A'}")
            
            if signal == "BUY":
                print("\n🟩 BUY Signal Triggers:")
                for key, val in flags.items():
                    if val and key not in ['candlestick_pattern'] and 'buy' in key:
                        print(f"  ✅ {key}")
            elif signal == "SELL":
                print("\n🟥 SELL Signal Triggers:")
                for key, val in flags.items():
                    if val and key not in ['candlestick_pattern'] and 'sell' in key:
                        print(f"  ✅ {key}")

        if signal != "HOLD":
            self.daily_trade_count += 1
        
    

        return signal, score, confidence, sl, tp

    def _process_timeframe(self, label, interval):
        """Fetch data and calculate signal for a timeframe"""
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=interval, limit=200)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            

            df = df.astype({
                'open': 'float', 'high': 'float', 'low': 'float',
                'close': 'float', 'volume': 'float'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            df = self.calculate_indicators(df)

            if len(df) < self.WINDOW_SIZE:
                return None, None

            window = df[-self.WINDOW_SIZE:]
            signal, score, confidence, sl, tp = self.get_signal(window)
            price = df['close'].iloc[-1]

            return {
                "signal": signal,
                "score": score,
                "confidence": confidence,
                "price": price,
                "sl": sl,
                "tp": tp,
                "timeframe": label
            }, df

        except Exception as e:
            if self.debug:
                print(f"Error processing {label} timeframe: {e}")
            return None, None

    def _save_signals(self, signals):
        """Save valid signals to JSON if they meet the threshold"""
        filtered = {tf: s for tf, s in signals.items() 
                   if s["signal"] in ['BUY', 'SELL'] and 
                   s["score"] >= self.score_threshold * self.max_possible_score}
        
        if not filtered:
            return

        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": self.symbol,
            "signals": filtered
        }

        Path(self.SIGNAL_FILE).write_text(json.dumps(output, indent=4))
        if self.debug:
            print(f"✅ Signals saved to {self.SIGNAL_FILE}")

    def _analyze_trends(self, signals):
        """Enhanced trend analysis with timeframe alignment"""
        short_term = ['15m', '30m']
        long_term = ['1h', '4h']
        
        short_signals = [signals[tf]["signal"] for tf in short_term 
                        if tf in signals and signals[tf]["signal"] != "HOLD"]
        long_signals = [signals[tf]["signal"] for tf in long_term 
                       if tf in signals and signals[tf]["signal"] != "HOLD"]
        
        if not short_signals or not long_signals:
            return
            
        short_consensus = max(set(short_signals), key=short_signals.count)
        long_consensus = max(set(long_signals), key=long_signals.count)
        
        if short_consensus == long_consensus:
            print(f"\n✅ Strong {short_consensus} consensus across timeframes")
            # Boost confidence for aligned signals
            for tf in signals:
                if signals[tf]["signal"] == short_consensus:
                    signals[tf]["confidence"] = min(100, signals[tf]["confidence"] + 12)
        else:
            print(f"\n⚠️ Trend Conflict: Short-term {short_consensus}, Long-term {long_consensus}")
            # Penalize conflicting signals
            for tf in signals:
                if tf in short_term and signals[tf]["signal"] != "HOLD":
                    signals[tf]["confidence"] = max(0, signals[tf]["confidence"] - 25)
                    if self.debug:
                        print(f"   Reduced confidence for {tf} due to trend conflict")

    def analyze(self):
        """Main analysis with multi-timeframe confirmation"""
        signals = {}
        
        for label, interval in self.TIMEFRAMES.items():
            signal_data, df = self._process_timeframe(label, interval)
            if signal_data and df is not None:
                signals[label] = signal_data
                self.indicator_data[label] = df
        
        if signals:
            self._analyze_trends(signals)
            self._save_signals(signals)
        
        return signals, self.indicator_data

    def log_trade(self, entry_price, exit_price, direction, timeframe, profit_pct):
        """Log trade details for performance analysis"""
        trade = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": self.symbol,
            "direction": direction,
            "timeframe": timeframe,
            "entry": entry_price,
            "exit": exit_price,
            "profit_pct": profit_pct,
            "balance": self.current_balance
        }
        self.trade_journal.append(trade)
        
        # Update performance tracker
        self.performance_tracker.add_trade("WIN" if profit_pct > 0 else "LOSS")
        
        # Save to file monthly
        if len(self.trade_journal) % 30 == 0:
            with open(f"trades_{datetime.utcnow().strftime('%Y%m')}.json", "w") as f:
                json.dump(self.trade_journal, f)

class PerformanceTracker:
    """Track trading performance metrics"""
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.trade_results = []
    
    def add_trade(self, result):
        """Add trade result (WIN/LOSS)"""
        self.trade_results.append(result)
        if len(self.trade_results) > self.window_size:
            self.trade_results.pop(0)
    
    def get_win_rate(self):
        """Calculate current win rate"""
        if not self.trade_results:
            return 0
        return sum(1 for r in self.trade_results if r == "WIN") / len(self.trade_results)
    
    def get_profit_factor(self):
        """Calculate profit factor (gross profit/gross loss)"""
        wins = [1 for r in self.trade_results if r == "WIN"]
        losses = [1 for r in self.trade_results if r == "LOSS"]
        if not losses:
            return float('inf')
        return sum(wins) / sum(losses)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Trading System...")
    generator = EnhancedSignalGenerator(
        symbol='BTCUSDT',
        score_threshold=0.45,
        debug=True,
        window_size=5
    )
    
    print("\n🔍 Analyzing market conditions across timeframes...")
    signals, indicator_data = generator.analyze()
    
    if signals:
        print("\n📈 Generated Signals:")
        for tf, data in signals.items():
            if data['signal'] in ['BUY', 'SELL']:
                print(f" - {tf}: {data['signal']} at {data['price']:.2f} "
                      f"(Score: {data['score']:.1f}/{generator.max_possible_score}, "
                      f"Confidence: {data['confidence']}%)")
                print(f"   SL: {data['sl']:.2f} | TP: {data['tp']:.2f}")
    else:
        print("\n🟨 No high-probability signals found - waiting for better market conditions")
