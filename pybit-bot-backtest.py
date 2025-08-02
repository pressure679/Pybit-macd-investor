import time
import math
from datetime import datetime
# import requests
import pandas as pd
# from pybit.unified_trading import HTTP  # pip install pybit
import numpy as np
from io import StringIO
import threading
import os

# === SETUP ===
symbols = ["BTCUSD", "BNBUSD", "ETHUSD", "XRPUSD", "XAUUSD"]
# balance = 50
# leverage = 75
# risk_pct = 0.3
interval_minutes = 1440  # 1 day, can be 10080 (week), 43200 (month)
balance = 1000

def load_last_mb(symbol, filepath="/mnt/chromeos/removable/sd_card/1m dataframes", mb_size=6*3):
    # Search for a file containing the symbol in its name
    matching_files = [f for f in os.listdir(filepath) if symbol.lower() in f.lower()]
    if not matching_files:
        raise FileNotFoundError(f"No file containing '{symbol}' found in {filepath}")
    
    # Use the first matching file
    fp = os.path.join(filepath, matching_files[0])
    bytes_to_read = mb_size * 1024 * 1024

    with open(fp, "rb") as f:
        f.seek(0, os.SEEK_END)
        start = max(0, f.tell() - bytes_to_read)
        f.seek(start)
        data = f.read().decode("utf-8", errors="ignore")

    lines = data.split("\n")[1:] if start else data.split("\n")
    df = pd.read_csv(StringIO("\n".join([l for l in lines if l.strip()])), header=None)
    # clean_lines = [l for l in lines if l.count(",") >= 11]  # crude validation for 12-column lines
    # csv_string = "\n".join(clean_lines)
    # df = pd.read_csv(StringIO(csv_string), header=None)
    
    df.columns = [
        "Open time","Open","High","Low","Close","Volume","Close time",
        "Quote asset vol","Trades","Taker buy base","Taker buy quote","Ignore"
    ]

    # Convert 'Open time' to datetime and set as index
    df['Open time'] = pd.to_datetime(df['Open time'])
    df.set_index('Open time', inplace=True)

    # Keep only OHLC columns
    df = df[['Open', 'High', 'Low', 'Close']].copy()

    # df = df.resample('15min').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last'
    # }).dropna()

    return df

def load_last_mb_xauusd(file_path="/mnt/chromeos/removable/sd_card/1m dataframes/XAUUSD_1m_data.csv", mb=2*3, delimiter=';', col_names=None):
    file_size = os.path.getsize(file_path)
    offset = max(file_size - mb * 1024 * 1024, 0)  # start position
    
    with open(file_path, 'rb') as f:
        # Seek to approximately 20 MB before EOF
        f.seek(offset)
        
        # Read to the end of file from that offset
        data = f.read().decode(errors='ignore')
        
        # If not at start of file, discard partial first line (incomplete)
        if offset > 0:
            data = data.split('\n', 1)[-1]
        
    df = pd.read_csv(StringIO(data), delimiter=delimiter, header=None, engine='python')
    
    #if col_names:
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    
    # Convert columns if needed, e.g.:
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M", errors='coerce')
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].copy()

    # df = df.resample('15min').agg({
    #     'Open': 'first',
    #     'High': 'max',
    #     'Low': 'min',
    #     'Close': 'last'
    # }).dropna()
    
    df = df.dropna()
    
    return df

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def RSI(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    # red_line = ema_fast - ema_slow
    return macd_line, signal_line, histogram

def Bollinger_Bands(series, period=20, num_std=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band

def ADX(df, period=14):
    """
    Returns +DI, -DI and ADX using Wilder's smoothing.
    Columns required: High, Low, Close
    """
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # --- directional movement -----------------------------------------
    # plus_dm  = (high.diff()  > low.diff())  * (high.diff()).clip(lower=0)
    # minus_dm = (low.diff()   > high.diff()) * (low.diff().abs()).clip(lower=0)̈́
    up  =  high.diff()
    dn  = -low.diff()

    plus_dm_array  = np.where((up  >  dn) & (up  > 0),  up,  0.0)
    minus_dm_array = np.where((dn  >  up) & (dn  > 0),  dn,  0.0)

    plus_dm = pd.Series(plus_dm_array, index=df.index) # ← wrap
    minus_dm = pd.Series(minus_dm_array, index=df.index) # ← wrap

    # --- true range ----------------------------------------------------
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    # --- Wilder smoothing ---------------------------------------------
    atr       = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di   = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di  = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di

def BullsPower(df, period=13):
    ema = EMA(df['Close'], period)
    return df['High'] - ema

def BearsPower(df, period=13):
    ema = EMA(df['Close'], period)
    return df['Low'] - ema

def Momentum(series, period=10):
    return series - series.shift(period)

# df['SAR'].iloc[n] returns sar values for n candle
def SAR(df: pd.DataFrame,
        step: float = 0.02,
        max_step: float = 0.2) -> pd.DataFrame:
    """
    Adds column 'sar' (Parabolic SAR) to df and returns df.

    Parameters
    ----------
    step : float
        AF increment (default 0.02)
    max_step : float
        Maximum AF (default 0.2)
    """
    high, low = df['High'].values, df['Low'].values
    n = len(df)
    sar = np.zeros(n)

    # Initialisation
    trend = 1                    # 1 = up, -1 = down
    sar[0] = low[0]              # seed with first low
    ep = high[0]                 # extreme point
    af = step

    for i in range(1, n):
        # 1) tentative SAR
        sar[i] = sar[i-1] + af * (ep - sar[i-1])

        # 2) keep SAR on the correct side of price
        if trend == 1:
            sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else sar[i])
        else:
            sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else sar[i])

        # 3) trend‑flip checks
        if trend == 1:
            if low[i] < sar[i]:                 # bullish → bearish flip
                trend = -1
                sar[i] = ep                     # reset SAR to last EP
                ep = low[i]
                af = step
            else:                               # still bullish
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            if high[i] > sar[i]:                # bearish → bullish flip
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = step
            else:                               # still bearish
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)

    df['SAR'] = sar
    return df
# df['Fractal_High].iloc[n] and df['Fractal_Low].iloc[n] return true if candle if a fractal high or low.
def Fractals(df: pd.DataFrame,
             window: int = 2) -> pd.DataFrame:
    """
    Adds columns 'fractal_high' and 'fractal_low' to df.
    A 5‑bar fractal uses window=2 (2 bars on each side of the pivot).

    Parameters
    ----------
    window : int
        Half‑window size. 2 → 5‑bar, 3 → 7‑bar, etc.
    """
    h, l = df['High'], df['Low']
    w = window

    high_mask = (
        (h.shift(w) > h.shift(w+1)) &
        (h.shift(w) > h.shift(w+2)) &
        (h.shift(w) > h.shift(w-1)) &
        (h.shift(w) > h)
    )

    low_mask = (
        (l.shift(w) < l.shift(w+1)) &
        (l.shift(w) < l.shift(w+2)) &
        (l.shift(w) < l.shift(w-1)) &
        (l.shift(w) < l)
    )

    # df['Fractal_High'] = high_mask.shift(-w).fillna(False).infer_objects(copy=False).astype(bool)
    # df['Fractal_Low']  = low_mask.shift(-w).fillna(False).infer_objects(copy=False).astype(bool)
    # df['Fractal_High'] = high_mask.shift(-w).astype('boolean').fillna(False).astype(bool)
    # df['Fractal_Low']  = low_mask.shift(-w).astype('boolean').fillna(False).astype(bool)
    df['Fractal_High'] = (df['High'] == df['High'].rolling(window=2*w+1, center=True).max())
    df['Fractal_Low'] = (df['Low'] == df['Low'].rolling(window=2*w+1, center=True).min())
    
    return df

def add_indicators(df):
    df['EMA_7'] = df['Close'].ewm(span=7).mean()
    df['EMA_14'] = df['Close'].ewm(span=14).mean()
    df['EMA_28'] = df['Close'].ewm(span=28).mean()
    df['EMA_7_Diff'] = df['EMA_7'].diff()
    df['EMA_14_Diff'] = df['EMA_14'].diff()
    df['EMA_28_Diff'] = df['EMA_28'].diff()

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    df['macd_line'], df['macd_signal'], df['macd_histogram'] = MACD(df['Close'])
    # === MACD Crossovers ===
    df['macd_cross_up'] = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
    df['macd_signal_cross_up'] = (df['macd_signal'] > 0) & (df['macd_signal'].shift(1) <= 0)
    df['macd_signal_cross_down'] = (df['macd_signal'] < 0) & (df['macd_signal'].shift(1) >= 0)
    df['macd_signal_diff'] = df['macd_signal'].diff()

    # === MACD Trend Status ===
    df['macd_trending_up'] = df['macd_line'] > df['macd_signal']
    df['macd_trending_down'] = df['macd_line'] < df['macd_signal']
    df['macd_histogram_increasing'] = df['macd_histogram'].diff() > 0
    df['macd_histogram_decreasing'] = df['macd_histogram'].diff() < 0

    df['bb_upper'], df['bb_lower'] = Bollinger_Bands(df['Close'])

    # df['Momentum'] = df['Close'] - df['Close'].shift(10)
    # Custom momentum (% of recent high-low range)
    high_14 = df['High'].rolling(window=14).max()
    low_14 = df['Low'].rolling(window=14).min()
    price_range = high_14 - low_14
    df['Momentum'] = 100 * (df['Close'] - df['Close'].shift(14)) / price_range
    # Momentum trend signals: compare current Momentum with previous
    df['Momentum_increasing'] = df['Momentum'] > df['Momentum'].shift(2)
    df['Momentum_decreasing'] = df['Momentum'] < df['Momentum'].shift(2)

    df['RSI'] = RSI(df['Close'], 14)
    df['ADX'], df['+DI'], df['-DI'] = ADX(df)
    df['Di_Diff'] = (df['+DI'] - df['-DI']).abs()

    # df['Bulls'] = BullsPower(df)     # High – EMA(close, 13)
    # df['Bears'] = BearsPower(df)     # Low  – EMA(close, 13)
    # df['Bulls'] = df['High'] - df['Close']
    # df['Bears'] = df['Close'] - df['Low']
    # df['Bullish_DI'] = df['Bulls'] - df['Bears']
    # df['Bullish_DI'] = df['+DI'] - df['-DI']
    # df['Bearish_DI'] = df['-DI'] - df['+DI']
    # df['Bull_Bear_Diff'] = (df['Bulls'] - df['Bears']) / df['ATR']
    # df['Bull_Bear_Diff'] = (df['Bulls'] - df['Bears'])

    df['OSMA'] = df['macd_line'] - df['macd_signal']
    df['OSMA_Diff'] = df['OSMA'].diff()

    # df = SAR(df)

    # df = Fractals(df)

    # df[]

    df.dropna(inplace=True)
    return df
def generate_signals(df):
    """
    Return a Series of 'Buy', 'Sell', or '' using
        • flat‑market veto             (10‑bar High/Low range)
        • ADX strength > 20
        • MACD‑angle filter            (current slope > 5‑bar avg slope)
        • MACD hook skip               (no trade immediately after peak/valley)
        • DI / Bull‑Bear / OSMA logic  (your original direction rules)
    """

    signals = [""] * len(df)

    # --- pre‑compute helpers once ------------------------------------
    macd_slope      = df['macd_line'].diff()
    macd_slope_ma_5 = macd_slope.rolling(5).mean()

    # “hook” is True on the bar *right after* MACD slope changes sign
    macd_hook = (
        (macd_slope.shift(1) > 0) & (macd_slope <= 0) |   # bullish peak
        (macd_slope.shift(1) < 0) & (macd_slope >= 0)     # bearish valley
    )

    for i in range(len(df)):
        if i < 10:        # need at least 10 bars for range filter
            continue

        latest = df.iloc[i]

        # -------------- flat‑market veto ------------------------------
        # window = df.iloc[i - 10 : i]
        # price_range = window["High"].max() - window["Low"].min()
        # if (price_range / window["Close"].mean()) <= 0.005:
        #     continue

        # -------------- trend strength check --------------------------
        if latest.ADX < 20:
            continue

        # -------------- MACD angle filter -----------------------------
        angle_ok = abs(macd_slope.iloc[i]) > abs(macd_slope_ma_5.iloc[i])
        if not angle_ok:
            continue

        # -------------- hook (peak/valley) skip -----------------------
        if macd_hook.iloc[i]:
            continue

        if df['macd_signal_cross_up'].iloc[i]:
           signals[i] = "Buy"
        elif df['macd_signal_cross_down'].iloc[i]:
            signals[i] = "Sell"

        # -------------- directional logic -----------------------------
        # if latest.macd_signal_diff > 0:
        #     if latest.macd_trending_up:
       #         signals[i] = "Buy"
        #     else:
        #         signals[i] = "Close"
        # if (latest.macd_signal_diff > 0 and
        #     latest['+DI'] > latest['-DI'] and
        #     latest.Bull_Bear_Diff > 0 and
        #     latest.OSMA_Diff > 0):
        #     signals[i] = "Buy"

        # elif (latest.macd_signal_diff < 0 and
        #       latest['+DI'] < latest['-DI'] and
        #       latest.Bull_Bear_Diff < 0 and
        #       latest.OSMA_Diff < 0):
        #     signals[i] = "Sell"
        # elif latest.macd_signal_diff < 0:
        #     if latest.macd_trending_down:
        #         signals[i] = "Sell"
        #     else:
        #         signals[i] = "Close"

    df["signal"] = signals
    return df["signal"]

def calculate_position_size(balance, risk_pct, entry_price, stop_loss, leverage, min_qty=0.001):
    risk_amount = max(balance * risk_pct, 50)
    position_size = risk_amount / abs(entry_price - stop_loss) * leverage
    # position_size = round(position_size, 3)

    if position_size < min_qty:
        position_size = min_qty

    return position_size

# ------------------------------------------------------------------
# 1) position‑sizing + classic ATR‑fib TP levels
# ------------------------------------------------------------------
def calculate_trade_parameters(entry_price, atr, balance,
                               side, leverage=50, risk_pct=0.05):
    """
    • Risk   : risk_pct of balance (margin) × leverage
    • SL     : 1.5 × ATR
    • TP     : 3 × ATR × Fibonacci([0.236,0.382,0.5,0.618])
    """

    fib_levels = [0.382, 0.618, 0.786]

    # position size
    margin          = balance * risk_pct
    position_value  = margin * leverage
    position_size   = position_value / entry_price

    # stop‑loss
    sl_dist  = 1.5 * atr
    stop_loss = (entry_price - sl_dist) if side.lower() == "buy" \
                else (entry_price + sl_dist)

    # position_size = calculate_position_size(balance, 0.05, entry_price, stop_loss, 50)

    # take‑profits
    base = 4.5 * atr
    tp_levels = [ (entry_price + base * f)  if side.lower() == "buy"
                  else (entry_price - base * f)
                  for f in fib_levels ]

    return {
        "position_size": round(position_size, 6),
        "stop_loss"    : round(stop_loss, 6),
        "tp_levels"    : [round(tp, 6) for tp in tp_levels]
    }

# ------------------------------------------------------------------
# 2) updater that supports 3 partial TP hits (40/20/20)
# ------------------------------------------------------------------
def update_trade(trade, current_price, atr, df_latest):
    if trade["status"] != "open":
        return trade, 0.0

    qty        = trade["qty"]
    entry      = trade["entry"]
    side_buy   = trade["side"].lower() == "buy"
    sl         = trade["sl"]
    tp_levels  = trade["tp_levels"]
    tp_hits    = trade.get("tp_hits", [False, False, False])
    realized   = 0.0

    # ----- stop‑loss --------------------------------------------------
    if side_buy and current_price <= sl:
        pnl = (sl - entry) * qty
        trade["status"] = "closed"; trade["pnl"] += pnl
        return trade, pnl
    elif (not side_buy) and current_price >= sl:
        pnl = (entry - sl) * qty
        trade["status"] = "closed"; trade["pnl"] += pnl
        return trade, pnl

    # ----- partial TPs (40/20/20/20) ---------------------------------
    portions = [0.4, 0.2, 0.2, 20]

    for i, (tp, hit) in enumerate(zip(tp_levels, tp_hits)):
        if hit:
            continue
        if (side_buy  and current_price >= tp) or \
           ((not side_buy) and current_price <= tp):
            part_qty = qty * portions[i]
            pnl = (tp - entry) * part_qty if side_buy \
                  else (entry - tp) * part_qty
            realized        += pnl
            trade["pnl"]    += pnl
            tp_hits[i]       = True

    trade["tp_hits"] = tp_hits

    # close trade if all targets hit
    if all(tp_hits):
        trade["status"] = "closed"

    return trade, realized

def run_bot(symbol):
    global balance
    df = None
    if symbol == "XAUUSD":
        df = load_last_mb_xauusd()
    else:
        df = load_last_mb(symbol)
    # df['Open time'] = pd.to_datetime(df['Open time'])
    # df.set_index('Open time', inplace=True)
    # df = df.dropna()
    df = add_indicators(df)
    df['signal'] = generate_signals(df)
    # print(df.columns)
    # balance = 1000
    risk_pct = 0.05
    leverage = 50

    trade_results = []
    total_trades = 0
    num_active_trades = 0
    active_trade = None
    investment = 0.0
    index_placed_order = 0

    for i in range(len(df)):
        if balance < 10:
            break
        if (i + 1) % 1440 == 0:
            if total_trades > 0:
                wins = [p for p in trade_results if p > 0]
                losses = [p for p in trade_results if p <= 0]
                win_rate = len(wins) / total_trades * 100
                avg_profit = sum(wins) / len(wins) if wins else 0
                avg_loss = sum(losses) / len(losses) if losses else 0
                
                print(f"[{symbol}] Stats at day {((i + 1) / 1440):.0f}:")
                print(f"  Total trades: {total_trades}")
                print(f"  Win rate: {win_rate:.2f}%")
                print(f"  Avg profit: {avg_profit:.2f}")
                print(f"  Avg loss: {avg_loss:.2f}")
                print(f"  Balance: {balance:.2f}")

                # Reset stats for next interval if desired
                trade_results = []
                total_trades = 0

        latest = df.iloc[i]
        signal = latest['signal']
        current_price = latest['Close']
        atr = latest['ATR']

        # Update active trade if any
        if active_trade:
            active_trade, pnl = update_trade(active_trade, current_price, atr, latest)
            if pnl != 0:
                # balance += pnl
                balance += active_trade['pnl']
                trade_results.append(pnl)
                if active_trade["status"] == "closed":
                    active_trade["exit"] = current_price
                    total_trades += 1
                    # print(f"Trade {total_trades} | Side: {active_trade['side'].capitalize()} | Entry: {active_trade['entry']:.2f} | Exit: {active_trade['exit']:.2f} | Size: {active_trade['qty']:.4f} | PnL: {active_trade['pnl']:.2f} | Balance: {balance:.2f} | ATR: {atr}")

                    active_trade = None
                    num_active_trades -= 1

        # Get current signal and mode
        
        # if active_trade and signal != active_trade["side"] and signal != "" or signal == "Close":
        if active_trade is not None and (
                signal == "Close" or
                (signal != "" and signal != active_trade["side"])
        ):
            # Close previous trade forcibly at current price
            active_trade["status"] = "closed"
            active_trade["exit"] = current_price

            active_trade["exit"] = current_price
            active_trade["pnl"] = (current_price - active_trade["entry"]) * active_trade["qty"] - active_trade["qty"] * 0.00075 * 2 - active_trade["qty"] * 0.00025 if active_trade["side"] == "buy" else (active_trade["entry"] - current_price) * active_trade["qty"] - active_trade["qty"] * 0.00075 * 2 - active_trade["qty"] * 0.00025
            # print(f"Trade {total_trades} | Side: {active_trade['side'].capitalize()} | Entry: {active_trade['entry']:.2f} | Exit: {active_trade['exit']:.2f} | Size: {active_trade['qty']:.4f} | PnL: {active_trade['pnl']:.2f} | Balance: {balance:.2f} | ATR: {atr}")
            active_trade["side"] = signal
            balance += active_trade["pnl"]
            trade_results.append(active_trade["pnl"])
            total_trades += 1
            active_trade = None

        # Place new trade if no active trade and valid signal
        # Open a new trade if there's a signal, no active trade, and ATR is valid
        if not active_trade and signal in ["Buy", "Sell"]:
            entry_price = current_price
            trade_params = calculate_trade_parameters(entry_price, atr, balance, signal, leverage, risk_pct)

            # allocated_margin = balance * risk_pct           # capital at risk
            # position_value   = allocated_margin * leverage  # notional
            # position_size    = position_value / entry_price
            
            active_trade = {
                "status": "open",
                "side": signal,
                "entry": entry_price,
                "qty": trade_params["position_size"],
                "sl": trade_params["stop_loss"],
                "tp_levels": trade_params["tp_levels"],
                "tp_hits": [False, False, False],
                "trail_active": False,
                "trail_offset": atr * 1.5,
                "pnl": 0
            }
            num_active_trades += 1

test_threads = []
for symbol in symbols:
    t = threading.Thread(target=run_bot, args=(symbol,))
    t.start()
    test_threads.append(t)
for t in test_threads:
    t.join()
