import csv

import requests
import numpy as np
import pandas as pd

# ======================================
# CONFIG
# ======================================
API_BASE_URL = "https://api.marketdata.app/v1"
API_KEY = "YWZ4ZWJVLWNrYlQyZGpYUnpLYUtXZjlhY3ptei1zcjNsbk5fMDg5NXdwZz0"
TICKERS = ["NVDA"]


FROM_DATE = "2024-11-30"
TO_DATE   = "2025-11-21"


# ======================================
# Fetch candles from MarketData.app
# ======================================
def get_mdapp_candles(symbol, resolution="D", from_date=None, to_date=None):
    url = f"{API_BASE_URL}/stocks/candles/{resolution}/{symbol}/"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"format": "json"}

    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    r = requests.get(url, headers=headers, params=params)

    if r.status_code != 200:
        print(f"Error: {symbol} → {r.status_code} {r.text}")
        return None

    return r.json()


# ======================================
# Build DataFrame + indicators
# ======================================
def build_df(data_json):
    df = pd.DataFrame({
        "open": data_json["o"],
        "high": data_json["h"],
        "low": data_json["l"],
        "close": data_json["c"],
        "volume": data_json["v"],
        "t": data_json["t"]
    })

    df = df.sort_values("t").reset_index(drop=True)

    # Indicators
    df['avg_volume'] = df['volume'].rolling(20).mean()
    df['adr'] = df['high'] - df['low']
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()

    # VWAP
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df.fillna(0, inplace=True)
    return df


# ======================================
# Convert df → extras_np (12 columns)
# ======================================
def make_extras_np(df):
    arr = np.column_stack([
        df['open'].to_numpy(),
        df['high'].to_numpy(),
        df['low'].to_numpy(),
        df['close'].to_numpy(),
        df['volume'].to_numpy(),
        df['avg_volume'].to_numpy(),
        df['adr'].to_numpy(),
        df['vwap'].to_numpy(),
        df['ma10'].to_numpy(),
        df['ma20'].to_numpy(),
        df['ma50'].to_numpy(),
        df['rsi'].to_numpy()
    ])
    return arr


# ======================================
# Fair Value Gap Detector
# ======================================
def find_fvg(stocks_data):
    """
    stocks_data: list of (symbol, numpy array)
    array columns:
       0=open, 1=high, 2=low, 3=close, 4=volume, 5=avg_volume
    """

    gaps = {}
    
    for symbol, data in stocks_data:

        gaps[symbol] = []
        
        for i in range(len(data) - 2):
            first_h = data[i][1]
            first_l = data[i][2]
            second_v = data[i+1][4]
            second_avg_v = data[i+1][5]
            third_h = data[i+2][1]
            third_l = data[i+2][2]

            # Require 2nd candle volume ≥ 2× avg volume
            if second_avg_v > 0:
                if second_v >= 2 * second_avg_v:
    
                    # Bullish FVG
                    if first_h < third_l:
                        gaps[symbol].append({
                            "index": i+2,        # third candle index
                            "threshold": third_l, # bullish threshold
                            "type": "bullish"
                        })
                        print(gaps)
                        print(f"{symbol} | Candle {i+2} | 🔵 Bullish FVG "
                              f"(vol {second_v:.0f} ≥ 2× avg {second_avg_v:.0f})")
    
                    # Bearish FVG
                    elif first_l > third_h:
                        gaps[symbol].append({
                            "index": i+2,        # third candle index
                            "threshold": third_l, # bullish threshold
                            "type": "bearish"
                        })
                        #print(gaps)
                        print(f"{symbol} | Candle {i+2} | 🔴 Bearish FVG "
                              f"(vol {second_v:.0f} ≥ 2× avg {second_avg_v:.0f})")

    return gaps


# ======================================
# MAIN EXECUTION
# ======================================
all_stocks_list = []

for symbol in TICKERS:
    print(f"\n=== Fetching {symbol} ===")
    raw = get_mdapp_candles(symbol, from_date=FROM_DATE, to_date=TO_DATE)

    if raw is None:
        continue

    df = build_df(raw)
    extras_np = make_extras_np(df)

    # Only OHLCV + avg volume needed for FVG logic
    ohlc_for_fvg = extras_np[:, :6]

    all_stocks_list.append((symbol, extras_np))

# Run FVG detector
positions = find_fvg(all_stocks_list)


import requests
import numpy as np
import pandas as pd
import json

# -----------------------------
# Configuration
# -----------------------------
API_BASE_URL = "https://api.marketdata.app/v1"
API_KEY = "YWZ4ZWJVLWNrYlQyZGpYUnpLYUtXZjlhY3ptei1zcjNsbk5fMDg5NXdwZz0"  # Optional, if you have a token
STOCK_SYMBOL = TICKERS

# -----------------------------
# Fetch historical daily candles
# -----------------------------
def get_historical_candles(symbol, resolution="D", from_date=None, to_date=None):
    url = f"{API_BASE_URL}/stocks/candles/{resolution}/{symbol}/"
    params = {
        "format": "json"
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching historical candles: {response.status_code} {response.text}")
        return None


def check_hits(window, fvg_type, threshold):
    """
    Returns the index (relative to window start) where the price touches the FVG threshold.
    Bullish: low <= threshold
    Bearish: high >= threshold
    """
    if fvg_type == "bullish":
        for i in range(len(window)):
            if window[i][2] < threshold:  # low <= threshold
                #print("LOW", window[i][2])
                #print("WINNY", window)
                return i
    elif fvg_type == "bearish":
        for i in range(len(window)):
            if window[i][1] > threshold:  # high >= threshold
                #print("HIGH", window[i][1])
                #print("WINNY", window)
                return i
    return None  # not touched within window


def get_fvg_candle_windows(extras_np, fvg_list, pre=20, post=50):
    """
    extras_np: full OHLCV+indicators array
    fvg_list: list of dicts like [{'index':33,'threshold':174.5,'type':'bullish'}, ...]
    
    Returns:
    List of dicts for each FVG containing:
      - window (np.array)
      - fvg_index
      - type
      - threshold
      - first_touch_index (relative to window start)
    """
    num_candles = extras_np.shape[0]
    results = []

    for fvg in fvg_list:
        idx = fvg["index"]
        fvg_type = fvg["type"]
        threshold = fvg["threshold"]

        start_idx = max(0, idx - pre)
        end_idx = min(num_candles, idx + post + 1)
        window = extras_np[start_idx:end_idx, :]

        print(fvg)
        #print("EEEEEEE",window[11:])
        #print("EEEEEEE",windows[:1][0][20:])
        first_touch = check_hits(window[11:], fvg_type, threshold)

        results.append({
            "window": window,
            "fvg_index": idx,
            "type": fvg_type,
            "threshold": threshold,
            "first_touch_index": first_touch
        })

    return results

# -----------------------------
# Process data into NumPy arrays and indicators
# -----------------------------

ticker = {}
for i in range(len(STOCK_SYMBOL)):
    
    dat = get_historical_candles(STOCK_SYMBOL[i], from_date=FROM_DATE, to_date=TO_DATE)
    
    # your MarketData.app JSON
    data_json = dat  # <-- replace with your JSON
    
    # Build DataFrame directly from the lists
    df = pd.DataFrame({
        "open": data_json["o"],
        "high": data_json["h"],
        "low": data_json["l"],
        "close": data_json["c"],
        "volume": data_json["v"]
    })
    
    # Ensure ascending order by timestamp
    df['t'] = data_json["t"]
    df = df.sort_values('t').reset_index(drop=True)
    
    # -----------------------------
    # Calculate indicators
    # -----------------------------
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['adr'] = df['high'] - df['low']
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # VWAP
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
    
    # RSI (14-day)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Replace NaNs with zeros
    df.fillna(0, inplace=True)
    
    # -----------------------------
    # Convert to NumPy arrays
    # -----------------------------
    opens = df['open'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    closes = df['close'].to_numpy()
    volumes = df['volume'].to_numpy()
    adr = df['adr'].to_numpy()
    vwap = df['vwap'].to_numpy()
    avg_volume = df['avg_volume'].to_numpy()
    ma10 = df['ma10'].to_numpy()
    ma20 = df['ma20'].to_numpy()
    ma50 = df['ma50'].to_numpy()
    rsi = df['rsi'].to_numpy()
    
    # -----------------------------
    # Extras array (all indicators + OHLCV)
    # -----------------------------
    extras_np = np.column_stack([
        opens, highs, lows, closes, volumes,
        avg_volume, adr, vwap, ma10, ma20, ma50, rsi
    ])
    
    print("Extras array shape:", extras_np.shape)  # (num_days, 12)
    print(extras_np)  # first 5 rows
    
    symbol = "AAPL"
    extras_np = all_stocks_list[[s for s, _ in all_stocks_list].index(STOCK_SYMBOL[i])][1]  # full data MAYBE DON'T USE STOCK_SYMBOL[i]
    fvg_list = positions[STOCK_SYMBOL[i]]  # [{'index':..,'threshold':..,'type':..}, ...]
    
    fvg_windows = get_fvg_candle_windows(extras_np, fvg_list, pre=10, post=50)
    print(fvg_windows)
    for fvg in fvg_windows:
        print(f"FVG candle: {fvg['fvg_index']} | Type: {fvg['type']} | Threshold: {fvg['threshold']}")
        print(f"First touch at relative candle: {fvg['first_touch_index']}")
        print("Window shape:", fvg['window'].shape)

    ticker[STOCK_SYMBOL[i]] = fvg_windows
    print(ticker)

print(ticker)



def convert_to_percent(fvg_window):
    #fvg_window = np.array(fvg_window, dtype=float)
    percentage_fvg_window = np.zeros_like(fvg_window)
    
    for i in range(fvg_window.shape[1]):
        for j in range(12):
            if j == 5 or j == 11:
                percentage_fvg_window[i][j] = fvg_window[i][j]
            elif i == 0:
                percentage_fvg_window[i][j] = 1
            else:
                percentage_fvg_window[i][j] = fvg_window[i][j] / fvg_window[i-1][j]
                
    return percentage_fvg_window



def export_fvg_to_csv(fvg_windows, tick, filename="data.csv"):
    """
    fvg_windows: list of dicts from get_fvg_candle_windows
    Each row in CSV will contain:
      - pre-FVG candles (flattened)
      - post-FVG candles up to first touch (flattened)
      - type, threshold, rebound (0/1), sector (placeholder)
    """
    all_bulls = []
    all_bears = []
    for fvg in fvg_windows:
        print("FIRSTONE", fvg)
        fv = convert_to_percent(fvg["window"])
        fvg["window"] = fv
        print("SECONDONE", fvg)
        window = fvg["window"]
        print("A",window)
        fvg_idx = fvg["fvg_index"]
        print("B",fvg_idx)
        if fvg["type"] == "bullish":
            fvg_type = 1
        else:
            fvg_type = 0
        print("C",fvg_type)
        threshold = fvg["threshold"]
        print("D",threshold)
        first_touch_idx = fvg["first_touch_index"]
        print("E",first_touch_idx)

        touch = 0
        
        # 10 candles before FVG (first 10 in window slice)
        pre_fvg_data = window[:10, :].flatten()  # 10 candles * 12 columns = 120 elements

        # Data from FVG candle until first touch (if first_touch_index exists)
        if first_touch_idx is not None:
            touch = 1
            post_fvg_data = window[10:10+first_touch_idx+1, :].flatten()
        else:
            touch = 0
            post_fvg_data = np.array([])  # No touch

        row = [str(fvg_idx), tick]
        
        # Flatten everything into a single list
        row += list(pre_fvg_data) + list(post_fvg_data)

        # Add metadata placeholders
        row += [str(threshold), str(first_touch_idx), str(touch)]

        if fvg_type == 1:
            all_bulls.append(row)
        else:
            all_bears.append(row)

    # Write to CSV
    with open("bulls.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for row in all_bulls:
            writer.writerow(row)

    with open("bears.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for row in all_bears:
            writer.writerow(row)


# =====================
# Example usage
# =====================
row = ["Ticker"]
row += [""] * 705  # 700 empty columns

import csv

with open("bulls.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(row)

with open("bears.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(row)

for i in range(len(ticker)):
    export_fvg_to_csv(ticker[STOCK_SYMBOL[i]], STOCK_SYMBOL[i], filename="data.csv")