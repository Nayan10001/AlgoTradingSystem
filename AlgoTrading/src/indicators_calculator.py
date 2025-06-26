import pandas as pd
import numpy as np
import ta
import ta.momentum
from src.logger_setup import setup_logger


logger = setup_logger(__name__)

"""All the factors for the calculation, added 2 more facrtors: bollinger bands, volume based"""
#Using Ta for the Calculation
def calculate_rsi(data_series, period=14):
    """Calculate Relative Strength Index (RSI)"""
    if data_series.empty or data_series.isnull().all() or len(data_series) < period:
        return pd.Series(index=data_series.index, dtype=float)
    return ta.momentum.RSIIndicator(close=data_series, window=period).rsi()

def calculate_sma(data_series, window=20):
    """Calculate Simple Moving Average (SMA)"""
    if data_series.empty or data_series.isnull().all() or len(data_series) < window:
        return pd.Series(index=data_series.index, dtype=float)
    return data_series.rolling(window=window, min_periods=1).mean() #<--calculate the moving mean

def calculate_macd(data_series, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if data_series.empty or data_series.isnull().all() or len(data_series) < slow: 
         return (
            pd.Series(index=data_series.index, dtype=float),
            pd.Series(index=data_series.index, dtype=float),
            pd.Series(index=data_series.index, dtype=float)
        )
    macd_indicator = ta.trend.MACD(close=data_series, 
                                   window_fast=fast, 
                                   window_slow=slow, 
                                   window_sign=signal)
    return macd_indicator.macd(), macd_indicator.macd_signal(), macd_indicator.macd_diff()


def calculate_bollinger_bands(data_series, window=20, std_dev=2):
    """calculate bollinger bands"""
    """ Upper Band: SMA + std_dev × standard deviation
        Middle Band: Simple Moving Average (SMA)
        Lower Band: SMA - std_dev × standard deviation """
    if data_series.empty or data_series.isnull().all() or len(data_series) < window:
        return (pd.Series(index=data_series.index, dtype=float), 
                pd.Series(index=data_series.index, dtype=float), 
                pd.Series(index=data_series.index, dtype=float)
                )
    bb_indicator = ta.volatility.BollingerBands(close=data_series, window=window, window_dev=std_dev, fillna=False)
    return (
        bb_indicator.bollinger_hband(),   # Upper Band
        bb_indicator.bollinger_mavg(),    # Middle Band (SMA)
        bb_indicator.bollinger_lband()    # Lower Band
    )
def calculate_volume_indicators(price_series, volume_series, window_ma=20):
    """Calculate volume-based indicators"""
    results = {}
    if volume_series.empty or volume_series.isnull().all():
        results['volume_ma'] = pd.Series(index=price_series.index, dtype=float)
        results['volume_ratio'] = pd.Series(index=price_series.index, dtype=float)
        results['obv'] = pd.Series(index=price_series.index, dtype=float)
        return results

    results['volume_ma'] = volume_series.rolling(window=window_ma, min_periods=1).mean()
    results['volume_ratio'] = volume_series / (results['volume_ma'] + 1e-9) # Avoid division by zero(1e-9)

    if not price_series.empty and not price_series.isnull().all() and len(price_series) == len(volume_series):
        results['obv'] = ta.volume.OnBalanceVolumeIndicator(close=price_series, volume=volume_series, fillna=False).on_balance_volume()
    else:
        results['obv'] = pd.Series(index=price_series.index, dtype=float)
    return results


def insertion_to_df(df, 
                    rsi_period=14, 
                    sma_short_period=20, 
                    sma_long_period=50,
                    macd_fast=12, 
                    macd_slow=26, 
                    macd_signal=9,
                    bb_window=20, 
                    bb_std_dev=2, 
                    volume_ma_window=20):
    
    """
    Adds indicators to the df
    Requires  columns: 'open', 'high', 'low', 'close', 'adj_close', 'volume'
    """
    if df is None or df.empty:
        logger.warning("Input DataFrame is empty. Cannot calculate indicators.")
        return pd.DataFrame()
    data = df.copy()

    # ----- Determine price and volume series -----
    price_col = 'adj_close' if 'adj_close' in data.columns and not data['adj_close'].isnull().all() else 'close'
    volume_col = 'volume'

    price_series = data.get(price_col, pd.Series(np.nan, index=data.index))
    volume_series = data.get(volume_col, pd.Series(np.nan, index=data.index))

    if price_series.isnull().all():
        logger.error(f"Price column '{price_col}' is missing or all NaNs. Skipping price-based indicators.")
    if volume_series.isnull().all():
        logger.warning(f"Volume column is missing or all NaNs. Volume indicators may be NaN.")

    logger.info(f"Using price column: '{price_col}', volume column: '{volume_col}'")

    try:
        
        data['RSI'] = calculate_rsi(price_series, period=rsi_period)
        data['SMA_short'] = calculate_sma(price_series, window=sma_short_period)
        data['SMA_long'] = calculate_sma(price_series, window=sma_long_period)
        data['MA_ratio'] = data['SMA_short'] / (data['SMA_long'] + 1e-9)

        macd_line, macd_signal_line, macd_hist = calculate_macd(
            price_series, fast=macd_fast, slow=macd_slow, signal=macd_signal
        )
        data['MACD'] = macd_line
        data['MACD_signal'] = macd_signal_line
        data['MACD_hist'] = macd_hist

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(
            price_series, window=bb_window, std_dev=bb_std_dev
        )
        data['BB_upper'] = bb_upper
        data['BB_middle'] = bb_mid
        data['BB_lower'] = bb_lower
        data['BB_position'] = (price_series - bb_lower) / (bb_upper - bb_lower + 1e-9)

        # Volume-Based Indicators 
        volume_indicators = calculate_volume_indicators(
            price_series, volume_series, window_ma=volume_ma_window
        )
        for key, value in volume_indicators.items():
            data[key.upper()] = value  # e.g., 'OBV', 'VOLUME_MA'

        #Returns, Volatility 
        data['PRICE_CHANGE'] = price_series.pct_change()
        data['PRICE_CHANGE_5D'] = price_series.pct_change(periods=5)
        data['VOLATILITY_20D'] = data['PRICE_CHANGE'].rolling(window=20, min_periods=1).std()

        #Support/Resistance 
        data['SUPPORT_20D'] = data['low'].rolling(window=20, min_periods=1).min() if 'low' in data else np.nan
        data['RESISTANCE_20D'] = data['high'].rolling(window=20, min_periods=1).max() if 'high' in data else np.nan

        logger.info(f"Added technical indicators. Final columns: {list(data.columns)}")

    except Exception as e:
        logger.error(f"Error during indicator calculation: {e}", exc_info=True)

    return data
