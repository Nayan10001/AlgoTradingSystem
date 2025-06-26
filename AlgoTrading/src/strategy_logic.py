import pandas as pd
import numpy as np
from src.logger_setup import setup_logger

logger = setup_logger("StrategyLogic")

def generate_strategy_signals(df_with_indicators,
    rsi_oversold_threshold=30,
    rsi_overbought_threshold=70,
    use_bb_for_buy=False,
    use_bb_for_sell=False,
    use_volume_for_buy=False,
    volume_ratio_min_buy=1.0,
    ma_cross_confirms_sell=True,
    price_above_sma_long_for_buy=False
):
    """
    Generates trading signals based on technical indicators.
    Implements the core RSI + MA crossover strategy and allows for additional configurable conditions.
    Returns:
        pd.DataFrame: DataFrame with 'price', 'signal' (1: Buy, -1: Sell, 0: Hold),
                      and 'signal_strength' columns.
    """
    if df_with_indicators is None or df_with_indicators.empty:
        logger.warning("Input DataFrame with indicators is empty. Cannot generate signals.")
        return pd.DataFrame() # Return empty DataFrame if no input

    data = df_with_indicators.copy() # Work on a copy

    # --- 1. Identify Price Column and Ensure Core Indicator Columns Exist ---
    price_col_name = 'adj_close' if 'adj_close' in data.columns and not data['adj_close'].isnull().all() else 'close'

    required_cols_for_core_strategy = ['RSI', 'SMA_short', 'SMA_long', price_col_name]
    missing_cols = [col for col in required_cols_for_core_strategy if col not in data.columns or data[col].isnull().all()]
    if missing_cols:
        logger.error(f"Missing required columns or all NaNs for core strategy: {missing_cols}. Cannot generate signals.")
        return pd.DataFrame()

    data['price'] = data[price_col_name] # Price at which signal is generated
    data['signal'] = 0  # Initialize signal: 0 for Hold

    # --- 2. Define Core Buy Signal Conditions (as per assignment) ---
    # Condition 1: RSI < rsi_oversold_threshold
    rsi_buy_cond = data['RSI'] < rsi_oversold_threshold
    # Condition 2: 20-DMA (SMA_short) crosses above 50-DMA (SMA_long)
    # A cross means SMA_short is now > SMA_long, AND it was <= on the previous day.
    sma_cross_buy_cond = (data['SMA_short'] > data['SMA_long']) & \
                         (data['SMA_short'].shift(1) <= data['SMA_long'].shift(1))

    # Combine core buy conditions
    core_buy_signal = rsi_buy_cond & sma_cross_buy_cond
    logger.debug(f"Intermediate: RSI Buy conditions met: {rsi_buy_cond.sum()} times")
    logger.debug(f"Intermediate: SMA Cross Buy conditions met: {sma_cross_buy_cond.sum()} times")
    logger.debug(f"Intermediate: Core Buy Signal (RSI & SMA_Cross) met: {core_buy_signal.sum()} times")


    # Advanced Buy Conditions ---
    advanced_buy_confirmation = pd.Series(True, index=data.index) # Start with True, AND with each condition

    if use_bb_for_buy:
        if 'BB_lower' in data.columns and not data['BB_lower'].isnull().all():
            advanced_buy_confirmation &= (data['price'] > data['BB_lower'])
            logger.debug(f"Advanced Buy: Applied BB condition. Met: {(data['price'] > data['BB_lower']).sum()} times")
        else:
            logger.warning("BB_lower column not available or all NaN for 'use_bb_for_buy' condition. Skipping BB buy condition.")

    if use_volume_for_buy:
        # Assuming 'VOLUME_RATIO' is calculated by indicators_calculator.py
        # VOLUME_RATIO = current_volume / volume_moving_average
        if 'VOLUME_RATIO' in data.columns and not data['VOLUME_RATIO'].isnull().all():
            advanced_buy_confirmation &= (data['VOLUME_RATIO'] > volume_ratio_min_buy)
            logger.debug(f"Advanced Buy: Applied Volume Ratio condition. Met: {(data['VOLUME_RATIO'] > volume_ratio_min_buy).sum()} times")
        else:
            logger.warning("VOLUME_RATIO column not available or all NaN for 'use_volume_for_buy' condition. Skipping volume buy condition.")

    if price_above_sma_long_for_buy:
        if 'SMA_long' in data.columns and not data['SMA_long'].isnull().all():
             advanced_buy_confirmation &= (data['price'] > data['SMA_long'])
             logger.debug(f"Advanced Buy: Applied Price > SMA_long condition. Met: {(data['price'] > data['SMA_long']).sum()} times")
        else:
            logger.warning("SMA_long column not available or all NaN for 'price_above_sma_long_for_buy' condition. Skipping.")


    # --- 4. Combine Core and Advanced for Final Buy Signal ---
    # The assignment implies the core conditions are primary.
    # Advanced conditions can act as additional confirmations.
    final_buy_signal = core_buy_signal & advanced_buy_confirmation
    data.loc[final_buy_signal, 'signal'] = 1
    logger.debug(f"Final Buy Signals applied: {final_buy_signal.sum()} times")


    # --- 5. Define Sell Signal Conditions (for backtesting or discretionary exit) ---
    # Example: Sell if RSI is overbought OR (if configured) short MA crosses below long MA
    rsi_sell_cond = data['RSI'] > rsi_overbought_threshold
    sma_cross_sell_cond = (data['SMA_short'] < data['SMA_long']) & \
                          (data['SMA_short'].shift(1) >= data['SMA_long'].shift(1))

    sell_signal_components = []
    sell_signal_components.append(rsi_sell_cond)
    if ma_cross_confirms_sell:
        sell_signal_components.append(sma_cross_sell_cond)

    # Optional Bollinger Band sell condition
    if use_bb_for_sell:
        if 'BB_middle' in data.columns and not data['BB_middle'].isnull().all():
            # Example: Price closes below middle Bollinger Band
            bb_sell_cond = (data['price'] < data['BB_middle'])
            sell_signal_components.append(bb_sell_cond)
            logger.debug(f"Sell: Applied BB condition. Met: {bb_sell_cond.sum()} times")

        else:
            logger.warning("BB_middle column not available for 'use_bb_for_sell' condition. Skipping BB sell condition.")

    # Combine sell conditions (using OR logic - any one can trigger a sell)
    final_sell_signal = pd.Series(False, index=data.index)
    if sell_signal_components:
        for condition in sell_signal_components:
            final_sell_signal |= condition # OR accumulate sell conditions

    # Apply sell signal only if not a buy signal on the same day (Buy takes precedence if conflicting)
    data.loc[final_sell_signal & (data['signal'] != 1), 'signal'] = -1
    logger.debug(f"Final Sell Signals applied (after Buy precedence): {(data['signal'] == -1).sum()} times")

    # --- 6. Calculate Signal Strength (Example based on your previous logic) ---
    data['signal_strength'] = 0.0  # Initialize as float

    # For Buy signals
    if 'VOLUME_RATIO' in data.columns: # Ensure necessary columns for strength calc
        rsi_strength = ((rsi_oversold_threshold - data['RSI']).clip(lower=0) / (rsi_oversold_threshold + 1e-9)) * 30
        ma_ratio_val = (data['SMA_short'] / (data['SMA_long'] + 1e-9)) # Avoid division by zero
        ma_strength = (ma_ratio_val - 1).clip(lower=0, upper=0.1) * 300 # Scaled impact for ratio slightly above 1
        volume_strength = (data['VOLUME_RATIO'] - 1).clip(lower=0, upper=2) * 20 # Scaled impact

        calculated_buy_strength = (rsi_strength + ma_strength + volume_strength).clip(lower=0, upper=100)
        data.loc[data['signal'] == 1, 'signal_strength'] = calculated_buy_strength[data['signal'] == 1]
    else:
        logger.warning("VOLUME_RATIO column missing, buy signal_strength may not be fully calculated or accurate.")
        # Fallback if volume ratio is missing but other parts are present
        if 'RSI' in data.columns and 'SMA_short' in data.columns and 'SMA_long' in data.columns:
            rsi_strength_fallback = ((rsi_oversold_threshold - data['RSI']).clip(lower=0) / (rsi_oversold_threshold + 1e-9)) * 45
            ma_ratio_val_fallback = (data['SMA_short'] / (data['SMA_long'] + 1e-9))
            ma_strength_fallback = (ma_ratio_val_fallback - 1).clip(lower=0, upper=0.1) * 450
            fallback_buy_strength = (rsi_strength_fallback + ma_strength_fallback).clip(lower=0, upper=90) # Cap slightly lower
            data.loc[data['signal'] == 1, 'signal_strength'] = fallback_buy_strength[data['signal'] == 1]


    #  Prepare final output DataFrame ---
    # Filter out initial rows where indicators might be NaN due to lookback periods
    # A simple way is to find the first valid index after all core indicators are non-NaN
    first_valid_idx = data.dropna(subset=required_cols_for_core_strategy).index.min()

    if pd.notna(first_valid_idx):
        signals_output_df = data.loc[first_valid_idx:, ['price', 'signal', 'signal_strength']].copy()
    else:
        logger.warning("Could not determine a valid start for signals due to NaNs in core indicators. "
                       "Returning all data; initial signals may be unreliable.")
        signals_output_df = data[['price', 'signal', 'signal_strength']].copy()

    logger.info(f"Signal generation complete. Buy signals: {(signals_output_df['signal'] == 1).sum()}, "
                f"Sell signals: {(signals_output_df['signal'] == -1).sum()}")
    return signals_output_df


