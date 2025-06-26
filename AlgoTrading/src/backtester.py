import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from src.logger_setup import setup_logger
from src.data_handler import DataFetcher
from src.indicators_calculator import insertion_to_df as calculate_indicators
from src.strategy_logic import generate_strategy_signals
from src.gsheet_manager import GoogleSheetsManager
from dotenv import load_dotenv
import os

logger = setup_logger('Backtester_Global') 

class Backtester:
    def __init__(self, config_path='config.yaml', gsheet_manager_instance=None):
        self.logger = setup_logger('Backtester', level='INFO') 
        self.config = self._load_config(config_path) 

        # Configure logger level from its own config
        if self.config and 'logging_level' in self.config: 
            self.logger.setLevel(self.config.get('logging_level', 'INFO'))
        elif self.config and 'logging_level' in self.config.get('backtest',{}): 
             self.logger.setLevel(self.config.get('backtest',{}).get('logging_level', 'INFO'))
        elif not self.config:
            self.logger.warning("Backtester config loading resulted in empty. Logger using default INFO.")

        self.data_fetcher = DataFetcher()
        self.gsheet_manager = None # Initialize to None

        if gsheet_manager_instance:
            self.gsheet_manager = gsheet_manager_instance
            self.logger.info("Using pre-initialized GoogleSheetsManager instance for Backtester.")
            if self.gsheet_manager: 
                 try:
                    self.gsheet_manager.create_project_worksheets() 
                    self.logger.info("Backtester GSheets worksheets ensured by pre-init manager.")
                 except Exception as e:
                     self.logger.error(f"Failed to ensure Backtester worksheets via pre-init manager: {e}")
        elif self.config.get('google_sheets'): 
            try:
                sheet_id = os.getenv('GOOGLE_SHEET_ID')
                creds_file = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE')
                
                if not sheet_id or sheet_id == "YOUR_SPREADSHEET_ID_HERE":
                    self.logger.warning("GOOGLE_SHEET_ID not set. GSheets disabled for Backtester.")
                elif not creds_file or not os.path.exists(creds_file):
                    self.logger.warning(f"GOOGLE_SHEETS_CREDENTIALS_FILE ('{creds_file}') not found. GSheets disabled for Backtester.")
                else:
                    temp_gs_manager = GoogleSheetsManager()
                    if temp_gs_manager.test_connection():
                        self.gsheet_manager = temp_gs_manager
                        self.gsheet_manager.create_project_worksheets() 
                        self.logger.info("GSheets initialized successfully by Backtester.")
                    else:
                        self.logger.error("GSheets connection test failed (Backtester init). Disabling.")
            except Exception as e:
                self.logger.error(f"Failed to init GSheetsManager in Backtester: {e}. Disabled.")
        else:
            self.logger.info("No 'google_sheets' config for Backtester or GSheet instance not provided. GSheets disabled.")
        
        self.indicator_params = self.config.get('indicators', {})
        self.strategy_rules = self.config.get('backtest', {}).get('strategy_rules', 
                                self.config.get('strategy_rules', {})) 
        self.backtest_config = self.config.get('backtest', {})
        self.google_sheets_config = self.config.get('google_sheets', {})

    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if config_data is None:
                    self.logger.warning(f"Configuration file '{config_path}' is empty. Using default empty config.")
                    return {}
                self.logger.info(f"Configuration loaded successfully from {config_path}")
                return config_data
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found at {config_path}. Using default empty config.")
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML in configuration file {config_path}: {e}. Using default empty config.")
            return {}
        except Exception as e:
            self.logger.error(f"An unexpected error occurred loading configuration from {config_path}: {e}. Using default empty config.")
            return {}

    def _get_backtest_dates(self):
        period_months = self.backtest_config.get('period_months', 6)
        end_date_dt = datetime.now()
        start_date_dt = end_date_dt - timedelta(days=period_months * 30.4375) 

        return start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d')

    def run_backtest(self, symbols=None):
        if symbols is None:
            symbols = self.config.get('stocks_to_track', [])
        if not symbols:
            self.logger.error("No stock symbols provided for backtesting.")
            return {"all_trades": pd.DataFrame(), "summary": {"error": "No symbols provided"}}

        start_date, end_date = self._get_backtest_dates()
        interval = self.config.get('backtest', {}).get('data_fetch_interval', # Get interval from backtest section
                                self.config.get('data_fetch_interval', '1d')) # Fallback to global
        self.logger.info(f"Effective backtest period: {start_date} to {end_date} with interval {interval}")

        all_trades_list = [] # Use a list to collect DataFrames 

        for symbol in symbols:
            self.logger.info(f"--- Starting backtest for {symbol} ---")
            
            stock_data = self.data_fetcher.fetch_stock_data(symbol, 
                                                             start_date=start_date, 
                                                             end_date=end_date, 
                                                             interval=interval)
            if stock_data.empty:
                self.logger.warning(f"No data fetched for {symbol}. Skipping.")
                continue
            self.logger.debug(f"Data for {symbol} (first 3, last 3 rows):\n{stock_data.head(3)}\n...\n{stock_data.tail(3)}")

            # Pass only relevant indicator parameters from the main config's 'indicators' section
            df_with_indicators = calculate_indicators(stock_data.copy(), **self.indicator_params)
            if df_with_indicators.empty:
                self.logger.warning(f"Indicator calculation failed for {symbol}. Skipping.")
                continue
            current_strategy_rules = self.config.get('backtest', {}).get('strategy_rules', 
                                        self.config.get('strategy_rules', {}))
            if not current_strategy_rules:
                 self.logger.warning(f"No strategy rules found in config for {symbol}. Signal generation might be affected.")

            signals_df = generate_strategy_signals(df_with_indicators.copy(), **current_strategy_rules)
            
            if signals_df.empty:
                self.logger.warning(f"Signal generation returned empty DataFrame for {symbol}. Skipping trade simulation.")
                continue

            # connects the  signals with indicator data for simulation process
            analysis_df = df_with_indicators.join(signals_df[['signal', 'signal_strength']], how='left')
            analysis_df['signal'] = analysis_df['signal'].fillna(0) # Default no signal
            analysis_df['signal_strength'] = analysis_df['signal_strength'].fillna(0) 

            trades_for_symbol = self._simulate_trades(analysis_df, symbol) # <--simulate_trades returns a list of dicts
            
            if trades_for_symbol:
                symbol_trades_df = pd.DataFrame(trades_for_symbol)
                all_trades_list.append(symbol_trades_df) # Add DataFrame to list
                
                pnl_sum = symbol_trades_df['PnL_Percent'].sum()
                win_rate = (symbol_trades_df['PnL_Percent'] > 0).mean() * 100 if not symbol_trades_df.empty else 0
                num_trades = len(symbol_trades_df)
                self.logger.info(f"Backtest for {symbol}: Total Trades={num_trades}, PnL Sum={pnl_sum:.2f}%, Win Rate={win_rate:.2f}%")
            else:
                self.logger.info(f"No trades simulated for {symbol} based on generated signals and trade logic.")

        all_trades_df = pd.DataFrame()
        if all_trades_list:
            all_trades_df = pd.concat(all_trades_list, ignore_index=True)

        # Initialize summary_dict with default values for the case of no trades
        summary_dict = {
            'Run_Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Total_Stocks_Tested': len(symbols),
            'Overall_PnL_Sum_Pct': "0.00",
            'Overall_Win_Rate_Pct': "0.00",
            'Total_Completed_Trades': 0,
            'Avg_PnL_Per_Trade_Pct': "0.00",
            'Notes': f"Backtest from {start_date} to {end_date} with interval {interval}. No trades executed across all symbols."
        }

        if not all_trades_df.empty:
            overall_pnl_sum = all_trades_df['PnL_Percent'].sum()
            overall_win_rate = (all_trades_df['PnL_Percent'] > 0).mean() * 100
            total_completed_trades = len(all_trades_df)
            avg_pnl_per_trade = all_trades_df['PnL_Percent'].mean()

            self.logger.info("--- Overall Backtest Summary ---")
            self.logger.info(f"Total Stocks Tested: {len(symbols)}")
            self.logger.info(f"Overall PnL Sum: {overall_pnl_sum:.2f}%")
            self.logger.info(f"Overall Win Rate: {overall_win_rate:.2f}%")
            self.logger.info(f"Total Completed Trades: {total_completed_trades}")
            self.logger.info(f"Average PnL per Trade: {avg_pnl_per_trade:.2f}%")

            # Update the dictionary with actual results
            summary_dict.update({ 
                'Overall_PnL_Sum_Pct': f"{overall_pnl_sum:.2f}",
                'Overall_Win_Rate_Pct': f"{overall_win_rate:.2f}",
                'Total_Completed_Trades': total_completed_trades,
                'Avg_PnL_Per_Trade_Pct': f"{avg_pnl_per_trade:.2f}",
                'Notes': f"Backtest from {start_date} to {end_date} with interval {interval}."
            })
            
            if self.gsheet_manager and self.config.get('google_sheets', {}).get('enabled_for_app', False): # Check main GSheet enable flag
                self._log_to_google_sheets(all_trades_df, summary_dict)
            else:
                self.logger.warning("Google Sheets manager not available or GSheets disabled in config. Skipping cloud logging for backtest.")
                
        else:
            self.logger.info("No trades were executed across all symbols in this backtest run.")
            # Log the "no trades" summary if GSheets is enabled
            if self.gsheet_manager and self.config.get('google_sheets', {}).get('enabled_for_app', False):
                self._log_to_google_sheets(pd.DataFrame(), summary_dict) # Log empty trades df and the no-trades summary

        # Return results to the streamlit application 
        return {"all_trades": all_trades_df, "summary": summary_dict}

    def _log_to_google_sheets(self, all_trades_df, summary_dict):
        """Enhanced Google Sheets logging with better error handling"""
        try:
            # Get configuration
            trades_log_sheet_name = self.google_sheets_config.get('backtest_trade_log_sheet_name', 'Backtest_TradeLog')
            summary_sheet_name = self.google_sheets_config.get('backtest_summary_sheet_name', 'Backtest_Summary')
            clear_trades_log = self.google_sheets_config.get('clear_trades_log_on_backtest', True)
            
            # Add timestamp to trades
            all_trades_df = all_trades_df.copy()
            all_trades_df['Log_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Define expected columns for trades log
            expected_trade_columns = [
                'Log_Timestamp', 'Symbol', 'Entry_Date', 'Signal_Action', 'Entry_Price',
                'Exit_Date', 'Exit_Price', 'PnL_Percent', 'Signal_Strength', 'Reason',
                'RSI_at_Entry', 'MA_Ratio_at_Entry'
            ]
            
            # Ensure all expected columns exist
            for col in expected_trade_columns:
                if col not in all_trades_df.columns:
                    all_trades_df[col] = np.nan
                    
            # Reorder columns to match expected order
            trades_df_ordered = all_trades_df[expected_trade_columns].copy()
            
            self.logger.info(f"Attempting to log {len(trades_df_ordered)} trades to Google Sheets...")
            self.logger.debug(f"Trades data preview:\n{trades_df_ordered.head()}")
            
            # Log trades
            trades_success = self.gsheet_manager.log_trades_from_dataframe(
                trades_df_ordered,
                sheet_name=trades_log_sheet_name,
                clear_sheet=clear_trades_log
            )
            
            if trades_success:
                self.logger.info(f"Successfully logged {len(trades_df_ordered)} trades to '{trades_log_sheet_name}'")
            else:
                self.logger.error(f"Failed to log trades to '{trades_log_sheet_name}'")
            
            # Log summary
            self.logger.info("Attempting to log summary to Google Sheets...")
            self.logger.debug(f"Summary data: {summary_dict}")
            
            summary_success = self.gsheet_manager.update_key_value_sheet(
                summary_dict,
                sheet_name=summary_sheet_name,
                clear_sheet=False
            )
            
            if summary_success:
                self.logger.info(f"Successfully logged summary to '{summary_sheet_name}'")
            else:
                self.logger.error(f"Failed to log summary to '{summary_sheet_name}'")
                
        except Exception as e:
            self.logger.error(f"Error during Google Sheets logging: {e}", exc_info=True)

    def _simulate_trades(self, data_df, symbol):
        trades = []
        in_position = False
        entry_price = 0.0
        entry_date = None
        entry_signal_strength = 0.0
        entry_rsi = np.nan
        entry_ma_ratio = np.nan

        commission = self.backtest_config.get('commission_per_trade_percent', 0.001) 
        trade_on = self.backtest_config.get('trade_on', 'next_open') 

        # Make a copy to avoid SettingWithCopyWarning if data_df is a slice
        sim_data_df = data_df.copy()
        if trade_on == 'next_open':
            sim_data_df['next_open'] = sim_data_df['open'].shift(-1)

        for i, row_series in sim_data_df.iterrows():
            current_signal = row_series.get('signal', 0)
            current_close = row_series.get('close')
            next_day_open = row_series.get('next_open')

            # Check if it's the last row for 'next_open' logic
            is_last_row_for_next_open = False
            if trade_on == 'next_open':
                 loc = sim_data_df.index.get_loc(i)
                 if loc == len(sim_data_df.index) -1 :
                      is_last_row_for_next_open = True

            if trade_on == 'next_open' and pd.isna(next_day_open) and not is_last_row_for_next_open:
                if current_signal == 1 or (in_position and current_signal == -1):
                    self.logger.debug(f"{i} {symbol}: Cannot trade on next_open as it is NaN and not the last day of data.")
                continue

            if current_signal == 1 and not in_position:
                entry_price_log = np.nan
                entry_date_actual = None

                if trade_on == 'next_open':
                    if not pd.isna(next_day_open):
                        entry_price_actual = next_day_open * (1 + commission)
                        entry_price_log = next_day_open
                        current_loc = sim_data_df.index.get_loc(i)
                        if current_loc + 1 < len(sim_data_df.index):
                             entry_date_actual = sim_data_df.index[current_loc + 1]
                        else:
                             self.logger.warning(f"Attempting to log entry for {symbol} on next_open, but it's the last index.")
                             entry_date_actual = i
                    elif is_last_row_for_next_open:
                        self.logger.debug(f"{i} {symbol}: BUY signal on last day, but cannot enter on next_open.")
                        continue
                elif trade_on == 'signal_close':
                    entry_price_actual = current_close * (1 + commission)
                    entry_price_log = current_close
                    entry_date_actual = i
                
                if entry_price_log is not np.nan and entry_date_actual is not None:
                    entry_price = entry_price_actual
                    in_position = True
                    entry_date = entry_date_actual
                    entry_signal_strength = row_series.get('signal_strength', 0.0)
                    entry_rsi = row_series.get('RSI', np.nan)
                    entry_ma_ratio = row_series.get('MA_ratio', np.nan)
                    self.logger.debug(f"TRADE_LOG: {entry_date_actual.date()} {symbol}: ENTER BUY based on signal from {i.date()} at price {entry_price_log:.2f} (cost: {entry_price:.2f}). Strength: {entry_signal_strength:.2f}")

            elif (current_signal == -1 and in_position) or \
                 (in_position and is_last_row_for_next_open and trade_on == 'next_open' and pd.isna(next_day_open)) or \
                 (in_position and i == sim_data_df.index[-1] and trade_on == 'signal_close'):
                
                exit_price_log = np.nan
                exit_date_actual = None
                exit_reason = "Sell signal"

                if trade_on == 'next_open':
                    if not pd.isna(next_day_open):
                        exit_price_actual = next_day_open * (1 - commission)
                        exit_price_log = next_day_open
                        current_loc = sim_data_df.index.get_loc(i)
                        if current_loc + 1 < len(sim_data_df.index):
                             exit_date_actual = sim_data_df.index[current_loc + 1]
                        else:
                             exit_date_actual = i
                    elif is_last_row_for_next_open:
                        self.logger.debug(f"{i.date()} {symbol}: In position on last day of data (next_open is NaN). Exiting at current close.")
                        exit_price_actual = current_close * (1 - commission)
                        exit_price_log = current_close
                        exit_date_actual = i
                        exit_reason = "Forced EOD (last day)"
                    else:
                        self.logger.debug(f"{i.date()} {symbol}: SELL signal, but next_open is NaN and not last day. Holding position.")
                        continue 
                elif trade_on == 'signal_close':
                    exit_price_actual = current_close * (1 - commission)
                    exit_price_log = current_close
                    exit_date_actual = i
                    if i == sim_data_df.index[-1] and current_signal != -1:
                        exit_reason = "Forced EOD (last day)"
                
                if exit_price_log is not np.nan and exit_date_actual is not None and entry_price > 0:
                    pnl_percent = ((exit_price_actual / entry_price) - 1) * 100
                    trades.append({
                        'Symbol': symbol, 
                        'Entry_Date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date), 
                        'Signal_Action': 'BUY',
                        'Entry_Price': entry_price / (1+commission) if (1+commission) != 0 else entry_price,
                        'Exit_Date': exit_date_actual.strftime('%Y-%m-%d') if hasattr(exit_date_actual, 'strftime') else str(exit_date_actual), 
                        'Exit_Price': exit_price_log,
                        'PnL_Percent': pnl_percent, 
                        'Signal_Strength': entry_signal_strength,
                        'Reason': exit_reason,
                        'RSI_at_Entry': entry_rsi, 
                        'MA_Ratio_at_Entry': entry_ma_ratio
                    })
                    self.logger.debug(f"TRADE_LOG: {exit_date_actual.date()} {symbol}: EXIT SELL based on signal from {i.date()} at price {exit_price_log:.2f} (recv: {exit_price_actual:.2f}). PnL: {pnl_percent:.2f}%. Reason: {exit_reason}")
                    in_position = False
                    entry_price = 0.0
        
        # Final check: if still in position after the loop
        if in_position and entry_price > 0:
             last_row_series = sim_data_df.iloc[-1]
             self.logger.warning(f"{symbol}: Still in position after loop. Forcing exit on last available close: {last_row_series.name.date()}. This indicates an edge case in exit logic.")
             exit_price_actual = last_row_series.get('close') * (1 - commission)
             pnl_percent = ((exit_price_actual / entry_price) - 1) * 100
             trades.append({
                'Symbol': symbol, 
                'Entry_Date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date), 
                'Signal_Action': 'BUY',
                'Entry_Price': entry_price / (1+commission) if (1+commission) != 0 else entry_price,
                'Exit_Date': last_row_series.name.strftime('%Y-%m-%d') if hasattr(last_row_series.name, 'strftime') else str(last_row_series.name), 
                'Exit_Price': last_row_series.get('close'),
                'PnL_Percent': pnl_percent, 
                'Signal_Strength': entry_signal_strength,
                'Reason': 'Forced EOD (end of data - safeguard)',
                'RSI_at_Entry': entry_rsi, 
                'MA_Ratio_at_Entry': entry_ma_ratio
            })
             self.logger.debug(f"TRADE_LOG: {last_row_series.name.date()} {symbol}: EXIT SELL (safeguard) at {last_row_series.get('close'):.2f}. PnL: {pnl_percent:.2f}%")
        return trades


if __name__ == '__main__':
    load_dotenv()  # Load environment variables
    logger.info("--- Initiating Backtester ---")
    
    # Check environment variables but don't exit - let the backtester handle it gracefully
    sheet_id = os.getenv('GOOGLE_SHEET_ID')
    creds_file = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE')
    
    if not sheet_id or sheet_id == "YOUR_SPREADSHEET_ID_HERE":
        logger.warning("GOOGLE_SHEET_ID is not properly configured in .env file")
    if not creds_file or not os.path.exists(creds_file):
        logger.warning(f"GOOGLE_SHEETS_CREDENTIALS_FILE not found: {creds_file}")

    backtester = Backtester(config_path='config.yaml')
    backtester.run_backtest()
    logger.info("--- Backtesting Complete ---")