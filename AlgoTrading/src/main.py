import os
import yaml
import asyncio 
from dotenv import load_dotenv
from src.logger_setup import setup_logger
from src.data_handler import DataFetcher
from src.indicators_calculator import insertion_to_df as calculate_indicators
from src.gsheet_manager import GoogleSheetsManager
from src.backtester import Backtester
from src.ml_predictor import predict_multiple_stocks
from src.alert_messenger import TelegramManager 



# Global logger 
main_logger = setup_logger('MainApp', level='INFO')

# Global Telegram Manager instance
tg_manager_main: TelegramManager = None 

def load_app_configuration(config_path='config.yaml'):
    """Loads the main YAML configuration file for the application."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                main_logger.warning(f"Main configuration file '{config_path}' is empty. App may not function correctly.")
                return {}
            main_logger.info(f"Main configuration loaded successfully from {config_path}")
            return config
    except FileNotFoundError:
        main_logger.error(f"Main configuration file not found: {config_path}. Exiting.")
        return {}
    except yaml.YAMLError as e:
        main_logger.error(f"Error parsing YAML in main config {config_path}: {e}. Exiting.")
        return {}
    except Exception as e:
        main_logger.error(f"Unexpected error loading main config {config_path}: {e}. Exiting.")
        return {}

def initialize_global_gsheet_manager(app_config):
    gs_config = app_config.get('google_sheets', {})
    if not gs_config.get('enabled_for_app', False):
        main_logger.info("Google Sheets integration is globally disabled ('google_sheets.enabled_for_app' is false).")
        return None
    sheet_id = os.getenv('GOOGLE_SHEET_ID')
    creds_file = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE')

    if not sheet_id or sheet_id == "YOUR_SPREADSHEET_ID_HERE":
        main_logger.warning("GOOGLE_SHEET_ID env var not properly set. Google Sheets integration disabled for the app.")
        return None
    if not creds_file:
        main_logger.warning("GOOGLE_SHEETS_CREDENTIALS_FILE env var is not set. Google Sheets disabled.")
        return None
    if not os.path.exists(creds_file):
        main_logger.warning(f"GOOGLE_SHEETS_CREDENTIALS_FILE ('{creds_file}') not found. Google Sheets disabled.")
        return None

    try:
        gs_manager = GoogleSheetsManager() 
        if gs_manager.test_connection(): 
            main_logger.info("GoogleSheetsManager initialized globally and connection tested successfully.")
            return gs_manager
        else:
            main_logger.error("GoogleSheetsManager global connection test failed. Disabling GSheet features for the app.")
            return None
    except Exception as e:
        main_logger.error(f"Failed to initialize GoogleSheetsManager globally: {e}", exc_info=True)
        return None

# --- Telegram Message Formatting Helpers for main.py ---
def format_backtest_summary_for_telegram(summary_data, trades_df=None):
    if not summary_data:
        return "Backtest summary data not available."
    
    lines = [
        f" Backtest Run Timestamp: {summary_data.get('Run_Timestamp', 'N/A')}",
        f"Strategy: {summary_data.get('Strategy_Name', 'N/A')}",
        f"Stocks: {', '.join(summary_data.get('Symbols_Tested', ['N/A']))}",
        f"Trades: {summary_data.get('Total_Completed_Trades', 'N/A')}",
        f"PnL Sum: {float(summary_data.get('Overall_PnL_Sum_Pct', 0)):.2f}%",
        f"Win Rate: {float(summary_data.get('Overall_Win_Rate_Pct', 0)):.2f}%",
        f"Avg PnL/Trade: {float(summary_data.get('Avg_PnL_Per_Trade_Pct', 0)):.2f}%",
        f"Sharpe: {float(summary_data.get('Sharpe_Ratio', 0)):.2f}",
        f"Max Drawdown: {float(summary_data.get('Max_Drawdown_Pct', 0)):.2f}%",
    ]
    if trades_df is not None and not trades_df.empty:
        lines.append(f"Total Trades Logged: {len(trades_df)}")
    
    notes = summary_data.get('Notes', '')
    if notes:
        lines.append(f"\n Notes: {notes}")
    return "\n".join(lines)

def format_ml_summary_for_telegram(ml_results):
    if not ml_results:
        return "ML prediction results not available."

    summary = ml_results.get('summary', {})
    predictions = ml_results.get('predictions', [])
    evaluations = ml_results.get('training_evaluations', [])

    lines = [
        f"Model Type: {summary.get('Model_Type_Used', 'N/A')}",
        f"Target Strategy: {summary.get('Target_Strategy_Used', 'N/A')}",
        f"Stocks Processed: {summary.get('Total_Stocks_Processed', 'N/A')}",
        f"Successful Preds: {summary.get('Successful_Predictions', 'N/A')}",
        f"Avg Max Conf: {float(summary.get('Average_Max_Confidence', 0)):.2f}"
    ]

    if predictions:
        lines.append("\n Top Predictions (Sample):")
        for pred in predictions[:min(3, len(predictions))]: # Show top 3 or fewer
            lines.append(f"  - {pred.get('Symbol', 'N/S')}: {pred.get('Predicted_Movement', 'N/A')} "
                         f"(Conf: {float(pred.get('Max_Confidence', 0)):.2f}, "
                         f"Signal: {pred.get('Final_Signal_Generated', 'N/A')})")
    
    if evaluations:
        avg_test_accuracy = sum(e.get('Test_Accuracy', 0) for e in evaluations) / len(evaluations) if evaluations else 0
        avg_auc = sum(e.get('AUC_Score', 0) for e in evaluations) / len(evaluations) if evaluations else 0
        lines.append("\nAvg Training Performance:")
        lines.append(f"  Avg Test Accuracy: {avg_test_accuracy:.2f}")
        lines.append(f"  Avg AUC Score: {avg_auc:.2f}")

    return "\n".join(lines)


async def main_async(): 
    """
    Main asynchronous function to orchestrate backtesting and ML predictions.
    """
    global tg_manager_main # Allow modification of global instance

    main_logger.info("Main Application Started (Async) ")
    load_dotenv() 

    app_config = load_app_configuration('config.yaml')
    if not app_config:
        main_logger.critical("Application cannot run without a valid configuration. Exiting.")
        return

    # Initialize Telegram Manager
    try:
        tg_manager_main = TelegramManager()
        if await tg_manager_main.health_check():
            main_logger.info("TelegramManager initialized and health check passed.")
            await tg_manager_main.send("Trading Engine Main Script: Startup successful.", prefix="System Info: ")
        else:
            main_logger.warning("Telegram health check failed. Notifications may not be sent. Continuing execution...")
    except ValueError as ve:
        main_logger.error(f"Failed to initialize TelegramManager (config error): {ve}. Telegram features disabled.")
        tg_manager_main = None 
    except Exception as e:
        main_logger.error(f"Failed to initialize TelegramManager: {e}. Telegram features disabled.", exc_info=True)
        tg_manager_main = None


    if 'logging_level' in app_config:
        main_logger.setLevel(app_config['logging_level'])
        main_logger.info(f"MainApp logger level set to {app_config['logging_level']}.")

    stocks_to_track = app_config.get('stocks_to_track', [])
    if not stocks_to_track:
        main_logger.warning("No 'stocks_to_track' defined in configuration. Some modules may not run.")
    
    data_fetcher = DataFetcher()
    gsheet_manager = initialize_global_gsheet_manager(app_config)

    backtest_summary_data = None
    ml_prediction_results = None

    try:
        #Run Backtester
        if app_config.get('run_backtester', False):
            main_logger.info("--- Initializing and Running Backtester ---")
            if tg_manager_main:
                await tg_manager_main.send(f" Starting Backtester for: {', '.join(stocks_to_track) if stocks_to_track else 'N/A'}", prefix=" BT Status: ")
            
            backtester = Backtester(config_path='config.yaml', gsheet_manager_instance=gsheet_manager)
            backtest_run_results = backtester.run_backtest(symbols=stocks_to_track) 
            
            if backtest_run_results and isinstance(backtest_run_results, dict):
                backtest_summary_data = backtest_run_results.get('summary')
                # trades_df_from_bt = backtest_run_results.get('all_trades') # If needed
                if backtest_summary_data:
                    main_logger.info("--- Backtesting Run Complete ---")
                    if tg_manager_main:
                        bt_report_tg = format_backtest_summary_for_telegram(backtest_summary_data)
                        await tg_manager_main.send(bt_report_tg, use_summary=True, prefix=" Backtest Report (Main Script):\n")
                else:
                    main_logger.warning("Backtester ran but did not return summary data.")
                    if tg_manager_main:
                         await tg_manager_main.send(" Backtester ran but no summary data found.", prefix=" BT Status: ")
            else:
                main_logger.warning("Backtester did not return expected results dictionary.")
                if tg_manager_main:
                    await tg_manager_main.send(" Backtester ran but did not return expected results.", prefix=" BT Status: ")

        else:
            main_logger.info("Backtester run is disabled in main configuration ('run_backtester': false).")

        # Ruun Ml_predictor
        if app_config.get('run_ml_predictor', False):
            main_logger.info("--- Initializing and Running ML Predictions ---")
            if not app_config.get('ml_predictor', {}).get('enabled', False):
                main_logger.info("ML Predictor module is internally disabled ('ml_predictor.enabled': false). Skipping ML run.")
            elif not stocks_to_track:
                main_logger.warning("No stocks to track for ML Predictor. Skipping ML run.")
            else:
                if tg_manager_main:
                    await tg_manager_main.send(f" Starting ML Predictor for: {', '.join(stocks_to_track) if stocks_to_track else 'N/A'}", prefix=" ML Status: ")
                
                ml_prediction_results = predict_multiple_stocks(
                    symbols=stocks_to_track,
                    config=app_config, 
                    data_fetcher=data_fetcher,
                    indicator_calculator_func=calculate_indicators,
                    gsheet_manager=gsheet_manager 
                )
                
                if ml_prediction_results and ml_prediction_results.get('summary'):
                    main_logger.info("ML Prediction Overall Summary (from main):")
                    for key, value in ml_prediction_results['summary'].items():
                        main_logger.info(f"  {key}: {value}")
                    if tg_manager_main:
                        ml_report_tg = format_ml_summary_for_telegram(ml_prediction_results)
                        await tg_manager_main.send(ml_report_tg, use_summary=True, prefix="🤖 ML Prediction Report (Main Script):\n")
                else:
                    main_logger.info("ML prediction process completed, but no overall summary was returned to main.")
                    if tg_manager_main:
                        await tg_manager_main.send(" ML Predictor ran but no summary data found.", prefix="🤖 ML Status: ")
                main_logger.info("--- ML Predictions Run Complete ---")
        else:
            main_logger.info("ML Predictor run is disabled in main configuration ('run_ml_predictor': false).")

    except Exception as e:
        main_logger.critical(f"Critical error during main execution: {e}", exc_info=True)
        if tg_manager_main:
            # Format a more detailed error message if possible
            error_details = f"Type: {type(e).__name__}\nMessage: {e}\n"
            # Potentially add a snippet of traceback if not too long, or rely on Gemini summary
            import traceback
            tb_str = traceback.format_exc()
            error_details += f"\nTraceback (summary will be attempted if too long):\n{tb_str}"
            await tg_manager_main.send_error_alert(error_details, context="Main Script Execution Block")
    finally:
        main_logger.info("--- Main Application Finished ---")
        if tg_manager_main:
            await tg_manager_main.send("Trading Engine Main Script: Shutdown complete.", prefix=" System Info: ")
            await tg_manager_main.stop() # Explicitly stop the client session for main script


if __name__ == '__main__':
    asyncio.run(main_async())