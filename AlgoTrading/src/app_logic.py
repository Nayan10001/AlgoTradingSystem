import os
import yaml
import pandas as pd
from dotenv import load_dotenv
import logging 
from src.logger_setup import setup_logger 
from src.data_handler import DataFetcher
from src.indicators_calculator import insertion_to_df as calculate_indicators
from src.gsheet_manager import GoogleSheetsManager
from src.backtester import Backtester
from src.ml_predictor import predict_multiple_stocks
from src.alert_messenger import TelegramManager 

# Global logger for app_logic 
logic_logger = setup_logger('AppLogic', level='INFO')

def load_configuration(config_path='config.yaml'):
    """Loads the main YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                logic_logger.warning(f"Config file '{config_path}' is empty.")
                return {}
            logic_logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError:
        logic_logger.error(f"Config file not found: {config_path}.")
        return {}
    except yaml.YAMLError as e:
        logic_logger.error(f"Error parsing YAML in {config_path}: {e}.")
        return {}
    except Exception as e:
        logic_logger.error(f"Unexpected error loading config {config_path}: {e}.")
        return {}

def save_configuration(config_data, config_path='config.yaml'):
    """Saves configuration data to a YAML file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)
        logic_logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logic_logger.error(f"Error saving configuration to {config_path}: {e}")
        return False

def initialize_gsheet_manager(app_config):
    """Initializes GoogleSheetsManager based on app_config."""
    gs_config = app_config.get('google_sheets', {})
    if not gs_config.get('enabled_for_app', False):
        logic_logger.info("GSheets globally disabled in app config.")
        return None

    sheet_id = os.getenv('GOOGLE_SHEET_ID')
    creds_file = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE')

    if not sheet_id or sheet_id == "YOUR_SPREADSHEET_ID_HERE":
        logic_logger.warning("GOOGLE_SHEET_ID env var not set. GSheets disabled.")
        return None
    if not creds_file or not os.path.exists(creds_file):
        logic_logger.warning(f"GOOGLE_SHEETS_CREDENTIALS_FILE ('{creds_file}') not found. GSheets disabled.")
        return None
    
    try:
        gs_manager = GoogleSheetsManager()
        # Simplified connection test: Rely on GSheetManager's own test or a simple operation
        if hasattr(gs_manager, 'test_connection') and gs_manager.test_connection():
            logic_logger.info("GSheetManager initialized and connection tested successfully.")
            return gs_manager
        elif hasattr(gs_manager, 'service'): # Basic check if service attribute exists
             # Try a simple read to confirm, e.g., get spreadsheet properties
            try:
                spreadsheet = gs_manager.service.spreadsheets().get(spreadsheetId=sheet_id).execute()
                logic_logger.info(f"GSheetManager initialized successfully. Connected to: {spreadsheet.get('properties', {}).get('title', 'Unknown GSheet')}")
                return gs_manager
            except Exception as e_info:
                logic_logger.warning(f"GSheetManager initialized, but could not retrieve sheet info: {e_info}. Assuming connected.")
                return gs_manager # Still return, might work for other ops
        else:
            logic_logger.warning("GSheetManager initialized, but no standard test_connection method or service attribute found for detailed check.")
            return gs_manager # Assume basic initialization worked

    except Exception as e:
        logic_logger.error(f"Failed to initialize GSheetManager: {e}", exc_info=True)
        return None

# Added for Streamlit app
async def initialize_telegram_manager_for_app():
    """Initializes TelegramManager for the Streamlit app with a health check."""
    try:
        tgm = TelegramManager()
        # Health check is important here to confirm it works before app relies on it
        if await tgm.health_check(): # health_check now sends a test message
            logic_logger.info("TelegramManager initialized and health check passed for Streamlit app.")
            return tgm
        else:
            logic_logger.error("TelegramManager health check failed for Streamlit app. Notifications will be unavailable.")
            return None
    except ValueError as ve: # Catch init errors like missing API ID/Hash
        logic_logger.error(f"Failed to initialize TelegramManager (config error): {ve}. Telegram features disabled for app.")
        return None
    except Exception as e:
        logic_logger.error(f"Exception initializing TelegramManager for Streamlit app: {e}", exc_info=True)
        return None


def run_backtesting_logic(app_config, gsheet_manager_instance, stocks_to_run):
    logic_logger.info("--- Running Backtester Logic ---")
    if not app_config.get('run_backtester', False):
        logic_logger.info("Backtester run is disabled in app_config.")
        return None
    if not stocks_to_run:
        logic_logger.warning("No stocks provided for backtesting.")
        return {"error": "No stocks provided for backtesting."}

    try:
        save_configuration(app_config, 'config.yaml') 
        backtester = Backtester(config_path='config.yaml', gsheet_manager_instance=gsheet_manager_instance)
        results = backtester.run_backtest(symbols=stocks_to_run)
        
        logic_logger.info("--- Backtesting Logic Complete ---")
        if results and isinstance(results, dict):
             return results # Expects {"all_trades": df, "summary": dict}
        else:
            logic_logger.warning("Backtester did not return the expected dictionary format.")
            return {"error": "Backtester returned unexpected data format."}
    except Exception as e:
        logic_logger.error(f"Error during backtesting logic: {e}", exc_info=True)
        return {"error": str(e)}

#Ml prediction 
def run_ml_prediction_logic(app_config, gsheet_manager_instance, stocks_to_run):
    logic_logger.info(" Running ML Prediction Logic")
    if not app_config.get('run_ml_predictor', False) or \
       not app_config.get('ml_predictor', {}).get('enabled', False):
        logic_logger.info("ML Predictor run is disabled in app_config.")
        return None
    if not stocks_to_run:
        logic_logger.warning("No stocks provided for ML prediction.")
        return {"error": "No stocks provided for ML prediction."}

    try:
        data_fetcher = DataFetcher()
        ml_results = predict_multiple_stocks(
            symbols=stocks_to_run,
            config=app_config,
            data_fetcher=data_fetcher,
            indicator_calculator_func=calculate_indicators,
            gsheet_manager=gsheet_manager_instance
        )
        logic_logger.info(" ML Prediction Logic Complete")
        if ml_results and isinstance(ml_results, dict):
            return ml_results # Expects dict with 'summary', 'predictions', 'training_evaluations'
        else:
            logic_logger.warning("ML Predictor did not return the expected dictionary format.")
            return {"error": "ML Predictor returned unexpected data format."}
    except Exception as e:
        logic_logger.error(f"Error during ML prediction logic: {e}", exc_info=True)
        return {"error": str(e)}

def fetch_data_from_gsheet(gs_manager, sheet_name, is_kv_sheet=False):
    if not gs_manager:
        logic_logger.warning(f"GSheet manager not available, cannot fetch {sheet_name}")
        return pd.DataFrame()
    if not sheet_name:
        logic_logger.warning("Sheet name not provided for fetching from GSheet")
        return pd.DataFrame()
    
    try:
        # Prioritize get_data_from_sheet if available
        if hasattr(gs_manager, 'get_data_from_sheet'):
            df = gs_manager.get_data_from_sheet(sheet_name, is_kv_sheet=is_kv_sheet)
        elif hasattr(gs_manager, 'read_sheet'): # Fallback
            df = gs_manager.read_sheet(sheet_name)
            if is_kv_sheet and not df.empty and len(df.columns) >= 2:
                # Basic conversion if read_sheet doesn't handle is_kv_sheet
                # This assumes first col is key, second is value.
                # Consider making get_data_from_sheet the standard.
                pass # Already a DataFrame, user of this function should handle KV format if necessary
        else:
            logic_logger.error(f"GoogleSheetsManager has no suitable method (get_data_from_sheet or read_sheet) to fetch '{sheet_name}'.")
            return pd.DataFrame({"Error": [f"No suitable read method in GSheetManager for {sheet_name}"]})

        if df is None: # Explicitly check for None, as empty DataFrame is valid
            logic_logger.warning(f"No data returned (None) from GSheet: {sheet_name}")
            return pd.DataFrame()
        
        logic_logger.info(f"Successfully fetched data from GSheet: {sheet_name} (shape: {df.shape if not df.empty else 'empty'})")
        return df
        
    except Exception as e:
        logic_logger.error(f"Error fetching {sheet_name} from GSheet: {e}", exc_info=True)
        return pd.DataFrame({"Error": [str(e)]})

# for testing purposes(initial build)
def test_gsheet_connection(gs_manager, sheet_id):
    """Test Google Sheets connection with fallback methods."""
    if not gs_manager:
        logic_logger.warning("GSheet manager not available for connection test.")
        return False
    try:
        if hasattr(gs_manager, 'test_connection'):
            return gs_manager.test_connection() 

        # Fallback to trying to get spreadsheet properties
        if hasattr(gs_manager, 'service') and gs_manager.service and sheet_id:
            spreadsheet = gs_manager.service.spreadsheets().get(spreadsheetId=sheet_id).execute()
            title = spreadsheet.get('properties', {}).get('title')
            logic_logger.info(f"GSheet connection test: successfully accessed sheet '{title}'.")
            return True
        elif hasattr(gs_manager, 'gc') and gs_manager.gc and sheet_id: # gspread client
            spreadsheet = gs_manager.gc.open_by_key(sheet_id)
            logic_logger.info(f"GSheet connection test: successfully accessed sheet '{spreadsheet.title}' via gspread.")
            return True
        else:
            logic_logger.info("GSheet connection test: No specific test method, basic GSheetManager object exists.")
            return True 
            
    except Exception as e:
        logic_logger.error(f"GSheet connection test failed: {e}")
        return False