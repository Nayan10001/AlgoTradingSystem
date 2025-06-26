
import os
import gspread
from google.oauth2.service_account import Credentials 
import pandas as pd
from datetime import datetime
from src.logger_setup import setup_logger 
from dotenv import load_dotenv



#logging setup
logger = setup_logger(__name__)
load_dotenv() 

class GoogleSheetsManager:
    """
    Manages Google Sheets operations for the trading system.
    Handles authentication, worksheet creation, data logging, and retrieval.
    """

    # Define scopes for Google APIs
    SCOPE = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file" ]

    def __init__(self, creds_path=None, spreadsheet_id=None):
        """
        Initialize Google Sheets connection.
        """
        self.logger = setup_logger('GSheetManager') # Consistent logger name

        self.creds_path = creds_path or os.getenv("GOOGLE_SHEETS_CREDENTIALS_FILE")
        self.spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SHEET_ID")

        if not self.creds_path:
            self.logger.critical("Google Sheets credentials file path not provided or found in .env.")
            raise ValueError("Missing Google Sheets credentials file path.")
        if not self.spreadsheet_id:
            self.logger.critical("Google Sheets ID not provided or found in .env.")
            raise ValueError("Missing Google Sheets ID.")

        self.client = None
        self.spreadsheet = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Google Sheets API using google-auth."""
        try:
            creds = Credentials.from_service_account_file(self.creds_path, scopes=self.SCOPE)
            self.client = gspread.authorize(creds)
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            self.logger.info(f"Successfully authenticated with Google Sheets and opened spreadsheet ID: {self.spreadsheet_id[:10]}...")
        except FileNotFoundError:
            self.logger.error(f"Google Sheets credentials JSON file not found at: {self.creds_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to authenticate/open Google Sheet: {e}", exc_info=True)
            raise

    def _get_or_create_worksheet(self, ws_name, headers=None, rows=1000, cols=25):
        """
        Internal helper to get or create a worksheet and set headers if new.
        """
        try:
            worksheet = self.spreadsheet.worksheet(ws_name)
            self.logger.debug(f"Worksheet '{ws_name}' found.")
        except gspread.exceptions.WorksheetNotFound:
            self.logger.info(f"Worksheet '{ws_name}' not found. Creating it...")
            actual_cols = len(headers) if headers else cols
            worksheet = self.spreadsheet.add_worksheet(title=ws_name, rows=int(rows), cols=int(actual_cols))
            if headers:
                worksheet.update('A1', [headers], value_input_option='USER_ENTERED')
                self.logger.info(f"Created worksheet: '{ws_name}' and set headers.")
            else:
                self.logger.info(f"Created worksheet: '{ws_name}'.")
        return worksheet

    def create_project_worksheets(self, worksheet_configs=None):
        """
        Create standard project worksheets if they don't exist, based on a config.
        This can be called once after initializing the manager.
        """
        if worksheet_configs is None:
            worksheet_configs = {
                'TradesLog': [ 
                    'Log_Timestamp', 'Symbol', 'Entry_Date', 'Signal_Action', 'Entry_Price',
                    'Exit_Date', 'Exit_Price', 'PnL_Percent', 'Signal_Strength', 'Reason',
                    'RSI_at_Entry', 'MA_Ratio_at_Entry' 
                ],
                'DailySummary': [ 
                    'Date', 'Total_Signals_Generated', 'Buy_Signals', 'Sell_Signals',
                    'Portfolio_Value_Est', 'Daily_PnL_Est', 'Notes'
                ],
                'BacktestSummary': [ 
                    'Run_Timestamp', 'Total_Stocks_Tested', 'Overall_PnL_Sum_Pct',
                    'Overall_Win_Rate_Pct', 'Total_Completed_Trades', 'Avg_PnL_Per_Trade_Pct', 'Notes'
                ],
                'ML_Predictions': [
                    'Log_Timestamp', 'Symbol', 'Prediction_For_Date', 'Predicted_Movement',
                    'Confidence_UP', 'Model_Test_Accuracy', 'Features_Used'
                ]
            }
        self.logger.info("Checking and creating project worksheets...")
        for ws_name, headers in worksheet_configs.items():
            try:
                self._get_or_create_worksheet(ws_name, headers=headers)
            except Exception as e:
                self.logger.error(f"Error ensuring worksheet '{ws_name}' exists: {e}")

    def log_trades_from_dataframe(self, trades_df, sheet_name='TradesLog', clear_sheet=False):
        """
        Logs a DataFrame of trades/data to the specified worksheet.
        Can append or clear and write. Robustly handles header alignment for append mode.
        """
        if trades_df is None or trades_df.empty:
            self.logger.info(f"Input DataFrame for '{sheet_name}' is empty. Nothing to log.")
            return True

        try:
            # Ensures sheet exists; if created new, _get_or_create_worksheet might set headers
            # based on its default logic if it were called with headers param.
            # Here, we pass no headers, so it just creates an empty sheet if not found.
            worksheet = self._get_or_create_worksheet(sheet_name) 

            df_cleaned = trades_df.copy()
            
            for col in df_cleaned.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetimetz']).columns:
                df_cleaned[col] = df_cleaned[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Handle Log_Timestamp: if a column named 'Log_Timestamp' exists and is entirely empty strings
            # (which can happen after .fillna('').astype(str)), then fill it.
            # This check is basic; iloc[0] assumes if the first is empty, all are or it's a new batch.
            if 'Log_Timestamp' in df_cleaned.columns and not df_cleaned.empty and df_cleaned['Log_Timestamp'].iloc[0] == '':
                 # A more robust check for all empty or all NaN before fillna might be needed for generic DFs.
                 # For current use cases (single ML row, or trades_df where timestamp is added before), this is okay.
                 if df_cleaned['Log_Timestamp'].replace('', pd.NA).isnull().all():
                    df_cleaned['Log_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


            current_sheet_headers = worksheet.row_values(1) if worksheet.row_count > 0 else []

            if clear_sheet:
                worksheet.clear()
                self.logger.info(f"Cleared sheet '{sheet_name}'.")
                # Use df_cleaned's headers as the new headers for the sheet
                headers_to_write = df_cleaned.columns.tolist()
                worksheet.update('A1', [headers_to_write], value_input_option='USER_ENTERED')
                # Data needs to be list of lists, and stringified
                data_values = df_cleaned.fillna('').astype(str).values.tolist()
                if data_values: # worksheet.append_rows needs non-empty list of lists
                    # Using append_rows after setting headers to add data below the header row
                    worksheet.append_rows(data_values, value_input_option='USER_ENTERED') 
                self.logger.info(f"Logged {len(df_cleaned)} rows to '{sheet_name}' (overwritten).")
            else: # Append mode
                if not current_sheet_headers:
                    # Sheet is empty (or headers couldn't be read, e.g. a truly empty sheet from start)
                    # Use df_cleaned's headers as the new headers for the sheet
                    headers_to_write = df_cleaned.columns.tolist()
                    worksheet.update('A1', [headers_to_write], value_input_option='USER_ENTERED')
                    current_sheet_headers = headers_to_write # These are now the effective headers
                    self.logger.info(f"Sheet '{sheet_name}' was empty. Set headers from DataFrame: {headers_to_write}")
                
                # Align df_cleaned to current_sheet_headers for appending
                # Create a new DataFrame with columns in the order of current_sheet_headers
                aligned_df = pd.DataFrame(columns=current_sheet_headers)
                for col in current_sheet_headers:
                    if col in df_cleaned.columns:
                        aligned_df[col] = df_cleaned[col]
                    else:
                        # If a sheet header is not in df_cleaned, it will be filled with pd.NA
                        # which then becomes '' by fillna('').astype(str)
                        aligned_df[col] = pd.NA 

                data_values = aligned_df.fillna('').astype(str).values.tolist()
                if data_values:
                    worksheet.append_rows(data_values, value_input_option='USER_ENTERED')
                    self.logger.info(f"Appended {len(data_values)} rows to '{sheet_name}'.")
                else:
                    self.logger.info(f"No data to append to '{sheet_name}' after alignment (append mode).")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging DataFrame to '{sheet_name}': {e}", exc_info=True)
            return False

    def update_key_value_sheet(self, data_dict, sheet_name, clear_sheet=True):
        """
        Updates a sheet with key-value pairs (e.g., summary, analytics).
        Each key-value pair becomes a row.
        """
        if not data_dict:
            self.logger.info(f"Data dictionary for '{sheet_name}' is empty. Nothing to update.")
            return True

        try:
            headers = ["Metric", "Value", "Last_Updated"]

            # Step 1: Get worksheet (create if needed)
            worksheet = self._get_or_create_worksheet(sheet_name)

            # Step 2: If clear_sheet is True or worksheet is empty, apply headers
            if clear_sheet or not (worksheet.row_count > 0 if hasattr(worksheet, 'row_count') else False):
                worksheet.update('A1', [headers], value_input_option='USER_ENTERED')
                self.logger.info(f"Set headers for sheet '{sheet_name}'")

            if clear_sheet:
                worksheet.clear()
                worksheet.update('A1', [headers], value_input_option='USER_ENTERED')
                self.logger.info(f"Cleared sheet '{sheet_name}' for key-value update.")

            # Prepare rows
            rows_to_update = []
            timestamp_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for key, value in data_dict.items():
                if isinstance(value, float): 
                    value_str = f"{value:.2f}"
                elif isinstance(value, pd.Timestamp): 
                    value_str = value.strftime('%Y-%m-%d %H:%M:%S')
                else: 
                    value_str = str(value)
                rows_to_update.append([str(key), value_str, timestamp_now])

            if rows_to_update:
                if clear_sheet:
                    worksheet.update('A2', rows_to_update, value_input_option='USER_ENTERED')
                else:
                    worksheet.append_rows(rows_to_update, value_input_option='USER_ENTERED')

                self.logger.info(f"Updated '{sheet_name}' with {len(rows_to_update)} key-value pairs.")

            return True

        except Exception as e:
            self.logger.error(f"Error updating key-value sheet '{sheet_name}': {e}", exc_info=True)
            return False
    def get_data_as_dataframe(self, sheet_name):
        """
        Retrieve all data from a worksheet as a Pandas DataFrame.
        """
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name) 
            records = worksheet.get_all_records() 
            if records:
                df = pd.DataFrame(records)
                df = df.infer_objects()
                for col in df.columns:
                    if 'date' in col.lower() or 'timestamp' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except Exception:
                            pass 
                self.logger.info(f"Retrieved {len(df)} records from '{sheet_name}'.")
                return df
            else:
                self.logger.info(f"No records found in worksheet '{sheet_name}'.")
                return pd.DataFrame()
        except gspread.exceptions.WorksheetNotFound:
            self.logger.warning(f"Worksheet '{sheet_name}' not found for retrieval.")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error retrieving data from '{sheet_name}': {e}", exc_info=True)
            return pd.DataFrame()

    def backup_data_to_csv(self, backup_path='gsheet_backups'):
        """Backup all worksheet data to local CSV files."""
        try:
            if not os.path.exists(backup_path):
                os.makedirs(backup_path)
                self.logger.info(f"Created backup directory: {backup_path}")

            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            for worksheet in self.spreadsheet.worksheets():
                df = self.get_data_as_dataframe(worksheet.title) 
                if not df.empty:
                    filename = f"{backup_path}/{worksheet.title}_{timestamp_str}.csv"
                    df.to_csv(filename, index=False)
                    self.logger.info(f"Backed up worksheet '{worksheet.title}' to {filename}")
                else:
                    self.logger.info(f"Worksheet '{worksheet.title}' is empty, skipping backup.")
            return True
        except Exception as e:
            self.logger.error(f"Error during data backup: {e}", exc_info=True)
            return False

    #For testing Period(inital build)
    # def test_connection(self):
    #     """Test method to verify Google Sheets connection and basic operations."""
    #     try:
    #         self.logger.info(f"Testing connection to spreadsheet: {self.spreadsheet.title}")
            
    #         test_data = pd.DataFrame({
    #             'Test_Column': ['Test_Value_1', 'Test_Value_2'], 
    #             'Timestamp': [datetime.now(), datetime.now()]
    #         })
            
    #         # Test with a sheet name that won't conflict, and ensure it clears for the test
    #         success = self.log_trades_from_dataframe(test_data, 'ConnectionTestSheet', clear_sheet=True)
    #         if success:
    #             self.logger.info("✓ Connection test successful - data written to Google Sheets (ConnectionTestSheet)")
    #             # Optionally, delete the test sheet
    #             # self.spreadsheet.del_worksheet(self.spreadsheet.worksheet('ConnectionTestSheet'))
    #             return True
    #         else:
    #             self.logger.error("✗ Connection test failed - could not write data")
    #             return False
                
    #     except Exception as e:
    #         self.logger.error(f"Connection test failed: {e}", exc_info=True)
    #         return False