import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from src.logger_setup import setup_logger 

class DataFetcher:
  

    def __init__(self, api_key=None): 
 
        self.api_key = api_key
        self.logger = setup_logger('DataFetcher') # <--Use consistent logger naming

        """sample NIFTY 50 stock symbol. The main scripts will use stocks from config.yaml"""
        self.nifty_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'BHARTIARTL.NS', 
            'ITC.NS', 'SBIN.NS', 'KOTAKBANK.NS', 
            'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS', 'LT.NS', 'HCLTECH.NS',
            'WIPRO.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'AXISBANK.NS'
        ] 

          

    def fetch_stock_data(self, symbol, period=None, interval='1d', start_date=None, end_date=None):
        """
        Fetch stock data using Yahoo Finance
        """
        try:
            self.logger.info(f"Fetching data for {symbol} (Interval: {interval}, Period: {period}, Start: {start_date}, End: {end_date})")
            ticker = yf.Ticker(symbol)

            if start_date and end_date:
                # yfinance end_date is exclusive for daily data, so add a day if it's a string
                if isinstance(end_date, str) and len(end_date) == 10: # YYYY-MM-DD
                    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                    end_date_fetch = end_date_dt.strftime('%Y-%m-%d')
                elif isinstance(end_date, datetime):
                    end_date_fetch = (end_date + timedelta(days=1))
                else:
                    end_date_fetch = end_date
                data = ticker.history(start=start_date, end=end_date_fetch, interval=interval)
            elif start_date:
                data = ticker.history(start=start_date, interval=interval)
            elif period:
                data = ticker.history(period=period, interval=interval)
            else: # Default if nothing specified (e.g., fetch last year)
                self.logger.warning(f"No period or date range specified for {symbol}. Fetching period '1y'.")
                data = ticker.history(period='1y', interval=interval)


            if data.empty:
                self.logger.warning(f"No data found for {symbol} with given parameters.")
                return pd.DataFrame()

            # Clean and standardize data
            data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True) # Drop if core values are NaN
            data.index = pd.to_datetime(data.index).tz_localize(None) # Ensure datetime index, remove timezone

            # Standardize column names (IMPORTANT for downstream modules)
            data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }, inplace=True)

            # Ensure adj_close exists, if not, use close
            if 'adj_close' not in data.columns and 'close' in data.columns:
                data['adj_close'] = data['close']
                self.logger.info(f"'Adj Close' not found for {symbol}, using 'close' as 'adj_close'.")

            # Ensure all expected columns are present, even if as NaN 
            expected_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in expected_cols:
                if col not in data.columns:
                    data[col] = np.nan # Or 0, depends on how indicators handle it
                    self.logger.warning(f"Column '{col}' was missing for {symbol}, filled with NaN.")

            data = data[expected_cols] # Select and order expected columns

            self.logger.info(f"Successfully fetched {len(data)} records for {symbol} between {data.index.min().date()} and {data.index.max().date()}")
            return data

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def fetch_multiple_stocks(self, symbols, period=None, interval='1d', start_date=None, end_date=None):
        data_dict = {}
        for symbol in symbols:
            data = self.fetch_stock_data(symbol, period=period, interval=interval, start_date=start_date, end_date=end_date)
            if not data.empty:
                data_dict[symbol] = data
            time.sleep(0.2) 

        if data_dict:
            self.logger.info(f"Successfully fetched data for {len(data_dict)} out of {len(symbols)} requested symbols.")
        else:
            self.logger.warning(f"Failed to fetch data for any of the requested symbols: {symbols}")
        return data_dict

    def fetch_nifty_stocks(self, count=5, period='1y', interval='1d'):
      
        if count > len(self.nifty_symbols):
            self.logger.warning(f"Requested count {count} is more than available Nifty symbols ({len(self.nifty_symbols)}). Fetching all.")
            count = len(self.nifty_symbols)

        selected_symbols = self.nifty_symbols[:count]
        self.logger.info(f"Fetching data for {count} NIFTY stocks: {selected_symbols}")
        return self.fetch_multiple_stocks(selected_symbols, period=period, interval=interval)

    def get_latest_price(self, symbol):
        
        try:
            # Fetch 1m data for the last trading day. '1d' period for '1m' interval gets recent data.
            data = self.fetch_stock_data(symbol, period='2d', interval='1m') # Fetch 2 days to ensure we get latest if market just closed
            if not data.empty:
                latest_price = float(data['close'].iloc[-1])
                self.logger.info(f"Latest price for {symbol}: {latest_price} at {data.index[-1]}")
                return latest_price
            else:
                self.logger.warning(f"Could not get 1m data to determine latest price for {symbol}")
                # Fallback to daily close if 1m fails
                daily_data = self.fetch_stock_data(symbol, period='5d', interval='1d')
                if not daily_data.empty:
                    latest_daily_price = float(daily_data['close'].iloc[-1])
                    self.logger.info(f"Using latest daily close for {symbol}: {latest_daily_price} at {daily_data.index[-1].date()}")
                    return latest_daily_price
                return None
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
#Function to get the market status with better time handling 
    def get_market_status(self, market_tz='Asia/Kolkata', open_time='09:15', close_time='15:30'):
        
        try:
            # For timezone handling, importing pytz
            import pytz

            now_utc = datetime.now(pytz.utc) 
            market_timezone = pytz.timezone(market_tz)
            now_market_time = now_utc.astimezone(market_timezone)

            market_open_hour, market_open_minute = map(int, open_time.split(':'))
            market_close_hour, market_close_minute = map(int, close_time.split(':'))

            market_open_dt = now_market_time.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
            market_close_dt = now_market_time.replace(hour=market_close_hour, minute=market_close_minute, second=0, microsecond=0)

            is_weekday = now_market_time.weekday() < 5  # Monday = 0, Friday = 4
            is_market_hours = market_open_dt <= now_market_time <= market_close_dt

            is_open = is_weekday and is_market_hours

            status = {
                'is_open': is_open,
                'current_market_time': now_market_time.strftime('%Y-%m-%d %H:%M:%S %Z%z'),
                'market_opens_at': market_open_dt.strftime('%H:%M:%S'),
                'market_closes_at': market_close_dt.strftime('%H:%M:%S'),
                'is_weekday': is_weekday,
                'note': 'This check does not account for public holidays.'
            }

            self.logger.info(f"Market status ({market_tz}): {'Open' if is_open else 'Closed'}. Current time: {status['current_market_time']}")
            return status
        except Exception as e:
            self.logger.error(f"Error getting market status: {e}")
            return {'is_open': False, 'error': str(e)}


if __name__ == '__main__':
    fetcher = DataFetcher()
