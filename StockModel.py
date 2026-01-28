from tkinter import messagebox
import yfinance as yf

class StockModel:
    def __init__(self, ticker_symbol, load_data=None):
        self._ticker_symbol = ticker_symbol
        self._start_date = '2024-01-01'
        self._end_date = '2025-12-31'
        self._loaded_data = load_data
        self._model = None
        self._ticker_directory_on_disk = None

#region properties    
    @property
    def ticker_symbol(self):
        return self._ticker_symbol
    
    @property
    def start_date(self):
        return self._start_date
    @start_date.setter
    def start_date(self, date:str):
        self._start_date = date

    @property
    def end_date(self):
        return self._end_date
    @end_date.setter
    def end_date(self, date:str):
        self._end_date = date
            
    @property
    def loaded_data(self):
        return self._loaded_data.copy()
    @loaded_data.setter
    def loaded_data(self, data):
        self._loaded_data = data

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, model):
        self._model = model

    @property
    def ticker_directory(self):
        return self._ticker_directory_on_disk
    @ticker_directory.setter
    def ticker_directory(self, dir):
        self._ticker_directory_on_disk = dir
#endregion properties

    def load_historical_data(self, start_date:str=None, end_date:str=None):
        """load historical data for a single ticker from yfinance"""
        if self._start_date is None and start_date is None:
            raise ValueError("start date is not set, yet!")
        if start_date is not None:
            self._start_date = start_date
        if self._end_date is None and end_date is None:
            raise ValueError("end date is not set, yet!")
        if end_date is not None:
            self._end_date = end_date
        print(f"!! Loading Ticker {self.ticker_symbol}: {self._start_date} - {self._end_date} !!")
        try:
            self._loaded_data = yf.Ticker(self._ticker_symbol).history(start=self._start_date, end=self._end_date)
            print(f"Loaded history data of ticker [{self._ticker_symbol}]")
            return True
        except Exception as e:
            print(f"Error downloading data for {self.ticker_symbol}: {e}")
            messagebox.showerror("Data Download Error", 
                                 f"Error downloading data for {self.ticker_symbol}: {e}")
            return False

    def download_ticker_data(self, start_date, end_date):
        if self._start_date is None and start_date is None:
            raise ValueError("start date is not set, yet!")
        if start_date is not None:
            self._start_date = start_date
        if self._end_date is None and end_date is None:
            raise ValueError("end date is not set, yet!")
        if end_date is not None:
            self._end_date = end_date
        try:
            self._loaded_data = yf.download(self.ticker_symbol, 
                                start=self._start_date, 
                                end=self._end_date,
                                group_by='ticker',
                                progress=True,  # 显示进度
                                timeout=60     # 延长超时时间
                                # threads=True    # 使用多线程
                                )
            return True
        except Exception as e:
            print(f"Error downloading data for {self.ticker_symbolicker}: {e}")
            messagebox.showerror("Data Download Error",
                                 f"Error downloading data for {self.ticker_symbolticker}: {e}")
            return False

