from datetime import datetime
import json
import os
from stockDefine import StockFeature
from tkinter import messagebox
import yfinance as yf

class StockModel:
    def __init__(self, ticker_symbol, load_data=None):
        self._ticker_symbol = ticker_symbol
        self._start_date = '2024-01-01'
        self._end_date = '2025-12-31'
        self._loaded_data = load_data
        self._model = None
        # data for processing
        self._train_test_data = None
        self._train_test_split_percentage = 0.2
        self._model_dir = None
        self._readme_content = ''

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
    def readme_content(self):
        return self._readme_content
    @readme_content.setter
    def readme_content(self, val):
        self._readme_content = val
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
        except Exception as e:
            print(f"Error downloading data for {self.ticker_symbol}: {e}")
            messagebox.showerror("Data Download Error", 
                                 f"Error downloading data for {self.ticker_symbol}: {e}")
            return None

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
        except Exception as e:
            print(f"Error downloading data for {self.ticker_symbolicker}: {e}")
            messagebox.showerror("Data Download Error",
                                 f"Error downloading data for {self.ticker_symbolticker}: {e}")
            return None

    def load_data_from_disk(self, path):
        pass

    def create_ticker_folder(self, path):
        try:
            folder_name = f"{self._ticker_symbol}@{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._model_dir = os.path.join(path, folder_name)
            if not os.path.exists(self._model_dir):
                os.makedirs(self._model_dir)
        except Exception as e:
            print(f"Error to create ticker folder for {self._ticker_symbol}: {str(e)}")

    def save_ticker_data(self, data, path='.'):
        self.create_ticker_folder(path)
        if not self._model_dir:
            return False
        data_file = os.path.join(self._model_dir, 'data')
        data.to_pickle(data_file)
        self._save_model_data()
        return True

    def _save_model_data(self, scaler, lookback):
        # 保存特征列表
        model_info = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "lookback": lookback,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features": self.features
        }
        with open(f"{self._model_dir}/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        # 保存缩放器文件
        scaler_file = f"{self._model_dir}/scaler.joblib"
        json.dump(scaler, scaler_file)
        # 保存模型文件
        self.model.save(f"{self._model_dir}/model.h5")
        return True

