import datetime
from tkinter import messagebox
import pandas as pd
import numpy as np
import yfinance as yf
from StockDefine import TICKER_DATA_PARAM
from Common.AutoNumber import AutoIndex


class StockModel:
    class INTERVAL(AutoIndex):
        ONE_MINUT = ()
        TWO_MINUTS = ()
        FIVE_MINUTS = ()
        FIFTEEN_MINUTS = ()
        THIRTY_MINUTS = ()
        SIXTY_MINUTS = ()
        NINETY_MINUTS = ()
        ONE_HOUR = ()
        ONE_DAY = ()
        FIVE_DAYS = ()
        ONE_WEEK = ()
        ONE_MONTH = ()
        THREE_MONTHS = ()

    class FEATURE(AutoIndex):
        Price = ()
        Volume = ()
        Open = ()
        Close = ()
        High = ()
        Low = ()
        # add more features as needed

    # extended features
    class ExtendFeature(AutoIndex):
        high_low_range = ()
        open_close_range = ()
        total_days = ()
        max_price = ()
        min_price = ()
        max_volume = ()
        volume_change = ()
        avg_price = ()
        total_volume = ()

    class IntervalIndex(AutoIndex):
        ITVL_1_minute = ()
        ITVL_2_minute = ()
        ITVL_5_minute = ()
        ITVL_15_minute = ()
        ITVL_30_minute = ()
        ITVL_60_minute = ()
        ITVL_90_minute = ()
        ITVL_1_hour = ()
        ITVL_1_day = ()
        ITVL_5_day = ()
        ITVL_1_week = ()
        ITVL_1_month = ()
        ITVL_3_month = ()

    Interval = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
    def __init__(self, ticker_symbol,
                 load_data=None,
                 start_date=None,
                 end_date=None,
                 model_save_root_path='models',
                 interval='1d'):
        self._ticker_symbol = ticker_symbol
        self._start_date = '2024-01-01' if start_date is None else start_date
        self._end_date = '2025-12-31' if end_date is None else end_date
        self._loaded_data = load_data
        self._model = None
        self._intervals = dict(zip([itv for itv in StockModel.INTERVAL], StockModel.Interval))
        self._ticker_directory_on_disk = None
        self._interval = '1d'
        self._feature_functions = None
        self._extend_features = {}
        self._interval = interval
        self._model_save_path = f'{model_save_root_path}/{ticker_symbol}'
        if load_data is not None:
            self._extracted_features()

#region properties    
    @property
    def ticker_symbol(self)->str:
        return self._ticker_symbol
    
    @property
    def start_date(self)->str:
        return self._start_date
    @start_date.setter
    def start_date(self, date:str):
        self._start_date = date

    @property
    def end_date(self)->str:
        return self._end_date
    @end_date.setter
    def end_date(self, date:str):
        self._end_date = date

    @property
    def interval(self):
        return self._interval
    @interval.setter
    def interval(self, intv:str|IntervalIndex):
        if type(intv) is str:
            self._interval = intv
        else:
            self._interval = self.Interval[intv.value]

    @property
    def start_datetime(self):
        return datetime.date.fromisoformat(self.start_date)
    
    @property
    def end_datetime(self):
        return datetime.date.fromisoformat(self.end_date)

    @property
    def loaded_data(self):
        return self._loaded_data.copy()
    @loaded_data.setter
    def loaded_data(self, data):
        if data is not None:
            self._loaded_data = data
            self.add_date_column()
            self._extracted_features()

    @property
    def model_save_path(self):
        return self._model_save_path
    @model_save_path.setter
    def model_save_path(self, path):
        self._model_save_path = path

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

    @property
    def is_data_loaded(self)->bool:
        return (self._loaded_data is not None)

#endregion properties

    @staticmethod
    def get_interval_index(intvl:str):
        idx = StockModel.Interval.index(intvl)
        if idx is None:
            return None
        for val in StockModel.IntervalIndex:
            if idx == val.value:
                return val
        
    @staticmethod
    def get_timedelta_of_interval(intvl:str|IntervalIndex, factor:int=1):
        if type(intvl) is str:
            intervalindex = StockModel.get_interval_index(intvl)
        else:
            intervalindex = intvl
        if intervalindex.value < 7: # minutes
            parts = intervalindex.name.split('_')
            delta = datetime.timedelta(minutes=int(parts[1])*factor)
        elif intervalindex.value == 7:
            delta = datetime.timedelta(hours=factor)
        elif intervalindex.value < 10:
            parts = intervalindex.name.split('_')
            delta = datetime.timedelta(days=int(parts[1])*factor)
        elif intervalindex.value == 10:
            delta = datetime.timedelta(weeks=factor)
        else:
            parts = intervalindex.name.split('_')
            delta = datetime.timedelta(weeks=4*int(parts[1])*factor)
        return delta
            
#region stock data operations
    def get_interval_text(self):
        idx = self.Interval.index(self._interval)
        for val in self.IntervalIndex:
            if idx == val.value:
                return ' '.join(val.name[5:].split('_'))
    
    def _extracted_features(self):
        """提取股票属性信息"""
        # 提取基本统计信息
        print(self._loaded_data.shape)
        data_num = self._loaded_data.shape[1] if self._loaded_data is not None else 0
        self._extend_features[StockModel.ExtendFeature.total_days] = data_num
        self._extend_features[StockModel.ExtendFeature.max_price] = self._loaded_data['High'].max()
        self._extend_features[StockModel.ExtendFeature.min_price] = self._loaded_data['Low'].min()
        self._extend_features[StockModel.ExtendFeature.max_volume] = self._loaded_data['Volume'].max()
        self._extend_features[StockModel.ExtendFeature.avg_price] = self._loaded_data['Close'].mean()
        self._extend_features[StockModel.ExtendFeature.total_volume] = self._loaded_data['Volume'].sum()
        self._extend_features[StockModel.ExtendFeature.high_low_range] = self._loaded_data['High'] - self._loaded_data['Low']
        self._extend_features[StockModel.ExtendFeature.open_close_range] = self._loaded_data['Open'] - self._loaded_data['Close']
        self._extend_features[StockModel.ExtendFeature.volume_change] = self._loaded_data['Volume'].diff()
        # self._extend_features[StockModel.ExtendFeature.volume_change] = self._loaded_data['Volume'].pct_change().fillna(0)

    def get_feature_value(self, feature: FEATURE, index:int|str=None):
        """获取股票特征值"""
        if self._loaded_data is None:
            raise ValueError("No data loaded for this stock model.")
        feature_data = self._loaded_data.get(feature.name, None)
        if type(index) is int:
            if index is None or index >= len(self._loaded_data):
                raise IndexError("Index out of range for loaded data.")
            if feature_data is not None:
                return feature_data.iloc[index]
        elif type(index) is str and feature_data is not None:
            return feature_data.loc[index]
        
    def get_ext_feature(self, ext_feature : ExtendFeature, index:int=None):
        """获取股票属性值"""
        if index is not None:
            feature_data = self._extend_features.get(ext_feature, None)
            if feature_data is not None and index < len(feature_data):
                return feature_data.iloc[index]
        return self._extend_features.get(ext_feature, None)
    
    def get_data_absolute_index_by_date_range(self, start_date:str, end_date:str, window:int):
        """获取指定日期范围的股票数据"""
        if self._loaded_data is None or self._loaded_data.empty:
            print(f"{self._ticker_symbol} must be loaded at first")
            return None
                
        try:
            # 转换日期范围
            if start_date:
                # convert to DataFrame datetime format
                start_date = pd.to_datetime(start_date)
                # get absolute index from start_date
                start_idx = np.abs(self._loaded_data['Date'] - start_date).argmin()
            else:
                start_idx = 0
            
            if end_date:
                # convert to DataFrame datetime format
                end_date = pd.to_datetime(end_date)
                # get absolute index from end_date
                end_idx = np.abs(self.loaded_data['Date'] - end_date).argmin()
            else:
                end_idx = len(self.loaded_data) - 1
            
            # 确保索引有效
            start_idx = max(window, start_idx)
            end_idx = min(len(self.loaded_data) - window - 1, end_idx)
            return (start_idx, end_idx)
            
        except Exception as e:
            print(f"筛选数据失败: {e}")
            return None
#endregion stock data operations
    @staticmethod
    def get_date_span(self, date1:str, date2:str, fmt='%Y-%m-%d')->int:
        '''
        get span days between two date
        '''
        return datetime.strptime(date2, fmt) - datetime.strptime(date1, fmt)

    def get_feature_value_difference(self, date1:str|int, date2:str|int, feature='Close', percentage=False):
        '''
        compare value of two points and return their changes or percentage of changers
        '''
        value_1 = self.get_feature_value(feature, date1)
        value_2 = self.get_feature_value(feature, date2)
        feature_value_diff = value_2 - value_1
        if percentage:
            percentage_change = (feature_value_diff / value_1 * 100) if value_1 != 0 else float('inf')
            return percentage_change
        else:
            return feature_value_diff

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
            self._extracted_features()
            # add index as a column
            self.add_date_column()
            self.save_model_data_to_disk()
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
            self._extracted_features()
            # add index as a column
            self.add_date_column()
            self.save_model_data_to_disk()
            return True
        except Exception as e:
            print(f"Error downloading data for {self.ticker_symbol}: {e}")
            messagebox.showerror("Data Download Error",
                                 f"Error downloading data for {self.ticker_symbol}: {e}")
            return False

    # add index as a column
    def add_date_column(self):
        '''
        add "Date" column after load stock data from csv file, so that they can be plotted
        '''
        if 'Data' not in self._loaded_data:
            self._loaded_data['Date'] = self._loaded_data.index

    def save_model_data_to_disk(self):
        '''
        write data to disk after assigning them to MODEL_TRAIN_DATA dict by using ModelIO module
        '''
        from ModelIO import ModelSaverLoader
        from StockDefine import MODEL_TRAIN_DATA
        import os
        self.model_save_path = os.path.join(self._model_save_path, 
                                            f'{self._ticker_symbol}_{self.start_date}_{self.end_date}')
        mio = ModelSaverLoader(self._model_save_path,
                               ticker_symbol=self._ticker_symbol)
        mio.set_model_train_data(MODEL_TRAIN_DATA.ticker_data, self._loaded_data)
        mio.set_model_train_data(MODEL_TRAIN_DATA.ticker_data_params, self.create_ticker_parameters())
        mio.set_model_train_data(MODEL_TRAIN_DATA.model, self._model)
        mio.save_train_data(MODEL_TRAIN_DATA.for_stock_model)

    def load_model_data_from_disk(self):
        '''
        load model data from disk and set them to MODEL_TRAIN_DATA
        '''
        from ModelIO import ModelSaverLoader
        from StockDefine import MODEL_TRAIN_DATA
        mio = ModelSaverLoader(self._model_save_path, 
                               ticker_symbol=self._ticker_symbol,
                               save=False)
        result = mio.load_train_data(MODEL_TRAIN_DATA.for_stock_model)
        if result[MODEL_TRAIN_DATA.ticker_data]:
            self._loaded_data = mio.get_model_train_data(MODEL_TRAIN_DATA.ticker_data)
            self.add_date_column()
        if result[MODEL_TRAIN_DATA.ticker_data_params]:
            data_params = mio.get_model_train_data(MODEL_TRAIN_DATA.ticker_data_params)
            self.assign_ticker_params_from_loading(data_params)
        if result[MODEL_TRAIN_DATA.model]:
            self.model = mio.get_model_train_data(MODEL_TRAIN_DATA.model)
        self._extracted_features()
        return result

    def create_ticker_parameters(self):
        return {TICKER_DATA_PARAM.ticker_symbol.name: self._ticker_symbol,
                TICKER_DATA_PARAM.start_date.name: self._start_date,
                TICKER_DATA_PARAM.end_date.name: self._end_date,
                TICKER_DATA_PARAM.interval.name: self._interval}
    
    def assign_ticker_params_from_loading(self, ticker_data:dict):
        self._start_date = ticker_data[TICKER_DATA_PARAM.start_date.name]
        self._end_date = ticker_data[TICKER_DATA_PARAM.end_date.name]
        self._interval = ticker_data[TICKER_DATA_PARAM.interval.name]
        print(f"Loaded ticker parameters: start date [{self._start_date}], end date [{self._end_date}], interval [{self._interval}]")
