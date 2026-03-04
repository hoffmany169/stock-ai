from tkinter import messagebox
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
        avg_price = ()
        total_volume = ()

    Interval = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
    def __init__(self, ticker_symbol, load_data=None, start_date=None, end_date=None):
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
        if load_data is not None:
            self._extracted_features()

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
    def interval(self):
        return self._interval
    @interval.setter
    def interval(self, intv):
        self._interval = intv

    @property
    def loaded_data(self):
        return self._loaded_data.copy()
    @loaded_data.setter
    def loaded_data(self, data):
        if data is not None:
            self._loaded_data = data
            self._extracted_features()


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

    def get_feature_value(self, feature: FEATURE, index:int=None):
        """获取股票特征值"""
        if self._loaded_data is None:
            raise ValueError("No data loaded for this stock model.")
        if index is None or index >= len(self._loaded_data):
            raise IndexError("Index out of range for loaded data.")
        return self._loaded_data.get(feature.name, None).iloc[index]

    def get_ext_feature(self, ext_feature : ExtendFeature, index:int=None):
        """获取股票属性值"""
        if index is not None:
            feature_data = self._extend_features.get(ext_feature, None)
            if feature_data is not None and hasattr(feature_data, '__len__') and index < len(feature_data):
                return feature_data.iloc[index]
        return self._extend_features.get(ext_feature, None)

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
            return True
        except Exception as e:
            print(f"Error downloading data for {self.ticker_symbolicker}: {e}")
            messagebox.showerror("Data Download Error",
                                 f"Error downloading data for {self.ticker_symbolticker}: {e}")
            return False

    def load_model_data_from_disk(self, data_dir):
        from ModelIO import ModelSaverLoader
        from StockDefine import MODEL_TRAIN_DATA
        mio = ModelSaverLoader(data_dir, 
                               ticker_symbol=self._ticker_symbol,
                               save=False)
        result = mio.load_train_data()
        if result[MODEL_TRAIN_DATA.ticker_data]:
            self._loaded_data = mio.get_model_train_data(MODEL_TRAIN_DATA.ticker_data)
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
