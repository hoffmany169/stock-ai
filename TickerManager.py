import os
from tkinter import messagebox
import ModelTrainLSTM
import yfinance as yf
import numpy as np
import pandas as pd
from StockDefine import FEATURE, TICKER, StockFeature
from StockModel import StockModel
from ModelTrainLSTM import LSTMModelTrain
from Common.AutoNumber import AutoIndex
class TickerData(AutoIndex):
    stock_model = ()
    ltsm_model_train = ()

class TickerManager:
    DefaultSaveDataDirectory = r'./models'
    def __init__(self, start_date = None, end_date = None, features:list=[], interval='1d'):
        self.tickers = {}
        self._start_date = start_date
        self._end_date = end_date
        self._stock_features = features
        self._interval = interval

    @staticmethod
    def set_save_data_directory(dir):
        TickerManager.DefaultSaveDataDirectory = dir

#region properties
    @property
    def start_date(self):
        return self._start_date
    @start_date.setter
    def start_date(self, value):
        self._start_date = value

    @property
    def end_date(self):
        return self._end_date
    @end_date.setter
    def end_date(self, value):
        self._end_date = value

    @property
    def stock_features(self):
        return self._stock_features
    @stock_features.setter
    def stock_features(self, feats):
        self._stock_features = feats

    @property
    def interval(self):
        return self._interval
    @interval.setter
    def interval(self, intvl):
        self._interval = intvl

    @property
    def ticker_list(self):
        return list(self.tickers.keys())
#endregion properties

    # def get_ticker_list(self):
    #     return list(self.tickers.keys())

    def add_ticker(self, ticker_symbol):
        """Add a ticker to the manager."""
        if ticker_symbol in self.tickers:
            print(f"Ticker [{ticker_symbol}] is added already.")
            return False
        print(f'Adding Stock: [{ticker_symbol}]')
        self.tickers[ticker_symbol] = {}
        sm = StockModel(ticker_symbol)
        # initialize stock model and lstm model train
        self.tickers[ticker_symbol][TickerData.stock_model] = sm
        self.tickers[ticker_symbol][TickerData.ltsm_model_train] = LSTMModelTrain(sm, self._stock_features)
        return True
    
    def remove_ticker(self, ticker:str|int):
        """Remove a ticker from the manager."""
        if type(ticker) is int: # index of ticker
            if ticker >= 0 and ticker < len(self.ticker_list):
                rmv_ticker = self.ticker_list[ticker]
            else:
                return None
        else:
            rmv_ticker = ticker
        if rmv_ticker not in self.tickers:
            return None
        return self.tickers.pop(rmv_ticker, None)

    def clear_all(self):
        self.tickers.clear()

    def get_all_tickers(self):
        """Return a list of all tickers managed."""
        return list(self.tickers.keys())

    def get_stock_model(self, ticker):
        if type(ticker) is str: 
            # ticker symbol
            for ticker_symbol, m in self.tickers.items():
                if ticker == ticker_symbol:
                    return self.tickers[ticker_symbol][TickerData.stock_model]
        elif type(ticker) is int:
            for i, ticker_symbol in enumerate(self.ticker_list):
                if i == ticker:
                    return self.tickers[ticker_symbol][TickerData.stock_model]
        else:
            raise ValueError("Data type of ticker is not supported!")

    def get_LSTM_model_train(self, ticker):
        if type(ticker) is str: 
            # ticker symbol
            for ticker_symbol, m in self.tickers.items():
                if ticker == ticker_symbol:
                    return self.tickers[ticker_symbol][TickerData.ltsm_model_train]
        elif type(ticker) is int:
            for i, ticker_symbol in enumerate(self.ticker_list):
                if i == ticker:
                    return self.tickers[ticker_symbol][TickerData.ltsm_model_train]
        else:
            raise ValueError("Data type of ticker is not supported!")

    def load_ticker_data(self, tickers:list, start_date:str, end_date:str, features, interval:str):
        self.start_date = start_date
        self.end_date = end_date
        no_data = []
        
        if not tickers:
            print("没有股票代码可下载")
            return
        
        print(f"开始下载 {len(tickers)} 个股票的数据...")
        
        try:
            # 方法1: 批量下载
            print(f"下载参数: start={self.start_date}, end={self.end_date}")
            
            # 确保日期格式正确
            start_str = str(self.start_date)
            end_str = str(self.end_date)
            
            if len(tickers) > 1:
                # 使用更稳定的下载方式
                all_data = yf.download(
                    tickers, 
                    start=start_str, 
                    end=end_str, 
                    group_by='ticker',
                    progress=True,  # 显示进度
                    timeout=60,     # 延长超时时间
                    threads=True    # 使用多线程
                )
                print(f"下载完成，数据形状: {all_data.shape}")
                # 处理数据
                batch_download_result = True
                for ticker in tickers:
                    try:
                        if ticker in all_data:
                            data_num = len(all_data[ticker])
                            if data_num > 1:
                                # create stock model object if download is successful
                                sm = self.get_stock_model(ticker)
                                sm.interval = interval
                                sm.loaded_data = all_data[ticker]
                                sm.start_date = start_str
                                sm.end_date = end_str
                                sm.save_model_data_to_disk()
                                self.get_LSTM_model_train(ticker).features = features
                                print(f"{ticker}: 成功加载 {len(all_data[ticker])} 条数据")
                            else:
                                batch_download_result = False
                        else:
                            batch_download_result = False
                        if batch_download_result == False:
                            batch_download_result = True # reset
                            print(f"{ticker}: 批量下载失败，尝试单独下载...")
                            sm = self.get_stock_model(ticker)
                            sm.load_historical_data(start_str, end_str)
                            if len(sm.loaded_data) <= 1:
                                raise ValueError("Error:", "Downloading data from markt fails")
                    except Exception as e:
                        print(f"{ticker}: 处理失败 - {str(e)}")
                        no_data.append(ticker)
                        continue
            else:
                sm = self.get_stock_model(ticker)
                sm.download_ticker_data()
        except Exception as e:
            print(f"批量下载失败: {str(e)}")
            print("尝试逐个下载...")
            
            # 方法2: 逐个下载
            for ticker in tickers:
                sm = self.get_stock_model(ticker)
                sm.download_ticker_data(self.start_date, self.end_date)
                if len(sm.loaded_data) <= 1:
                    no_data.append(ticker)
        print(f"\n下载完成: {len(tickers) - len(no_data)}/{len(tickers)} 个股票数据下载成功")
        return no_data

    # def single_load_ticker_data(self, ticker, start_date, end_date):
    #     try:
    #         sm = self.get_stock_model(ticker)
    #         ret = sm.load_historical_data(start_date, end_date)
            
    #         if ret:
    #             print(f"{ticker}: 成功单独下载 {len(sm.loaded_data)} 条数据")
    #             self.add_ticker(ticker)
    #             return True
    #         else:
    #             print(f"{ticker}: 无数据")
    #             return False
    #     except Exception as e:
    #         print(f"{ticker}: 下载失败 - {str(e)}")
    #         return False

    def process_train_model(self, lookback=60):
        for ticker in self.ticker_list:
            mt = self.get_LSTM_model_train(ticker)
            mt.lookback = lookback
            if mt.features is None or len(mt.features) == 0:
                if len(self._stock_features) == 0:
                    raise ValueError("features are not defined, yet!")
                else:
                    mt.features = self._stock_features
            mt.process_train_data()
    
    def save_train_data(self, ticker_symbol):
        sm = self.get_stock_model(ticker_symbol)
        sm.save_model_data_to_disk()
        ss = self.get_LSTM_model_train(ticker_symbol)
        ss.save_model_train_data_to_disk()

    def process_save_train_data(self, path):
        for ticker in self.ticker_list:
            self.save_train_data(ticker, path)

    def _parse_models_directory(self, directory):
        # parse ticker name
        base_name = os.path.basename(directory)
        parts = base_name.split('_')
        print(parts)
        ticker_symbol = parts[0]
        # parse parent directory
        parent_directory = directory.split(base_name)[0]
        print(f"ticker: {ticker_symbol}, directory: {parent_directory}")
        return (parent_directory, ticker_symbol)

    def process_load_train_data(self, data_dir):
        """
        Docstring for process_load_train_data
        
        :param self: Description
        :param data_dir: concrete path of ticker data, for example: TSLA_20260126_174853 if ticker is True, otherwise, it is parent directory of ticker data
        :param ticker: bool, False - data_dir is a parent directory; True - it is a concrete directory of ticker data
        """
        from ModelIO import ModelSaverLoader
        from StockDefine import MODEL_TRAIN_DATA
        models_dir, ticker_symbol = self._parse_models_directory(data_dir)
        # mio = ModelSaverLoader(data_dir,
        #                         save=False)
        self.add_ticker(ticker_symbol)
        sm = self.get_stock_model(ticker_symbol)
        sm.model_save_path = data_dir
        sm.load_model_data_from_disk()
        ss = self.get_LSTM_model_train(ticker_symbol)
        ss.load_model_train_data_to_disk()
        
    def select_stocks(self, start_date, end_date, lookback=60, selction_threshold=0.7):
        """
        period : str
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        Either Use period parameter or use start and end
        interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        Intraday data cannot extend last 60 days
        """
        print("!! selecting stocks !!")
        selected_stocks = []
        current_data = yf.download(self.ticker_list, start=start_date, end=end_date, group_by='ticker')
        for ticker in self.ticker_list:
            ticker_data = current_data[ticker]
            local_ss = StockModel(ticker, ticker_data)
            # get trained model
            local_ss.model = self.get_stock_model(ticker).model
            local_lstm = LSTMModelTrain(local_ss, self.get_LSTM_model_train(ticker).features, lookback)
            ss = self.get_LSTM_model_train(ticker)
            prediction = local_lstm.process_selection(ss.scaler)
            print(f'prediction: [{prediction}] <--> {selction_threshold}')
            if prediction > selction_threshold:  # 设置较高阈值
                selected_stocks.append(ticker)
        if (len(selected_stocks) > 0):
            print(selected_stocks)
            messagebox.showinfo("Select Stocks", f"Selected Stocks: {selected_stocks}")
        else:
            print("No ticker are selected")
            messagebox.showinfo("Select Stocks", "No stocks is selectable.")


if __name__ == "__main__":
    manager = TickerManager(start_date="2024-01-01", end_date="2025-01-01", lookback=60)
    manager.add_ticker("TSLA")
    manager.add_ticker("NVDA")
    manager.selector.disable_feature(FEATURE.PB)
    manager.selector.disable_feature(FEATURE.PE)

    if not manager.load_ticker_data():
        print("Failed to load ticker data.")
        exit(1)
    manager.process_select_stocks()
    manager.select_stocks(180, lookback=60, prediction_threshold=0.7)
    selected_stocks = manager.get_selected_stocks()
    if len(selected_stocks) > 0:
        print("Final selected stocks:", )
    else:
        print("No selected stocks.")