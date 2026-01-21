from tkinter import messagebox
import select_stock
import yfinance as yf
import numpy as np
import pandas as pd
from stock import FEATURE, TICKER

class TickerManager:
    def __init__(self, start_date = None, end_date = None, lookback=60):
        self.tickers = {}
        self._start_date = start_date
        self._end_date = end_date
        self._selector = select_stock.LSTM_Select_Stock(lookback)

    @property
    def selector(self):
        return self._selector

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
    def lookback(self):
        return self.selector.lookback
    @lookback.setter
    def lookback(self, value):
        self.selector.lookback = value

    def add_ticker(self, ticker_symbol):
        """Add a ticker to the manager."""
        print(f'Adding Stock: [{ticker_symbol}]')
        self.tickers[ticker_symbol] = dict(zip([t for t in TICKER], [None]*len(TICKER) ))
        self.tickers[ticker_symbol][TICKER.ID] = ticker_symbol
        self.tickers[ticker_symbol][TICKER.SELECTED] = False

    def remove_ticker(self, ticker):
        """Remove a ticker from the manager."""
        self.tickers.pop(ticker, None)

    def get_all_tickers(self):
        """Return a list of all tickers managed."""
        return list(self.tickers.keys())

    def load_ticker_data(self):
        tickers = self.get_all_tickers()
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
            for ticker in tickers:
                try:
                    if ticker in all_data:
                        data_num = len(all_data[ticker])
                        if data_num > 1:
                            self.tickers[ticker][TICKER.DATA] = all_data[ticker]
                            print(f"{ticker}: 成功加载 {len(all_data[ticker])} 条数据")
                            continue
                    # 尝试单独下载
                    print(f"{ticker}: 批量下载失败，尝试单独下载...")
                    ticker_obj = yf.Ticker(ticker)
                    ticker_data = ticker_obj.history(start=start_str, end=end_str)
                    if not ticker_data.empty:
                        self.tickers[ticker][TICKER.DATA] = ticker_data
                        print(f"{ticker}: 单独下载成功，{len(ticker_data)} 条数据")
                    else:
                        print(f"{ticker}: 无数据可用")
                        no_data.append(ticker)
                            
                except Exception as e:
                    print(f"{ticker}: 处理失败 - {str(e)}")
                    no_data.append(ticker)
                    continue
        except Exception as e:
            print(f"批量下载失败: {str(e)}")
            print("尝试逐个下载...")
            
            # 方法2: 逐个下载
            for ticker in tickers:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    ticker_data = ticker_obj.history(
                        start=self.start_date,
                        end=self.end_date
                    )
                    
                    if not ticker_data.empty:
                        self.tickers[ticker][TICKER.DATA] = ticker_data
                        print(f"{ticker}: 成功下载 {len(ticker_data)} 条数据")
                    else:
                        print(f"{ticker}: 无数据")
                        no_data.append(ticker)
                        
                except Exception as e:
                    print(f"{ticker}: 下载失败 - {str(e)}")
                    no_data.append(ticker)
        print(f"\n下载完成: {len(tickers) - len(no_data)}/{len(tickers)} 个股票数据下载成功")
        return no_data

    def process_select_stocks(self):
        for ticker in self.get_all_tickers():
            self.selector.ticker = self.tickers[ticker]
            self.selector.process_train_data()
    
    def select_stocks(self, date_offset, lookback, prediction_threshold=0.7):
        print("!! selecting stocks !!")
        start_date = pd.to_datetime(self.end_date) - pd.DateOffset(date_offset)
        end_date = pd.to_datetime(self.end_date)

        current_data = yf.download(self.get_all_tickers(), start=start_date, end=end_date, group_by='ticker')
        for ticker in self.get_all_tickers():
            local_selector = select_stock.LSTM_Select_Stock(lookback)
            ticker_data = current_data[ticker]
            
            if len(ticker_data) < lookback:
                continue
            local_selector.ticker[TICKER.ID] = ticker,
            local_selector.ticker[TICKER.DATA] = current_data[ticker]
            local_selector.ticker[TICKER.FEATURES] = self.tickers[ticker][TICKER.FEATURES]
            local_selector.ticker[TICKER.MODEL] = self.tickers[ticker][TICKER.MODEL]
            local_selector.ticker[TICKER.SCALER] = self.tickers[ticker][TICKER.SCALER]

            local_selector.preprocess_data()            
            # 获取最近lookback天的数据
            latest_window = np.stack(local_selector.ticker[TICKER.TRAIN_DATA]['Features'].iloc[-lookback:])
            # latest_window_3d = latest_window[np.newaxis, ...] # 转换为3D数组,以适应LSTM输入要求,但模型已经适应3D输入,不需要再转换
            # 预测
            prediction = local_selector.ticker[TICKER.MODEL].predict(latest_window)[0][0]
            print(f'prediction: [{prediction}] <--> {prediction_threshold}')
            if prediction > prediction_threshold:  # 设置较高阈值
                self.tickers[ticker][TICKER.SELECTED] = True

    def get_selected_stocks(self):
        selected_stocks = []
        for ticker in self.get_all_tickers():
            if self.tickers[ticker][TICKER.SELECTED]:
                print(f"Selected stock: {ticker}")
                selected_stocks.append(ticker)
        return selected_stocks

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