from tkinter import messagebox
import select_stock
import yfinance as yf
import numpy as np
import pandas as pd
from stock import TICKER

class TickerManager:
    def __init__(self, start_date = None, end_date = None, lookback=60):
        self.tickers = {}
        self._start_date = start_date
        self._end_date = end_date
        self.selector = select_stock.LSTM_Select_Stock(lookback)
        self._reload_data = True

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
    def reload_data(self):
        return self._reload_data
    @reload_data.setter
    def reload_data(self, value):
        self._reload_data = value

    @property
    def lookback(self):
        return self.selector.lookback
    @lookback.setter
    def lookback(self, value):
        self.selector.lookback = value

    def add_ticker(self, ticker_symbol):
        """Add a ticker to the manager."""
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
        if not self.reload_data:
            messagebox.showinfo("Info", "Ticker data is already loaded and has not been changed.")
            return True
        tickers = self.get_all_tickers()
        rmv = []
        try:
            all_data = yf.download(tickers, start=self.start_date, end=self.end_date, group_by='ticker', threads=True, progress=False)
            for ticker in tickers:
                if all_data[ticker] is None or all_data[ticker].empty:
                    messagebox.showerror("Error", f"No data found for ticker: {ticker}")
                    rmv.append(ticker)
                    continue
                self.tickers[ticker][TICKER.DATA] = all_data[ticker]
            if len(rmv) > 0:
                for t in rmv:
                    del self.tickers[t]
            self.reload_data = False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ticker data: {e}")
            return False
        return True

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
    if not manager.load_ticker_data():
        print("Failed to load ticker data.")
        exit(1)
    manager.process_select_stocks()
    selected_stocks = manager.select_stocks(180, lookback=60, prediction_threshold=0.7)
    print("Final selected stocks:", manager.get_selected_stocks())