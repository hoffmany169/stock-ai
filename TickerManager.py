import select_stock
import yfinance as yf
import numpy as np
import pandas as pd
from stock import TICKER

class TickerManager:
    def __init__(self, start_date, end_date, lookback=60):
        self.tickers = {}
        self.start_date = start_date
        self.end_date = end_date
        self.selector = select_stock.LSTM_Select_Stock(lookback)

    @property
    def stock_selector(self):
        return self.selector

    def add_ticker(self, ticker_symbol):
        """Add a ticker to the manager."""
        self.tickers[ticker_symbol] = {
            TICKER.ID: ticker_symbol,
            TICKER.DATA: None,
            TICKER.FEATURES: None,
            TICKER.MODEL: None,
            TICKER.SCALER: None,
            TICKER.PERFORMANCE: None,
            TICKER.SELECTED: False
        }

    def remove_ticker(self, ticker):
        """Remove a ticker from the manager."""
        self.tickers.pop(ticker, None)

    def get_all_tickers(self):
        """Return a list of all tickers managed."""
        return list(self.tickers.keys())
    
    def load_ticker_data(self):
        tickers = self.get_all_tickers()
        all_data = yf.download(tickers, start=self.start_date, end=self.end_date, group_by='ticker')
        for ticker in tickers:
            self.tickers[ticker][TICKER.DATA] = all_data[ticker]

    def process_select_stocks(self):
        for ticker in self.get_all_tickers():
            self.selector.ticker = self.tickers[ticker]
            self.selector.process_train_data()
    
    def process_valuate_models(self):
        for ticker in self.get_all_tickers():
            self.selector.ticker = self.tickers[ticker]
            self.selector.evaluate_model()

    def select_stocks(self, date_offset, lookback, prediction_threshold=0.7):
        selected_stocks = []
        start_date = pd.to_datetime(self.end_date) - pd.DateOffset(date_offset)
        end_date = pd.to_datetime(self.end_date)

        current_data = yf.download(self.get_all_tickers(), start=start_date, end=end_date, group_by='ticker')
        for ticker in self.get_all_tickers():
            local_selector = select_stock.LSTM_Select_Stock(lookback)
            ticker_data = current_data[ticker]
            
            if len(ticker_data) < lookback:
                continue
            local_selector.ticker = {
                TICKER.ID: ticker,
                TICKER.DATA: current_data[ticker],
                TICKER.FEATURES: self.tickers[ticker][TICKER.FEATURES],
                TICKER.MODEL: self.tickers[ticker][TICKER.MODEL],
                TICKER.SCALER: self.tickers[ticker][TICKER.SCALER]
            }


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
    manager.load_ticker_data()
    manager.process_select_stocks()
    selected_stocks = manager.select_stocks(180, lookback=60, prediction_threshold=0.7)
    print("Final selected stocks:", manager.get_selected_stocks())