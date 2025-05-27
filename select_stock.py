import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

class Select_Stock:
    def __init__(self, tickers:list, start_date, end_date=None, win_size=60):
        self._yesterday = date.today() - timedelta(days = 1)
        self._tickers = tickers
        self._start_date =  start_date
        self._end_date = f"{self._yesterday.year}-{self._yesterday.month}-{self._yesterday}" if end_date is None else end_date
        self._window_size = win_size
        self.all_data = None
        self._full_data = None

    def get_tickers_data(self):
        # download tickers
        self.all_data = yf.download(self._tickers, start=self._start_date, end=self._end_date, group_by='ticker')

        # create sequences and features
        for ticker in self._tickers:
            # put ticker data
            df = self.all_data[ticker].copy()
            df['Ticker'] = ticker
            # put technical data
            df['MA_5'] = df['Close'].rolling(5).mean()      # 5 days mean
            df['MA_20'] = df['Close'].rolling(20).mean()    # 20 days mean
            df['RSI'] = self._compute_rsi(df['Close'])      # rsi
            df['MACD'] = self._compute_macd(df['Close'])



    # 2. compute rsi
    def _compute_rsi(self, data_sequence, period=14):
        delta = data_sequence.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # compute macd
    def _compute_macd(self, data_sequence, slow=26, fast=12, signal=9):
        ema_fast = data_sequence.ewm(span=fast).mean()
        ema_slow = data_sequence.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
