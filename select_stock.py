from Common.AutoNumber import AutoIndex
from datetime import date, time, timedelta
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import numpy as np
import pandas as pd
import time
from functools import lru_cache

class FEATURE(AutoIndex):
    Open_Close = ()
    High_Low = ()
    Open_Low = ()
    Avg_Price = ()
    Volume_Change = ()
    MA_5 = ()
    MA_20 = ()
    RSI = ()
    MACD = ()
    Volume_MA_5 = ()
    Price_Volume_Ratio = ()
    PE = ()
    PB = ()
    Volume = ()

class Select_Stock:
    def __init__(self, tickers:list, start_date, end_date=None, lookback=60):
        self._tickers = tickers
        self._start_date =  start_date
        self._yesterday = date.today() - timedelta(days=1)
        self._end_date = f"{self._yesterday.year}-{self._yesterday.month}-{self._yesterday}" if end_date is None else end_date
        self._lookback = lookback
        self.all_data = None # data downloaded from yfinance
        self._full_dataset = []
        self._features = []
        self._threshold = 0.05  # 5%收益率阈值
        self._prediction_threshold = 0.7

    @property
    def threshold(self):
        return self._threshold
    
    @property
    def prediction_threshold(self):
        return self._prediction_threshold
    
    @property
    def lookback(self):
        return self._lookback

    def _download_data_from_yfinance(self):
        self.all_data = yf.download(self._tickers, start=self._start_date, end=self._end_date, group_by='ticker')

    def _preprocess_data(self):
        """prepare features for model training"""
        for ticker in self._tickers:
            df = self.all_data[ticker].copy()
            df['Ticker'] = ticker
            # 计算技术指标
            df[FEATURE.Open_Close] = df['Close'] - df['Open']
            df[FEATURE.High_Low] = abs(df['High'] - df['Low']) / df['High'] # 相对最高价最大波动率
            df[FEATURE.Open_Low] = abs(df['Open'] - df['Low']) / df['Open'] # 相对开盘价最大波动率
            df[FEATURE.Avg_Price] = (df['Open'] + df['Close']) / 2
            df[FEATURE.Volume_Change] = df['Volume'].pct_change().fillna(0)
            df[FEATURE.MA_5] = df['Close'].rolling(window=5).mean()
            df[FEATURE.MA_20] = df['Close'].rolling(window=20).mean()
            df[FEATURE.RSI] = self._compute_rsi(df['Close'])
            df[FEATURE.MACD] = self._compute_macd(df['Close'])
            df[FEATURE.Volume_MA_5] = df['Volume'].rolling(5).mean()
            df[FEATURE.Price_Volume_Ratio] = df['Close'] / df['Volume_MA_5']
            # 获取基本面指标
            df[FEATURE.PE] = self.get_pe_ratio(ticker)
            df[FEATURE.PB] = self.get_pb_ratio(ticker)
            df[FEATURE.Volume] = df['Volume']

            # 创建时间序列窗口
            features = [f for f in FEATURE]
            target = 'Label'
            
            # 生成时间序列样本
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[features])
            
            X = []
            y = []
            for i in range(self._lookback, len(scaled_data)-5):
                X.append(scaled_data[i-self._lookback:i, :])
                y.append(1 if (df['Close'].iloc[i+5]/df['Close'].iloc[i] -1) > 0.05 else 0)
            
            self.ticker_df = pd.DataFrame({
                'Ticker': [ticker]*len(X),
                'Features': X,
                'Label': y
            })
            
        
        # return pd.concat(full_dataset).sample(frac=1).reset_index(drop=True)

    def create_data_sequence(self, ticker, df):
        '''
        function _create_data_sequence
        param[ticker]: ticker of stock
        param[df]: dataframe of stock data
        '''
        print("Implementation _create_data_sequence")

    def create_train_test_data(self, test_size=0.2):
        '''
        function create_train_test_data
        param[test_size]: ratio of dividing train- and test-data
        '''
        print("Implementation create_train_test_data")


    # compute rsi
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

    # get trailing PE (or forward PE) 市盈率
    # 示例用法
    @lru_cache(maxsize=100)  # 缓存最近100支股票的查询
    def get_pe_ratio(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            # 优先使用TTM市盈率，若不存在则尝试其他类型
            return {
                "trailingPE": stock.info.get('trailingPE'),
                "forwardPE": stock.info.get('forwardPE'),
                "priceToBook": stock.info.get('priceToBook')
            }
        except Exception as e:
            print(f"Error getting P/E for {ticker}: {str(e)}")
            return np.nan
        
    # get price to book ratio 市净率
    # 示例用法
    @lru_cache(maxsize=100)
    def get_pb_ratio(self, ticker, max_retries=3):
        """
        带缓存和重试机制的市净率获取函数
        """
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                # 直接获取市净率
                pb_ratio = stock.info.get('priceToBook')
                
                # 如果不存在，尝试计算
                if pb_ratio is None or np.isnan(pb_ratio):
                    current_price = stock.history(period='1d')['Close'].iloc[-1]
                    book_value = stock.info.get('bookValue')
                    
                    if book_value and book_value > 0:
                        pb_ratio = current_price / book_value
                
                return pb_ratio if pb_ratio and pb_ratio > 0 else np.nan
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"Retry {attempt+1} for {ticker} in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error getting P/B ratio for {ticker}: {str(e)}")
                    return np.nan
                
    def create_scaled_data(self, X):
        features = [f.name for f in FEATURE]
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
