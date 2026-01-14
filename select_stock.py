from Common.AutoNumber import AutoIndex
from datetime import date, time, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import time
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from MLFramework import MachineLearningFramework

class FEATURE(AutoIndex):
    Open_Close = ()
    High_Low = ()
    Close_Low = ()
    Close_High = ()
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

class LSTM_Select_Stock(MachineLearningFramework):
    FEATURE_STATE_LIST = {}
    def __init__(self, tickers:list, start_date, end_date=None, lookback=60):
        self._tickers = tickers
        self._start_date =  start_date
        self._yesterday = date.today() - timedelta(days=1)
        self._end_date = f"{self._yesterday.year}-{self._yesterday.month}-{self._yesterday}" if end_date is None else end_date
        self._lookback = lookback
        self.all_data = None # data downloaded from yfinance
        self._features = []
        self._threshold = 0.05  # 5%收益率阈值
        self.lstm_model = None
        self.all_ticker_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        LSTM_Select_Stock.create_feature_list()

#region static methods
    @staticmethod
    def create_feature_list():
        LSTM_Select_Stock.FEATURE_STATE_LIST = dict(zip([f for f in FEATURE], [True]*len(FEATURE )))

    @staticmethod
    def get_feature_count():
        return len(FEATURE)

    @staticmethod
    def get_feature_name(feature):
        for f in FEATURE:
            if f == FEATURE.Open_Close:
                name = "Open-Close Difference"
            elif f == FEATURE.High_Low:
                name = "High-Low Difference"
            elif f == FEATURE.Close_Low:
                name = "Close-Low Difference"
            elif f == FEATURE.Close_High:
                name = "Close-High Difference"
            elif f == FEATURE.Avg_Price:
                name = "Average Price"
            else:
                name = ' '.join(f.name.split('_'))
        return name

    @staticmethod
    def is_feature_used(feature):
        return LSTM_Select_Stock.FEATURE_STATE_LIST[feature]
#endregion

#region properties
    @property
    def threshold(self):
        return self._threshold
    
    @property
    def prediction_threshold(self):
        return self._prediction_threshold
    
    @property
    def lookback(self):
        return self._lookback
#endregion

    def load_train_data(self):
        """load historical stock data from yfinance"""
        self.all_data = yf.download(self._tickers, start=self._start_date, end=self._end_date, group_by='ticker')

    def preprocess_data(self):
        """prepare features for model training"""
        full_dataset = []
        for ticker in self._tickers:
            df = self.all_data[ticker].copy()
            df['Ticker'] = ticker
            # 计算技术指标
            if LSTM_Select_Stock.is_feature_used(FEATURE.Open_Close) :df[FEATURE.Open_Close] = df['Close'] - df['Open'] 
            if LSTM_Select_Stock.is_feature_used(FEATURE.High_Low) :df[FEATURE.High_Low] = df['High'] - df['Low']
            if LSTM_Select_Stock.is_feature_used(FEATURE.Close_Low) :df[FEATURE.Close_Low] = df['Close'] - df['Low']
            if LSTM_Select_Stock.is_feature_used(FEATURE.Close_High) :df[FEATURE.Close_High] = df['Close'] - df['High'] 
            if LSTM_Select_Stock.is_feature_used(FEATURE.Avg_Price) :df[FEATURE.Avg_Price] = (df['Open'] + df['Close']) / 2
            if LSTM_Select_Stock.is_feature_used(FEATURE.Volume_Change) :df[FEATURE.Volume_Change] = df['Volume'].pct_change().fillna(0) 
            if LSTM_Select_Stock.is_feature_used(FEATURE.MA_5) :df[FEATURE.MA_5] = df['Close'].rolling(window=5).mean() 
            if LSTM_Select_Stock.is_feature_used(FEATURE.MA_20) :df[FEATURE.MA_20] = df['Close'].rolling(window=20).mean() 
            if LSTM_Select_Stock.is_feature_used(FEATURE.RSI) :df[FEATURE.RSI] = self._compute_rsi(df['Close'])
            if LSTM_Select_Stock.is_feature_used(FEATURE.MACD) :df[FEATURE.MACD] = self._compute_macd(df['Close'])
            if LSTM_Select_Stock.is_feature_used(FEATURE.Volume_MA_5) :df[FEATURE.Volume_MA_5] = df['Volume'].rolling(5).mean() 
            if LSTM_Select_Stock.is_feature_used(FEATURE.Price_Volume_Ratio) :df[FEATURE.Price_Volume_Ratio] = df['Close'] / df['Volume_MA_5'] 
            # 获取基本面指标
            value = self.get_pe_ratio(ticker)
            if np.isnan(value):
                LSTM_Select_Stock.FEATURE_STATE_LIST[FEATURE.PE] = False
            else:
                df[FEATURE.PE] = value
            value = self.get_pb_ratio(ticker)
            if np.isnan(value):
                LSTM_Select_Stock.FEATURE_STATE_LIST[FEATURE.PB] = False
            else:
                df[FEATURE.PB] = value
            df[FEATURE.PB] = value
            df[FEATURE.Volume] = df['Volume']

            # 创建时间序列窗口
            target = 'Label'
            scaled_data = self.create_scaled_data(df) 

            X = []
            y = []
            for i in range(self._lookback, len(scaled_data)-5):
                X.append(scaled_data[i-self._lookback:i, :])
                y.append(1 if (df['Close'].iloc[i+5]/df['Close'].iloc[i] -1) > 0.05 else 0)
            
            self.ticker_df = pd.DataFrame({
                'Ticker': [ticker]*len(X), # corresponding row number of X data
                'Features': X,
                'Label': y
            })
            full_dataset.append(df)
        self.all_ticker_data = pd.concat(full_dataset).sample(frac=1).reset_index(drop=True)

    def create_scaled_data(self, df):
        features = [f for f in FEATURE if LSTM_Select_Stock.is_feature_used(f)]
        scaler = StandardScaler()
        return scaler.fit_transform(df[features])

    def build_model(self, input_shape):
        '''
        function _build_lstm_model
        '''
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        self.lstm_model.add(Dropout(0.2))
        self.lstm_model.add(LSTM(32, return_sequences=False))
        self.lstm_model.add(Dropout(0.2))
        self.lstm_model.add(Dense(16, activation='relu'))
        self.lstm_model.add(Dense(1, activation='sigmoid'))
        
        self.lstm_model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    def create_train_test_data(self, percent_split=0.2):
        # 数据集划分
        split_idx = int(len(self.all_ticker_data)*(1-percent_split))
        
        self.X_train = np.stack(self.all_ticker_data['Features'].iloc[:split_idx])
        self.y_train = self.all_ticker_data['Label'].iloc[:split_idx]
        self.X_test = np.stack(self.all_ticker_data['Features'].iloc[split_idx:])
        self.y_test = self.all_ticker_data['Label'].iloc[split_idx:]

    def train_model(self):
        self.create_train_test_data()
        # 模型构建
        self.build_model((self.X_train.shape[1], self.X_train.shape[2]))
        
        # 早停法
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        
        # 训练模型
        history = self.lstm_model.fit(self.X_train, self.y_train,
                                    epochs=50,
                                    batch_size=32,
                                    validation_split=0.1,
                                    callbacks=[early_stop],
                                    verbose=1)
    def evaluate_model(self, output=False):        
        # 评估模型
        y_pred = (self.lstm_model.predict(self.X_test) > 0.5).astype(int)
        print(classification_report(self.y_test, y_pred, output_dict=output))

    # def process_train_data(self, output=True):
    #     # 加载并预处理数据
    #     self.load_train_data()
    #     self.preprocess_data()
        
    #     # 训练模型
    #     self.train_model()
    #     self.evaluate_model(output=output)

    def predict(self, data=None):
        """
        Docstring for predict
        predict with trained model on new data
        :param self: Description
        :param data: Description
        """
        if data is None:
            raise ValueError("Input data for prediction cannot be None")
        return self.lstm_model.predict(data)

    def select_stocks(self, start_date, end_date, lookback, prediction_threshold=0.7):
        selected_stocks = []
        current_data = yf.download(self._tickers, start=start_date, end=end_date, group_by='ticker')
        for ticker in current_data['Ticker'].unique():
            ticker_data = current_data[current_data['Ticker'] == ticker]
            
            if len(ticker_data) < self._lookback:
                continue
                
            # 获取最近lookback天的数据
            latest_window = np.stack(ticker_data['Features'].iloc[-lookback:])
            
            # 预测
            prediction = self.predict(latest_window[np.newaxis, ...])[0][0]
            
            if prediction > prediction_threshold:  # 设置较高阈值
                selected_stocks.append(ticker)
        print(f"Selected {len(selected_stocks)} stocks based on LSTM predictions:")
        print(selected_stocks)
        return selected_stocks

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
                
    
# 主程序
if __name__ == "__main__":
    # 配置参数
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    lookback = 60  # 使用60天历史数据
    
    ss = LSTM_Select_Stock(tickers, start_date, end_date=end_date, lookback=lookback)
    ss.process_train_data(output=True)

    # 执行选股策略
    selected = ss.select_stocks(pd.to_datetime(end_date) - pd.DateOffset(90),
                    pd.to_datetime(end_date),
                    lookback=lookback)
    print("LSTM推荐股票：", selected)