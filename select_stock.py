import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from Common.AutoNumber import AutoIndex

class FEATURE(AutoIndex):
    Close = ()
    MA_5 = ()
    MA_20 = ()
    RSI = ()
    MACD = ()
    PE = ()
    PB = ()
    Volume = ()
    
class Select_Stock:
    def __init__(self, tickers:list, start_date, end_date=None, win_size=60):
        self._tickers = tickers
        self._start_date =  start_date
        self._end_date = f"{self._yesterday.year}-{self._yesterday.month}-{self._yesterday}" if end_date is None else end_date
        self._window_size = win_size
        self.all_data = None
        self._full_dataset = []
        self._features = [f.name for f in FEATURE]
        self._target = 'Label'
        self._future_days=5 # days used to calculate 
        self._threshold = 0.05  # 5%收益率阈值
        self._prediction_threshold = 0.7

    #region Step 1
    def get_tickers_data(self):
        # download tickers
        self.all_data = yf.download(self._tickers, start=self._start_date, end=self._end_date, group_by='ticker')

        # create sequences and features
        for ticker in self._tickers:
            # put ticker data
            df = self.all_data[ticker].copy()
            df['Ticker'] = ticker
            self._create_feature_data(ticker, df)
    
    def _create_feature_data(self, ticker, df):
        # put technical data
        df[FEATURE.MA_5.name] = df[FEATURE.Close.name].rolling(5).mean()      # 5 days mean
        df[FEATURE.MA_20.name] = df[FEATURE.Close.name].rolling(20).mean()    # 20 days mean
        df[FEATURE.RSI.name] = self._compute_rsi(df[FEATURE.Close.name])      # rsi
        df[FEATURE.MACD.name] = self._compute_macd(df[FEATURE.Close.name])

        # 添加基本面数据（示例）
        df[FEATURE.PE.name] = get_pe_ratio(ticker)  # 需要实现
        df[FEATURE.PB.name] = get_pb_ratio(ticker)  # 需要实现
        
        scaled_data = self._create_scaled_data(df)
        self._create_data_sequence(ticker, df, scaled_data)
    
    def _create_scaled_data(self, df):
        features = [f.name for f in FEATURE]
        scaler = StandardScaler()
        return scaler.fit_transform(df[features])
    
    def _create_data_sequence(self, ticker, df, scaled_data):
        X = []
        y = []
        for i in range(self._window_size, len(scaled_data)-self._future_days):
            X.append(scaled_data[i-self._window_size:i, :])
            y.append(1 if (df['Close'].iloc[i+self._future_days]/df['Close'].iloc[i] -1) > self._threshold else 0)
        
        ticker_df = pd.DataFrame({
            'Ticker': [ticker]*len(X),
            'Features': X,
            'Label': y
        })
        self._full_dataset.append(ticker_df)
        
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
    #endregion Step 1
    
    #region Step 2 修改后的训练流程
    def create_train_test_data(self, test_size=0.2):
        self._full_pd_data = pd.concat(self._full_dataset).sample(frac=1).reset_index(drop=True)
        split_idx = int(len(self._full_pd_data)*(1-test_size))
        
        self._X_train = np.stack(self._full_pd_data['Features'].iloc[:split_idx])
        self._y_train = self._full_pd_data['Label'].iloc[:split_idx]
        self._X_test = np.stack(self._full_pd_data['Features'].iloc[split_idx:])
        self._y_test = self._full_pd_data['Label'].iloc[split_idx:]
    #endregion Step 2
        
    #region Step 3 LSTM模型构建函数
    def build_lstm_model(self):
        self._model = Sequential()
        self._model.add(LSTM(64, return_sequences=True, input_shape=(self._X_train.shape[1], self._X_train.shape[2])))
        self._model.add(Dropout(0.2))
        self._model.add(LSTM(32, return_sequences=False))
        self._model.add(Dropout(0.2))
        self._model.add(Dense(16, activation='relu'))
        self._model.add(Dense(1, activation='sigmoid'))
        self._model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        self._early_stop = EarlyStopping(monitor='val_loss', patience=5)
    #endregion Step 3
                
    #region Step 4 训练模型
    def train_model(self):
        history = self._model.fit(self._X_train, self._y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=[self._early_stop],
                        verbose=1)
    #region Step 4
        
    #region Step 5 评估模型
    def evaluate_data(self):
        y_pred = (self._model.predict(self._X_test) > 0.5).astype(int)
        print(classification_report(self._y_test, y_pred))
    #endregion Step 5
    
    #region Step 6 修改后的选股策略
    def lstm_selection_strategy(self):
        selected_stocks = []
        
        for ticker in self._full_pd_data['Ticker'].unique():
            ticker_data = self._full_pd_data[self._full_pd_data['Ticker'] == ticker]
            
            if len(ticker_data) < self._window_size:
                continue
                
            # 获取最近window_size天的数据
            latest_window = np.stack(ticker_data['Features'].iloc[-self._window_size:])
            
            # 预测
            prediction = self._model.predict(latest_window[np.newaxis, ...])[0][0]
            
            if prediction > self._prediction_threshold:  # 设置较高阈值
                selected_stocks.append(ticker)
        print(f"LSTM推荐股票：{selected_stocks}")
        return selected_stocks
    #endregion Step 6
    
if __name__ == "__main__":
    # 配置参数
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    lookback = 60  # 使用60天历史数据
    
    ss = Select_Stock(tickers, start_date, end_date=end_date)
    # 获取LSTM格式数据
    ss.get_tickers_data()
    ss.create_train_test_data()
    # 训练模型
    ss.build_lstm_model()
    ss.train_model()
    ss.evaluate_data()
    
    # 执行选股策略
    ss.lstm_selection_strategy()
