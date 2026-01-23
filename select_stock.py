from datetime import date, datetime, time, timedelta
import json
import os
import sys
from tkinter import messagebox
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
from stock import TICKER, FEATURE


class LSTM_Select_Stock(MachineLearningFramework):
    FEATURE_STATE_LIST = None
    def __init__(self):
        self._threshold = 0.05  # 5%收益率阈值
        self._percent_train_test_split = 0.2
        self._evaluation_output = False
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        LSTM_Select_Stock.create_feature_list()

    def load_historical_data(self, ticker_symbol, start_date, end_date):
        """load historical data for a single ticker from yfinance"""
        print(f"!! Loading Ticker {ticker_symbol}: {start_date} - {end_date} !!")
        try:
            ticker_data = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker_symbol}: {e}")
            messagebox.showerror("Data Download Error", f"Error downloading data for {ticker_symbol}: {e}")
            return None
        return ticker_data

    def download_ticker_data(self, ticker:list|str, start_date, end_date):
        try:
            data = yf.download(ticker, 
                                start=start_date, 
                                end=end_date,
                                group_by='ticker',
                                progress=True,  # 显示进度
                                timeout=60     # 延长超时时间
                                # threads=True    # 使用多线程
                                )
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
            messagebox.showerror("Data Download Error", f"Error downloading data for {ticker}: {e}")
            return None
        return data


#region static methods
    @staticmethod
    def create_feature_list():
        LSTM_Select_Stock.FEATURE_STATE_LIST = dict(zip([f for f in FEATURE], [True]*len(FEATURE )))

    @staticmethod
    def get_feature_count():
        return len(FEATURE)

    @staticmethod
    def get_feature_name(feature):
        name = 'No such feature'
        for f in FEATURE:
            if f == feature:
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
    
    @staticmethod
    def enable_feature(feature):
        LSTM_Select_Stock.FEATURE_STATE_LIST[feature] = True

    @staticmethod
    def disable_feature(feature):
        LSTM_Select_Stock.FEATURE_STATE_LIST[feature] = False
#endregion

#region properties
    @property
    def percent_train_test_split(self):
        return self._percent_train_test_split
    @percent_train_test_split.setter
    def percent_train_test_split(self, value):
        self._percent_train_test_split = value

    @property
    def lookback(self):
        return self._lookback
    @lookback.setter
    def lookback(self, value):
        self._lookback = value

    @property
    def evaluation_output(self):
        return self._evaluation_output
    @evaluation_output.setter
    def evaluation_output(self, value):
        self._evaluation_output = value

    @property
    def feature_stats(self):
        return self._ticker[TICKER.FEATURES]
    @feature_stats.setter
    def feature_stats(self, value):
        self._ticker[TICKER.FEATURES] = value
#endregion
    
    def preprocess_data(self, ticker, data, features:list):
        """prepare features for model training"""
        df = data.copy()
        print(f"Processing ticker: {ticker} with features: {[LSTM_Select_Stock.get_feature_name(f) for f in features]}")
        for f in FEATURE:
            # 计算技术指标
            if FEATURE.Open_Close in features    :   df[FEATURE.Open_Close] = df['Close'] - df['Open'] 
            if FEATURE.High_Low in features      :   df[FEATURE.High_Low] = df['High'] - df['Low']
            if FEATURE.Close_Low in features     :   df[FEATURE.Close_Low] = df['Close'] - df['Low']
            if FEATURE.Close_High in features    :   df[FEATURE.Close_High] = df['Close'] - df['High'] 
            if FEATURE.Avg_Price in features     :   df[FEATURE.Avg_Price] = (df['Open'] + df['Close']) / 2
            if FEATURE.Volume_Change in features :   df[FEATURE.Volume_Change] = df['Volume'].pct_change().fillna(0) 
            if FEATURE.MA_5 in features          :   df[FEATURE.MA_5] = df['Close'].rolling(window=5).mean() 
            if FEATURE.MA_20 in features         :   df[FEATURE.MA_20] = df['Close'].rolling(window=20).mean() 
            if FEATURE.RSI in features           :   df[FEATURE.RSI] = self._compute_rsi(df['Close'])
            if FEATURE.MACD in features          :   df[FEATURE.MACD] = self._compute_macd(df['Close'])
            if FEATURE.Volume_MA_5 in features   :   df[FEATURE.Volume_MA_5] = df['Volume'].rolling(5).mean() 
            if FEATURE.Price_Volume_Ratio in features:df[FEATURE.Price_Volume_Ratio] = df['Close'] / df[FEATURE.Volume_MA_5] 
            if FEATURE.Volume in features        :   df[FEATURE.Volume] = df['Volume']
            # 获取基本面指标
            if FEATURE.PE in features:
                value = self.get_pe_ratio(ticker)
                if value is None:
                    LSTM_Select_Stock.FEATURE_STATE_LIST[FEATURE.PE] = False
                    for i, f in enumerate(features):
                        if f == FEATURE.PE:
                            print("Removing PE feature due to unavailable data.")
                            del features[i]
                            break
                else:
                    df[FEATURE.PE] = value
            if FEATURE.PB in features:
                value = self.get_pb_ratio(ticker)
                if value is None:
                    LSTM_Select_Stock.FEATURE_STATE_LIST[FEATURE.PB] = False
                    for i, f in enumerate(features):
                        if f == FEATURE.PB:
                            print("Removing PB feature due to unavailable data.")
                            del features[i]
                            break
                else:
                    df[FEATURE.PB] = value

        scaled_data = self.create_scaled_data(df)

        X = []
        y = []
        for i in range(self._lookback, len(scaled_data)-5):
            X.append(scaled_data[i-self._lookback:i, :])
            y.append(1 if (df['Close'].iloc[i+5]/df['Close'].iloc[i] -1) > 0.05 else 0)

        self._ticker[TICKER.TRAIN_DATA] = pd.DataFrame({
            'Ticker': [ticker]*len(X), # corresponding row number of X data, this is used for ticker identification
            'Features': X,
            'Label': y
        })
        return features

    def create_scaled_data(self, df, features):
        if self._ticker[TICKER.SCALER] is None:
            self._ticker[TICKER.SCALER] = StandardScaler()
        return self._ticker[TICKER.SCALER].fit_transform(df[features])

    def build_model(self, model, input_shape):
        '''
        function _build_lstm_model
        '''
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        return model

    def create_train_test_data(self):
        # 数据集划分
        split_idx = int(len(self._ticker[TICKER.TRAIN_DATA])*(1-self._percent_train_test_split))
        train_test_data = {}
        train_test_data['X Tain'] =  np.stack(self._ticker[TICKER.TRAIN_DATA]['Features'].iloc[:split_idx])
        train_test_data['Y Tain'] = self._ticker[TICKER.TRAIN_DATA]['Label'].iloc[:split_idx]
        train_test_data['X Test'] = np.stack(self._ticker[TICKER.TRAIN_DATA]['Features'].iloc[split_idx:])
        train_test_data['Y Test'] = self._ticker[TICKER.TRAIN_DATA]['Label'].iloc[split_idx:]
        return train_test_data

    def train_model(self, model, x_train, y_train):
        self.create_train_test_data()
        # 模型构建
        self.build_model((x_train.shape[1], x_train.shape[2]))
        
        # 早停法
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        
        # 训练模型
        hist = model.fit(x_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=[early_stop],
                        verbose=1)
        print("!! Complete Model Training !!")

    def create_ticker_folder(self, ticker_name, cur_path='.'):
        try:
            folder_name = f"{ticker_name}@{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_dir = os.path.join(cur_path, folder_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            return model_dir
        except Exception as e:
            print(f"Error to create ticker folder for {ticker}: {str(e)}")
            return None

    def save_ticker_data(self, data, path='model'):
        model_dir = self.create_ticker_folder(cur_path=path)
        if not model_dir:
            return False
        data_file = os.path.join(model_dir, 'data')
        data.to_pickle(data_file)
        return True

    def save_model(self, path, model, scaler, features, start_date, end_date, lookback):
        model_dir = self.create_ticker_folder(cur_path=path)
        if model_dir is None:
            return False
        # 保存特征列表
        model_info = {
            "start_date": start_date,
            "end_date": end_date,
            "lookback": lookback,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features": features
        }
        with open(f"{model_dir}/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        # 保存缩放器文件
        scaler_file = f"{model_dir}/scaler.joblib"
        json.dump(scaler, scaler_file)
        # 保存模型文件
        model.save(f"{model_dir}/model.h5")
        return True

    def evaluate_model(self, model, x_test, y_test):        
        # 评估模型
        print("!! Evaluate Model !!")
        y_pred = (model.predict(x_test) > 0.5).astype(int)
        self._ticker[TICKER.PERFORMANCE] = classification_report(y_test, y_pred, output_dict=self._evaluation_output, zero_division=0)
        print(self._ticker[TICKER.PERFORMANCE])

    def predict(self, model, features):
        """
        Docstring for predict
        predict with trained model on new data
        :param self: Description
        :param data: Description
        """
        return model.predict(features)

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
            return None
        
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
                
                return pb_ratio if pb_ratio and pb_ratio > 0 else None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"Retry {attempt+1} for {ticker} in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error getting P/B ratio for {ticker}: {str(e)}")
                    return None
                
    
# 主程序
if __name__ == "__main__":
    # 配置参数
    ticker = 'NVDA'
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    lookback = 60  # 使用60天历史数据

    ss = LSTM_Select_Stock(lookback=lookback)
    ss.disable_feature(FEATURE.PB)
    ss.disable_feature(FEATURE.PE)
    ss.disable_feature(FEATURE.RSI)
    ss.evaluation_output = True
    if ss.download_ticker_data(ticker, start_date, end_date) is None:
        print("Failed to load historical data.")
        sys.exit(1)
    ss.process_train_data(evaluation=True)
