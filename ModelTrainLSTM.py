from datetime import datetime, time
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
from StockDefine import LTSM_MODEL_PARAM, FEATURE, StockFeature
from StockModel import StockModel

class LSTMModelTrain:
    def __init__(self, stock_model:StockModel, features:list=None, lookback=60):
        self._stock_model = stock_model
        self._percent_train_test_split = 0.2
        self._evaluation_output = True
        self._stock_train_data = None
        self._ticker_processed_data = None
        self._scaler = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self._lookback = lookback
        self._future_days = 5
        self._threshold = 0.05
        self._features = features
        self._performance = None
        self._train_params = {}
        self._train_history = None

#region properties
    @property
    def stock_model(self):
        return self._stock_model
    
    @property
    def scaler(self):
        return self._scaler
    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler

    @property
    def lookback(self):
        return self._lookback
    @lookback.setter
    def lookback(self, val):
        self._lookback = val

    @property
    def future_days(self):
        return self._future_days
    @future_days.setter
    def future_days(self, val):
        self._future_days = val

    @property
    def threshold(self):
        return self._threshold
    @threshold.setter
    def threshold(self, val):
        self._threshold = val

    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, f):
        self._features = f

    @property
    def feature_count(self):
        return len(self._features)

    @property
    def percent_train_test_split(self):
        return self._percent_train_test_split
    @percent_train_test_split.setter
    def percent_train_test_split(self, value):
        self._percent_train_test_split = value

    @property
    def evaluation_output(self):
        return self._evaluation_output
    @evaluation_output.setter
    def evaluation_output(self, value):
        self._evaluation_output = value

    @property
    def performance(self):
        return self._performance

    @property
    def train_parameters(self):
        return self._train_params
    
    @property
    def train_history(self):
        return self._train_history
#endregion
    
    def preprocess_data(self, scaler=None):
        """prepare features for model training"""
        df = self._stock_model.loaded_data
        model_features = self._features
        print(f"Processing ticker: {self._stock_model.ticker_symbol} with features: {[StockFeature.get_feature_name(f) for f in model_features]}")
        for f in model_features:
            # 计算技术指标
            if FEATURE.Open_Close == f :
                df[FEATURE.Open_Close] = np.where(df['Open'] == 0, 1, df['Close'] / df['Open']) 
            elif FEATURE.High_Low == f:
                df[FEATURE.High_Low] = np.where(df['Low'] == 0, 1, df['High'] / df['Low'])
            elif FEATURE.Close_Low == f:
                df[FEATURE.Close_Low] = np.where(df['Low'] == 0, 1, df['Close'] / df['Low'])
            elif FEATURE.Close_High == f:
                df[FEATURE.Close_High] = np.where(df['High'] == 0, 1, df['Close'] / df['High']) 
            elif FEATURE.Avg_Price == f:
                df[FEATURE.Avg_Price] = (df['Open'] + df['Close']) / 2
            elif FEATURE.Volume_Change == f:
                df[FEATURE.Volume_Change] = df['Volume'].pct_change().fillna(0) 
            elif FEATURE.MA_5 == f:
                df[FEATURE.MA_5] = df['Close'].rolling(window=5).mean() 
            elif FEATURE.MA_20 == f:
                df[FEATURE.MA_20] = df['Close'].rolling(window=20).mean() 
            elif FEATURE.RSI == f:
                df[FEATURE.RSI] = self._compute_rsi(df['Close'])
            elif FEATURE.MACD == f:
                df[FEATURE.MACD] = self._compute_macd(df['Close'])
            elif FEATURE.Volume_MA_5 == f:
                df[FEATURE.Volume_MA_5] = df['Volume'].rolling(5).mean() 
            elif FEATURE.Price_Volume_Ratio == f:
                df[FEATURE.Price_Volume_Ratio] = np.where(df[FEATURE.Volume_MA_5] == 0, 1, df['Close'] / df[FEATURE.Volume_MA_5])
            elif FEATURE.Volume == f:
                df[FEATURE.Volume] = df['Volume']
            # 获取基本面指标
            elif FEATURE.PE == f:
                df[FEATURE.PE] = self.get_pe_ratio()
            elif FEATURE.PB == f:
                df[FEATURE.PB] = self.get_pb_ratio()
            else:
                raise ValueError("Undefined feature")
        self._stock_train_data = df
        scaled_data = self._create_scaled_data(scaler)
        X = []
        y = []
        lookback = self.lookback
        threshold = self.threshold
        for i in range(lookback, len(scaled_data)-5):
            X.append(scaled_data[i-lookback:i, :])
            y.append(1 if (df['Close'].iloc[i+5]/df['Close'].iloc[i] -1) > threshold else 0)

        self._ticker_processed_data = pd.DataFrame({
            'Ticker': [self._stock_model.ticker_symbol]*len(X), # corresponding row number of X data, this is used for ticker identification
            'Features': X,
            'Label': y
        })

    def _create_scaled_data(self, scaler):
        if scaler is None:
            self._scaler = StandardScaler()
        else:
            self._scaler = scaler
        model_data = self._stock_train_data
        return self._scaler.fit_transform(model_data[self.features])

    def build_model(self, input_shape):
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
        self._stock_model.model = model

    def _create_train_test_data(self):
        # 数据集划分
        split_idx = int(len(self._ticker_processed_data)*(1-self._percent_train_test_split))
        train_test_data = {}
        train_test_data['X Tain'] =  np.stack(self._ticker_processed_data['Features'].iloc[:split_idx])
        train_test_data['Y Tain'] = self._ticker_processed_data['Label'].iloc[:split_idx]
        train_test_data['X Test'] = np.stack(self._ticker_processed_data['Features'].iloc[split_idx:])
        train_test_data['Y Test'] = self._ticker_processed_data['Label'].iloc[split_idx:]
        return train_test_data

    def train_model(self):
        train_test_data = self._create_train_test_data()
        # 模型构建
        x_train = train_test_data['X Tain']
        y_train = train_test_data['Y Tain']
        self.build_model((x_train.shape[1], x_train.shape[2]))
        
        # 早停法
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        
        # 训练模型
        self._train_history = self._stock_model.model.fit(x_train, y_train,
                                epochs=50,
                                batch_size=32,
                                validation_split=0.1,
                                callbacks=[early_stop],
                                verbose=1)
        print("!! Complete Model Training !!")
        self.evaluate_model(self._stock_model.model,
                            train_test_data['X Test'], 
                            train_test_data['Y Test'])

    def evaluate_model(self, model, x_test, y_test):        
        # 评估模型
        print("!! Evaluate Model !!")
        print(f"Shape of x_test: {x_test.shape}")
        y_pred = (model.predict(x_test)).astype(int)
        self._performance = classification_report(y_test, y_pred, output_dict=self._evaluation_output, zero_division=0)
        print(self._performance)

    def prepare_input(self, window_data, expected_shape):
        """准备LSTM输入数据"""
        # 确保是numpy数组
        if not isinstance(window_data, np.ndarray):
            window_data = np.array(window_data)
        
        # 检查维度
        if len(window_data.shape) == 2:
            # 已经是 (timesteps, features)
            if window_data.shape[0] != expected_shape[0]:
                # 调整时间步长
                if window_data.shape[0] > expected_shape[0]:
                    window_data = window_data[-expected_shape[0]:, :]
                else:
                    padding = expected_shape[0] - window_data.shape[0]
                    window_data = np.pad(window_data, ((padding, 0), (0, 0)), mode='constant')
            
            # 添加batch维度
            window_data = window_data[np.newaxis, ...]
        
        return window_data
    
    def predict(self):
        """
        Docstring for predict
        predict with trained model on new data
        :param self: Description
        :param data: Description
        """
        model = self._stock_model.model
        expected_shape = model.input_shape[1:]
        prepared_data = self.prepare_input(self._ticker_processed_data['Features'], expected_shape)
        return model.predict(prepared_data, verbose=0)[0][0]

    def create_model_parameters(self):
        self._train_params = {
            LTSM_MODEL_PARAM.timestamp.name: datetime.now().strftime('%Y%m%d_%H%M%S'),
            LTSM_MODEL_PARAM.lookback.name: self._lookback,
            LTSM_MODEL_PARAM.future_days.name: self._future_days,
            LTSM_MODEL_PARAM.threshold.name: self._threshold,
            LTSM_MODEL_PARAM.features.name: [f.name for f in self._features],
            LTSM_MODEL_PARAM.feature_count.name: self.feature_count,
        }
        return self._train_params

    def assign_parameters_from_loading(self, params):
        self._lookback = params[LTSM_MODEL_PARAM.lookback.name]
        self._future_days = params[LTSM_MODEL_PARAM.future_days.name]
        self._threshold = params[LTSM_MODEL_PARAM.threshold.name]
        self._features = params[LTSM_MODEL_PARAM.features]

    def get_model_summary(self):
        """获取模型架构信息"""
        import io
        string_buffer = io.StringIO()
        self._stock_model.model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
        summary_string = string_buffer.getvalue()
        string_buffer.close()
        return summary_string
    
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
    def get_pe_ratio(self):
        try:
            stock = yf.Ticker(self._stock_model.ticker_symbol)
            # 优先使用TTM市盈率，若不存在则尝试其他类型
            return {
                "trailingPE": stock.info.get('trailingPE'),
                "forwardPE": stock.info.get('forwardPE'),
                "priceToBook": stock.info.get('priceToBook')
            }
        except Exception as e:
            print(f"Error getting P/E for {self._stock_model.ticker_symbol}: {str(e)}")
            return None
        
    # get price to book ratio 市净率
    # 示例用法
    @lru_cache(maxsize=100)
    def get_pb_ratio(self, max_retries=3):
        """
        带缓存和重试机制的市净率获取函数
        """
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(self._stock_model.ticker_symbol)
                
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
                    print(f"Retry {attempt+1} for {self._stock_model.ticker_symbol} in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error getting P/B ratio for {self._stock_model.ticker_symbol}: {str(e)}")
                    return None
                
    def process_train_data(self):
        """
        ### 加载并预处理数据
        """
        self.preprocess_data()

        # 训练模型
        self.train_model()

    def process_prediction(self, scaler):
        self.preprocess_data(scaler)
        return self.predict()

# 主程序
if __name__ == "__main__":
    # 配置参数
    ticker = 'NVDA'
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    lookback = 60  # 使用60天历史数据

    stock_model = StockModel(ticker)
    stock_model.download_ticker_data(start_date, end_date)

    stock_feature = StockFeature()

    ss = LSTMModelTrain(stock_model, lookback=lookback)
    ss.features = stock_feature.get_features()
    ss.evaluation_output = True
    ss.process_train_data()
