
"""
class Model_Prediction
predict trend of stock
NOTE: 新数据应紧接在训练数据之后，保持时间序列连贯性。

"""
import os, sys
sys.path.append('.')
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from joblib import load
from stock_model import Stock_Model, FILE_TYPE
import yfinance as yf
import json
SYMBOL = "IFX.DE"
HISTORY = "3mo"

class Model_Prediction:
    CONFIG_FILE = 'config.json'
    def __init__(self, stock, period='3mo', interval="1d", win_size=60, delay_days=3, rsc_path='.'):
        self._config = None
        self._stock_model = Stock_Model(stock, period, interval=interval, win_size=win_size, path=rsc_path)
        self._stock = stock if stock is not None else self._stock
        self._period = period if period is not None else self._period
        self._interval = interval if interval is not None else self._interval    
        #self._session = requests.Session(impersonate="chrome")
        self._new_data = None
        self._window_size = win_size

    def load_config(self):
        try:
            with open(self.CONFIG_FILE, 'r') as cfg:
                json.load(self._config, cfg)
        except Exception as e:
            print(f"Load configure file fails: {e}")

    def process_predicting_data(self):
        self.load_stock()
        self._stock_model.load_scaler()
        self._stock_model.set_working_data()
        self._stock_model.scale_data(create=False, save=False)
        self._create_sequence()
    
    def load_stock(self):
        # self._new_data = yf.Ticker(self._stock, session=self.session).history(period=self._period, interval=self._interval)
        self._stock_model.load_stock(self._stock, self._period, self._interval)
        

    def _prepare_predict_data(self):
        scaled_new_data = self._scaler.transform(self._new_data)
        self._prepared_data = np.array([scaled_new_data[-self._window_size:]])

    def _predict_new_data(self):
        self._model_predict_data = self._model.predict(self._predict_data)

    def _invert_normalized_data(self):
        # 反归一化需要重建完整的多变量矩阵（仅Close列有值，其他列置0）
        dummy_matrix = np.zeros((1, self._features.shape[1]))
        dummy_matrix[:, 3] = self._model_predict_data.flatten()  # 第4列是Close
        self._predicted_data = self._scaler.inverse_transform(dummy_matrix)[0, 3]

    def sigle_day_predict(self):
        self._prepare_predict_data()
        self._predict_new_data()
        self._invert_normalized_data()
        return self._predicted_data
        
    def multi_day_predict(self, days=5) -> list:
        self.predictions = []
        scaled_new_data = self._scaler.transform(self._new_data)
        initial_sequence = scaled_new_data[-self._window_size:]
        current_sequence = initial_sequence.copy()
        self._prepare_predict_data()

        for _ in range(days):
            self._model_predict_data = self._model.predict(current_sequence.reshape(1, self._window_size, -1))

            # update sequence: slide window
            new_row = current_sequence[-1].copy()
            new_row[3] = next_day_scaled[0][0]  # 更新Close列
            current_sequence = np.vstack([current_sequence[1:], new_row])
            self._invert_normalized_data()
            self.predictions.append(self._predicted_data)            
        return self.predictions
        
