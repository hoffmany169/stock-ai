
"""
class Model_Prediction
predict trend of stock
NOTE: 新数据应紧接在训练数据之后，保持时间序列连贯性。

"""
import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from keras.models import load_model
from joblib import load
from stock_model import Stock_Model, FILE_TYPE
import yfinance as yf
import json
SYMBOL = "IFX.DE"
HISTORY = "3mo"

class Model_Prediction:
    CONFIG = None
    CFG_ENTRY = ['model_name', 'stock', 'model_period', 'interval', 'win_size', 'delay', 'last_date']
    def __init__(self, model_file, period='3mo', pred_days=5, path='.'):
        self._model_file = model_file
        if Model_Prediction.CONFIG is None:
            parts = self._get_stock_info_from_model_file()
        self._predicting_days = pred_days
        self._path = path
        self._stock_model = Stock_Model(parts[0], period, interval=parts[1], win_size=int(parts[2]), path=parts[3], delay_days=delay_days)
        self._pred_file_name = f"{self._stock_model.stock_symbol}_{self._stock_model.interval}_{self._stock_model.window_size}_{self._predicting_days}"

    #region Property
    @property
    def model_file(self):
        return self._model_file
    
    @property
    def stock(self):
        return self._stock_model
        
    @property
    def stock_model(self):
        return self._stock_model.model
        
    @property
    def path(self):
        return self._path
        
    @property
    def predicted_data(self):
        return self._Y_pred_actual
    
    @property
    def predicting_days(self):
        return self._predicting_days
    
    @predicting_days.setter
    def predicting_days(self, d):
        self._predicting_days = d
    #endregion Property

    def load_config(self):
        try:
            with open(self.CONFIG_FILE, 'r') as cfg:
                json.load(self._config, cfg)
        except Exception as e:
            print(f"Load configure file fails: {e}")

    def _get_stock_info_from_model_file(self):
        basename = os.path.basename(self._model_file)
        fn = os.path.splitext(basename)
        model_path = self._model_file.split(basename)
        ## stock symbol, period, interval, window size
        parts = fn[0].split('_')
        if len(parts) < 3:
            raise ValueError("Error: The format of model file name is invalid!")
        parts.append(model_path[0] if len(model_path[0]) > 0 else '.')
        return parts
    
    def prepare_prediction_data(self, scaled_data):
        self._prepared_predict_data = np.array(scaled_data[-self._stock_model.window_size:])

    def predict_data(self):
        if self._stock_model.model is None:
            print("Warning: model is None. It must be created or loaded!")
            return
        predict_data = self._stock_model.model.predict(self._prepared_predict_data)
        self._Y_pred_actual = self._stock_model.invert_normalized_data(predict_data)
        
    def multi_day_predict(self, days=-1) -> list:
        self._Y_pred_actual = []
        if days > 0:
            self._predicting_days = days
        current_sequence = self._prepared_predict_data.copy()

        for _ in range(self._predicting_days + self._stock_model.delay_days):
            # 确保输入形状为 (1, window_size, n_features)
            X = np.expand_dims(current_sequence, axis=0)
            next_day_scaled_data = self._stock_model.model.predict(X, verbose=0)[0][0]
            # update sequence: slide window
            new_row = current_sequence[-1].copy()
            new_row[3] = next_day_scaled_data  # Close列
            new_row[0] = next_day_scaled_data  # Open
            new_row[1] = next_day_scaled_data  # High
            new_row[2] = next_day_scaled_data  # Low
            new_row[4] = 0  # Volume设为0（或自定义逻辑）
            current_sequence = np.vstack([current_sequence[1:], new_row])
            # new_predicted_data = self._stock_model.invert_normalized_data(next_day_scaled_data)
            # 反归一化Close价格
            dummy = np.zeros((1, self.stock.features.shape[1]))
            dummy[0, 3] = next_day_scaled_data
            predicted_close = self.stock.scaler.inverse_transform(dummy)[0, 3]
            self._Y_pred_actual.append(predicted_close)            

    def save_prediction_result(self, sep=':', decimal=','):
        if type(self._Y_pred_actual) is not list:
            result_y_pred_actual = np.array([self._Y_pred_actual]).reshape(-1,1)
        else:    
            result_y_pred_actual = np.array(self._Y_pred_actual).reshape(-1,1)
        result = pd.DataFrame(result_y_pred_actual, columns=['predicted price'])
        path_name = os.path.join(self._path, f"{self._pred_file_name}.csv")
        result.to_csv(path_name, sep=sep, decimal=decimal)

    #region main methods
    def process_prediction(self):
        self._stock_model.load_stock()
        self._stock_model.load_keras_model()
        self._stock_model.load_scaler()
        data = self._stock_model.prepare_data()
        scaled_data = self._stock_model.scale_data(data, create=False, save=False)
        self.prepare_prediction_data(scaled_data)
        if self._stock_model.delay_days + self.predicting_days == 1:
            self.predict_data()
        else:
            self.multi_day_predict()   
        self.save_prediction_result()            
    #endregion main methods
    
if __name__ == "__main__":
    import json, sys, argparse
    parser = argparse.ArgumentParser(prog='predict_by_model.py', usage='%(prog)s Stock [options]', description='Train AI-model for one or more stock(s)')
    parser.add_argument('-c', '--config', action='store_true', help='load arguments from config file')
    parser.add_argument("file", help='model file which will be loaded or configure file if -c is specified')
    parser.add_argument('-t', '--path', default='.', help='common path of resource and output')
    parser.add_argument('-d', '--delay-days', default=3, dest="delay", type=int, help='delay days of predicted data relative to real data')
    parser.add_argument('-r', '--predict-days', default=3, dest="predays", type=int, help='delay days of predicted data relative to real data')
    # parser.add_argument('-s', '--sigle', action="store_true", help='process single prediction')
    # parser.add_argument('-m', '--multi', action="store_true", help='process multi-day prediction')
    args = parser.parse_args()
    print("--- arguments ---")
    if args.config == False:
        mp = Model_Prediction(args.mfile, pred_days=args.predays, path=args.path, delay_days=args.delay)
        print(f"Model file: {mp.model_file}")
        print(f"Predict days: {mp.predicting_days}")
        print(f"Path: {mp.stock.path}")
        print(f"Delay Days: {mp.stock.delay_days}")
        mp.process_prediction()
    else:
        print(f"File: {args.file}")
        with open(args.file, 'r') as f:
            config = json.load(f)
        Model_Prediction.CONFIG = config
        mp = Model_Prediction(config['model_file'], period=config['pred_days'])
        mp.process_prediction()
        

        
