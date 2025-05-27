"""class Stock_Model
predict stock with LSTM. 2 way to use:
------- 1st way: obtain trained model with data from markt ---------
1) create instance:            stock = Stock_Model()
2) load stock data from markt: stock.load_stock()
3) prepare data:               stock.prepare_data()
4) construct data:             stock.create_train_test_data()
5) construct model:            stock.create_LSTM_model()
6) compile model:              stock.compile_model()
7) start training model:       stock.train_model()
8) predict data:               stock.model_predict()
9) evaluate result:            stock.calculate_accuracy()
10)visual result:              stock.visual_result()
------- 2nd way: training model with saved training data ---------
1) create instance:            stock = Stock_Model()
2) load data from file:        stock.load_train_data()
3) construct model:            stock.create_LSTM_model()
4) compile model:              stock.compile_model()
5) start training model:       stock.train_model()
6) load scaler                 stock.load_scaler()
7) predict data:               stock.model_predict()
8) evaluate result:            stock.calculate_accuracy()
9) visual result:              stock.visual_result()
------- 3rd way: training model with new training data ---------
1) create instance:            stock = Stock_Model()
2) load stock data from markt: stock.load_stock()
3) prepare data:               stock.prepare_data()
4) load saved model:           stock.load_model()
5) load scaler                 stock.load_scaler()
6) predict data:               stock.model_predict()
7) evaluate result:            stock.calculate_accuracy()
8) visual result:              stock.visual_result()
------- 4th way: evaluate predicted data from load file ---------
1) create instance:            stock = Stock_Model()
2) load predicted data:        stock.load_predict_data()
3) evaluate data:              stock.evaluate_result()
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import Input
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from curl_cffi import requests
from joblib import dump, load
from datetime import date, timedelta


FILE_TYPE = {'NON':-1, 'MODEL':0, 'SCALER':1, 'TRAIN_DATA':2, 'RESULT_DATA':3, 'PREDICT_DATA':4, "EVALUATE_DATA":5}
SESSION = None
def create_session():
    return requests.Session(impersonate="chrome")
class Stock_Model:
    """
    class Stock_Model
    create model and train model using LTSM for stock markt
    param[stock] str: symbol of stock
    param[period] str: length of history of data to be loaded,
                        they are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    param[interval] str: interval of data to be obtained, they are 1m, 2m, 5m, 15m, 
                        30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo                        
    """
    CLASSES = ["Bull", "Bear"]
    EVALUATE = ["mse", "mae", "rmse", "r2", "acc_score"]
    CONFIG = ['model_name', 'stock', 'period', 'interval', 'win_size', 'delay', 'last_date']
    def __init__(self, stock, period, interval="1d", win_size=60, path='.', delay_days=3):
        self._stock = None
        self._period = period
        self._interval = interval
        self._window_size = win_size # default 60 days
        self._path = path # path of saved file if saving or loading them by default name
        if self._path is not None and not os.path.exists(self._path):
            os.mkdir(os.path.abspath(self._path))
        self._asumpt_delay_days = delay_days
        self._delay_days = self._asumpt_delay_days
        self.day_statistic = [0 for i in range(self._asumpt_delay_days)]
        # record last date of training model
        self._last_date = date.today() - timedelta(days=1)
        self._model_name = None
        if type(stock) == str:
            self._stock = stock
            self._model_name = f"{self._stock}_{self._interval}_{self._window_size}"
        else:
            raise ValueError("stock has only str type!")
        global SESSION
        if SESSION is None:
            SESSION = create_session()
        ### default all features
        self._base_features = np.array([['Open', 'High', 'Low', 'Close', 'Volume', "Dividends", "Stock Splits"]])
        self._features = []
        self._stock_data = None
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._Y_pred_actual = None
        self._Y_test_actual = None
        self._scaled_data = None
        self._model = None
        self._evaluate_result = {entry:None for entry in Stock_Model.EVALUATE}
    
    #region Property
    @property
    def stock_symbol(self):
        return self._stock
    
    @property
    def model_name(self):
        return self._model_name
    
    @property
    def model(self):
        return self._model
        
    @property
    def scaler(self):
        return self._scaler
        
    @property
    def period(self):
        return self._period
    
    @property
    def interval(self):
        return self._interval
            
    @property
    def stock_data(self):
        return self._stock_data

    @property
    def scaled_data(self):
        return self._scaled_data

    @property
    def tested_data_number(self):
        return self._scaled_data.shape[0]
        
    @property
    def session(self):
        return SESSION
        
    @property
    def features(self):
        return self._features

    @property
    def path(self):
        return self._path

    @property
    def delay_days(self):
        return self._asumpt_delay_days

    @property
    def base_features(self):
        return self._base_features
        
    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, ws):
        self._window_size = ws

    @property
    def predicted_data(self):
        return self._Y_pred_actual
    #endregion Property
       
    ### Step 1. obtain stock data from markt
    def load_stock(self, stock=None, period=None, interval=None):
        ## read stock data from stock markt
        global SESSION
        self._stock = stock if stock is not None else self._stock
        self._period = period if period is not None else self._period
        self._interval = interval if interval is not None else self._interval
        try:
            self._stock_data = yf.Ticker(self._stock, session=SESSION).history(period=self._period, interval=self._interval)
            if self._stock_data.empty:
                raise ValueError("Return empty data")
            # save date of yesterday
            length = len(self._stock_data)
            timestamp = self._stock_data.index[length-2]
            self._last_date = f"{timestamp.year}-{timestamp.month}-{timestamp.day}"
        except Exception as e:
            print(f"Exception: {e}")

    ### Step 2. prepare data for training
    def prepare_data(self, start_feature=0, end_feature=5, remove_cols=["Dividends", "Stock Splits"]):
        self._features = np.array([self._base_features[0, start_feature:end_feature]])
        ### remove meaningless columns
        _data_with_valid_features = self._stock_data.drop(columns=remove_cols)
        # remove today data
        return _data_with_valid_features[:-1]
    
    def scale_data(self, data, save=True, create=True):
        if create: # create new scaler
            self._scaler = MinMaxScaler(feature_range=(0, 1))
            self._scaler.fit(data)
        if self._scaler is None:
            raise ValueError("Error: Scaler is None")
        if save: # save scaler
            self.save_scaler()
        self._scaled_data = self._scaler.transform(data)
        return self._scaled_data

    def _create_sequences(self):
        if self.tested_data_number < self._window_size:
            raise ValueError("Too few number of data")
        X, Y = [], []
        for i in range(self.tested_data_number - self._window_size):
            # Input: add all features: Open, High, Low, Close, Volume
            X.append(self._scaled_data[i:i+self._window_size, :]) 
            # Output: predict Close price
            Y.append(self._scaled_data[i+self._window_size, 3]) 
        return np.array(X), np.array(Y)

    ### Step 3. construct data for training
    def create_train_test_data(self, train_data_percentage=0.7, save=True):
        if self.tested_data_number < self._window_size:
            raise ValueError("Too few number of data")
        if train_data_percentage >= 1.0:
            raise ValueError("Percent must less than 1.0")
        X, Y = self._create_sequences()
        split = int(train_data_percentage * len(X))
        self._X_train, self._X_test = X[:split], X[split:]
        self._Y_train, self._Y_test = Y[:split], Y[split:]
        if save: # save normalized train- and test- data
            self.save_train_data()
            
    ### Step 4. construct test model
    def create_LSTM_model(self, load_data_file=None):
        input_layer = Input(shape=(self._X_train.shape[1], self._X_train.shape[2]))
        self._model = Sequential([
            input_layer,
            LSTM(64, return_sequences=True),  # 输入维度为 (窗口大小, 特征数)
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        '''
        self._model = Sequential()
        self._model.add(Input(shape=(self._X_train.shape[1], self._X_train.shape[2])))
        self._model.add(LSTM(64, return_sequences=True, self._X_train.shape[2])))
        self._model.add(Dropout(0.3))
        self._model.add(LSTM(32, return_sequences=False))
        self._model.add(Dropout(0.2))
        self._model.add(Dense(1))
        '''

    ### Step 5. compile model for training
    def compile_model(self, optim='adam', loss='mse'):
        self._model.compile(optimizer=optim, loss=loss)

    ### Step 6. start training
    def train_model(self, early_stop=True, monitor="val_loss", patience=10, epochs=100, batch_size=32, save=True):
        callback=[]
        if early_stop:
            cb_early_stop = EarlyStopping(monitor=monitor, patience=patience)
            callback.append(cb_early_stop)
        self._history = self._model.fit(
            self._X_train, self._Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self._X_test, self._Y_test),
            callbacks=callback,
            verbose=1
        )
        if save: # save model
            self.save_model()
        # create configure file for prediction model
        self.export_training_model_config_file()
        
    ### Step 7. predict data with the trained model
    def model_predict(self):
        '''
        model_predict
        predict data
        param[data] data to be predicted. if it is None, use test data in the model instance
        param[raw_data] boolean: True, data should be normalized, otherwise False
        '''
        if self._model is None:
            print("Warning: model is None. It must be created or loaded!")
            return
        self._Y_predict = self._model.predict(self._X_test)
        self._Y_pred_actual = self.invert_normalized_data(self._Y_predict)
        # 真实值反归一化
        self._Y_test_actual = self.invert_normalized_data(self._Y_test)

    def invert_normalized_data(self, data):
        # 反归一化需要重建完整的多变量矩阵（仅Close列有值，其他列置0）
        dummy_matrix = np.zeros((len(data), self.features.shape[1]))
        dummy_matrix[:, 3] = data.flatten()  # 第4列是Close
        return self._scaler.inverse_transform(dummy_matrix)[:, 3]

    ### Step 8. show result
    def visual_result(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self._Y_test_actual, label='True Close Price')
        plt.plot(self._Y_pred_actual, label='Predicted Close Price', alpha=0.7, linestyle='--')
        plt.title('Multivariate LSTM Stock Price Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    #region Process Methods
    def train_model_with_actual_data(self):
        '''
        train_model_with_actual_data
        the connection to stock yahoo finance markt is neccessary to obtain training data
        for training model
        '''
        self.load_stock()
        data = self.prepare_data()
        self.scale_data(data)
        self.create_train_test_data()
        self.create_LSTM_model()
        self.compile_model()
        self.train_model()
        self.model_predict()
        self.save_predict_data()
        self.evaluate_result()
        self.save_evaluation_data()
        
    def train_model_with_loaded_data(self, data_file=None, 
                                     model_file=None, 
                                     scaler_file=None):
        '''
        train_model_with_actual_data
        the connection to stock yahoo finance markt is not neccessary. the data is loaded from saved file
        for training model
        '''
        self.load_train_data(file_path_name=data_file)
        self.load_keras_model(model_file=model_file)
        self.load_scaler(file_path_name=scaler_file)
        self.model_predict()
        self.save_predict_data()
        self.evaluate_result()
        self.save_evaluation_data()

    def export_training_model_config_file(self):
        config_file_name = f"{self._model_name}.cfg"
        path_name = os.path.join(self._path, config_file_name)
        if os.path.exists(path_name):
            with open(path_name, 'r') as cfg:
                config = json.load(cfg)
            ## correct delay days after evaluation
            config['delay'] = self._delay_days
        else:
            mfile = self.get_path_file_name(None, FILE_TYPE['MODEL'])
            config = dict(zip(Stock_Model.CONFIG, [mfile, self._stock, self._period, self._interval, self._window_size, self._path, self._delay_days, self._last_date]))
        with open(path_name, 'w') as cfg:
            json.dump(config, cfg, indent=4)

    #endregion Process Methods
        
    #region Save Process Data
    def save_training_result(self, file_path_name=None, sep=':', decimal=','):
        result_y_test = np.array(self._Y_test_actual).reshape(-1, 1)
        result_y_test_pred = np.array(self._Y_pred_actual).reshape(-1,1)
        print(result_y_test.shape, result_y_test_pred.shape)
        result = np.concatenate((result_y_test, result_y_test_pred), axis=1)
        print(result.shape)
        result = pd.DataFrame(result, columns=['real price', 'predicted price'])
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE['RESULT_DATA'])
        result.to_csv(path_name, sep=sep, decimal=decimal)

    def save_model(self, model_file=None):
        mfile = self.get_path_file_name(model_file, FILE_TYPE['MODEL'])
        self._model.save(mfile, overwrite=True, include_optimizer=False)

    def save_scaler(self, file_path_name=None):
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE['SCALER'])
        dump(self._scaler, path_name)

    def save_train_data(self, file_path_name=None):
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE['TRAIN_DATA'])
        np.savez(path_name, x_train=self._X_train, y_train=self._Y_train, x_test=self._X_test, y_test=self._Y_test)
    
    def save_predict_data(self, fname=None):
        path_name = self.get_path_file_name(fname, FILE_TYPE['PREDICT_DATA'])
        np.savez(path_name, y_test=self._Y_test_actual, y_pred=self._Y_pred_actual)
        
    def save_evaluation_data(self, fname=None, sep=':', decimal=','):
        result_arr = [np.array(arr).reshape(-1, 1) for name, arr in self._evaluate_result.items()]
        result = np.concatenate(result_arr, axis=1)
        result = pd.DataFrame(result, columns=Stock_Model.EVALUATE)
        path_name = self.get_path_file_name(fname, FILE_TYPE['EVALUATE_DATA'])
        result.to_csv(path_name, sep=sep, decimal=decimal)        
    #endregion Save Process Data
    
    #region Load Process Data
    def load_keras_model(self, model_file=None, compile=True):
        '''
        load_keras_model
        after load model, process step 5 or 6 depending if model is compiled.
        '''
        mfile = self.get_path_file_name(model_file, FILE_TYPE['MODEL'])
        if os.path.exists(mfile):
            basename = os.path.basename(mfile)
            model_name = os.path.splitext(basename)
            self._model_name = model_name[0]
        else:
            raise ValueError("The load data file is not found!")
        self._model = load_model(mfile, compile=False)
        if compile:
            self.compile_model()

    def load_predict_data(self, fname=None):
        path_name = self.get_path_file_name(fname, FILE_TYPE['PREDICT_DATA'])
        datasets = np.load(path_name)
        self._Y_test_actual, self._Y_pred_actual = datasets["y_test"], datasets["y_pred"]
        
    def load_train_data(self, file_path_name=None):
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE['TRAIN_DATA'])
        datasets = np.load(path_name)
        self._X_train, self._Y_train = datasets["x_train"], datasets["y_train"]
        self._X_test, self._Y_test = datasets["x_test"], datasets["y_test"]
        self._features = np.array([self._base_features[0, 0:self._X_train.shape[2]]])

    def load_scaler(self, file_path_name=None):
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE['SCALER'])
        self._scaler = load(path_name)
    #endregion Load Process Data
    
    #region Evaluation Data
    def evaluate_result(self, delay_days=None):
        days = self._asumpt_delay_days + 1
        if delay_days is not None:
            days = delay_days + 1
        self.evaluate_result_data(days)
        self.evaluate_with_direction(days)
        max_day = max(self.day_statistic)
        for i, value in enumerate(self.day_statistic):
            if value == max_day:
                self._delay_days = i
                break
        self.export_training_model_config_file()
        
    def get_nth_day_evaluate_data(self, nth_day):
        days = 0
        if type(nth_day) == str:
            days = int(nth_day)
        else:
            days = nth_day
        if days == 0:
            new_test = self._Y_test_actual
            new_pred = self._Y_pred_actual
        else:
            new_test = np.array(list(self._Y_test_actual)[0:-days])
            new_pred = np.array(list(self._Y_pred_actual)[days:])
        if new_test.shape != new_pred.shape:
            raise ValueError("Error: different data shape in evaluation!")
        return new_test, new_pred
        
    def evaluate_result_data(self, days):
        mse = []
        mae = []
        rmse = []
        r2 = []
        for d in range(days):
            # new_test, new_pred = self.get_nth_day_evaluate_data(d)
            mse.append(self.get_mse(d))
            mae.append(self.get_mae(d))
            rmse.append(np.sqrt(mse[d]))
            r2.append(self.get_r2(d))
        print("Best delay days:")
        mse_day = np.array(mse).argmin()
        self.day_statistic[mse_day] += 1
        self._evaluate_result['mse'] = f"day[{mse_day}]:{mse[mse_day]:.2%}"
        print(f"Mean Squared Error: delay day [{mse_day}], value: [{mse[mse_day]}]")
        
        mae_day = np.array(mae).argmin()
        self.day_statistic[mae_day] += 1
        self._evaluate_result['mae'] = f"day[{mae_day}]:{mae[mae_day]:.2%}"
        print(f"Mean Absolute Error: delay day [{mae_day}], value: [{mae[mae_day]}]")
        
        rmse_day = np.array(rmse).argmin()
        self.day_statistic[rmse_day] += 1
        self._evaluate_result['rmse'] = f"day[{rmse_day}]:{rmse[rmse_day]:.2%}"
        print(f"RMSE: delay day [{rmse_day}], value: [{rmse[rmse_day]}]")
        
        r2_day = np.array(r2).argmax()
        self.day_statistic[r2_day] += 1
        self._evaluate_result['r2'] = f"day[{r2_day}]:{r2[r2_day]:.2%}"
        print(f"R2 Score: delay day [{r2_day}], value: [{r2[r2_day]}]")

    def evaluate_with_direction(self, days):
        # calculate direction
        accuracy = []
        cm = []
        for d in range(days):
            accuracy.append(self.get_direction_rate(d))
        direction_day = np.array(accuracy).argmax()
        self.day_statistic[direction_day] += 1
        self._evaluate_result['acc_score'] = f"day[{direction_day}]:{accuracy[direction_day]:.2%}"
        print(f"Best Correct Rate of Prediction of Direction: delay day [{direction_day}], value [{accuracy[direction_day]:.2%}]")

    def get_path_file_name(self, fn, ftype):
        if fn is None:
            if ftype == FILE_TYPE['MODEL']:
                file_name = f"{self._model_name}.keras"
            elif ftype == FILE_TYPE['SCALER']:
                file_name = f"{self._model_name}_scaler.joblib"
            elif ftype == FILE_TYPE['PREDICT_DATA']:
                file_name = f"{self._model_name}_pred.npz"
            elif ftype == FILE_TYPE['RESULT_DATA']:
                file_name = f"{self._model_name}_result.csv"
            elif ftype == FILE_TYPE['TRAIN_DATA']:
                file_name = f"{self._model_name}_train.npz"
            elif ftype == FILE_TYPE['EVALUATE_DATA']:
                file_name = f"{self._model_name}_eval.csv"
            else:
                raise ValueError("Error: false file type")
            path_name = os.path.join(self._path, file_name)
        else:
            path_name = fn
        return path_name
        
    def get_mse(self, nth_day):
        from sklearn.metrics import mean_squared_error
        new_test, new_pred = self.get_nth_day_evaluate_data(nth_day)
        return mean_squared_error(new_test, new_pred)
        
    def get_mae(self, nth_day):
        from sklearn.metrics import mean_absolute_error
        new_test, new_pred = self.get_nth_day_evaluate_data(nth_day)
        return mean_absolute_error(new_test, new_pred)

    def get_rmse(self, nth_day):
        mse = self.get_mse(nth_day)
        return np.sqrt(mse)
        
    def get_r2(self, nth_day):
        from sklearn.metrics import r2_score
        new_test, new_pred = self.get_nth_day_evaluate_data(nth_day)
        return r2_score(new_test, new_pred)

    def get_direction_rate(self, nth_day):
        from sklearn.metrics import accuracy_score
        new_test, new_pred = self.get_nth_day_evaluate_data(nth_day)
        actual_direction = np.where(np.diff(new_test) > 0, 1, 0)
        predict_direction = np.where(np.diff(new_pred) > 0, 1, 0)
        return accuracy_score(actual_direction, predict_direction)

    def get_confusion_matrix(self, nth_day):
        from sklearn.metrics import confusion_matrix
        new_test, new_pred = self.get_nth_day_evaluate_data(nth_day)
        actual_direction = np.where(np.diff(new_test) > 0, 1, 0)
        predict_direction = np.where(np.diff(new_pred) > 0, 1, 0)
        cm = confusion_matrix(actual_direction, predict_direction)
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        ## Accuracy = (TP + TN)/(TP + TN + FP + FN): ratio of all positive answers
        Accuracy = (TP + TN)/(TP + TN + FP + FN)
        ## Sensitivity = Recall = TP/(TP + FN): ratio of correct predition of positive answers in all correct predictions
        Sensitivity = TP/(TP + FN)
        ## Specificity = TN/(TN + FP): 
        Specificity = TN/(TN + FP)
        ## Precision = TP/(TP + FP): correct ratio of all predicting positive answers
        Precision = TP/(TP + FP)
        ## F1 score: F1_Score = 2*Precision * Sensitivity / (Precision + Sensitivity)
        F1_Score = 2*Precision * Sensitivity / (Precision + Sensitivity)
        print(f"=== Confusion Matrix [Day{nth_day}] ===")
        print(f"Accuracy: {Accuracy:.2%}")
        print(f"Specificity: {Specificity:.2%}")
        print(f"Sensitivity: {Sensitivity:.2%}")
        print(f"Precision: {Precision:.2%}")
        print(f"F1 Score: {F1_Score:.2%}")
    #endregion Evaluation Data

if __name__ == "__main__":
    import json, sys, argparse
    parser = argparse.ArgumentParser(prog='stock_model.py', usage='%(prog)s Stock [options]', description='Train AI-model for one or more stock(s)')
    parser.add_argument('stock', help='stock symbol to be loaded')
    parser.add_argument('action', help='process action: tmf - train model from finance markt data, tmd - train model from data file, evl - evaluate data, cfm - calculate confusion matrix data, mse - mean squared error, mas - mean absolute error, rmse - sqared mse, r2 - r2 score, ars - accuracy score')
    parser.add_argument('-f', '--file', help='load arguments from file')
    parser.add_argument('-p', '--period', default='10y', help='period of stock to be loaded')
    parser.add_argument('-i', '--interval',default='1d', help='interval of stock data')
    parser.add_argument('-w', '--window-size', default=60, dest='win_size', type=int, help='window size of training model')
    parser.add_argument('-t', '--path', default='.', help='common path of resource and output')
    parser.add_argument('-d', '--delay-days', default=3, dest="delay", type=int, help='delay days of predicted data relative to real data')
    parser.add_argument('-n', '--nth-day', dest="day", type=int, default=1, help='Nth-delay day for confusion matrix, mse, mae, rmse, r2 and ars')
    parser.add_argument('-c', '--config', action='store_true', help='create a configure file for prediction')
    args = parser.parse_args()
    print("--- arguments ---")
    if args.file is None:
        print(f"Stock: {args.stock}")
        print(f"Period: {args.period}")
        print(f"Interval: {args.interval}")
        print(f"Window Size: {args.win_size}")
        print(f"Path: {args.path}")
        print(f"Delay Days: {args.delay}")
        print(f"Config: {args.config}")
        sm = Stock_Model(args.stock, args.period, interval=args.interval, win_size=args.win_size, path=args.path, delay_days=args.delay)
    else:
        '''
        configure file for stock model:
        {'stock': str or list,
         'period' : str, # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
         'interval': str, # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
         'win_size': int,
         'path': str,
         'delay': int
        }
        '''
        print(f"File: {args.file}")
        with open(args.file, 'r') as f:
            config = json.load(f)
        sm = Stock_Model(config['stock'], period=config['period'], interval=config['interval'], win_size=config['win_size'], path=config['path'], delay_days=config['delay'])
    
    if args.action == 'tmf':
        sm.train_model_with_actual_data()
    elif args.action == 'tmd':
        sm.train_model_with_loaded_data()
    elif args.config == True:
         for s in sm.stock_symbols:
            sm.current_stock = s
            sm.save_config_file()
    else:
        if args.action == 'evl':
            for index, s in enumerate(sm.stock_symbols):
                sm.current_stock = s
                sm.load_predict_data()
                sm.evaluate_result()
        elif args.action == 'cfm':
            for index, s in enumerate(sm.stock_symbols):
                sm.current_stock = s
                sm.load_predict_data()
                sm.get_confusion_matrix(args.day)
        elif args.action == 'mse':
            for index, s in enumerate(sm.stock_symbols):
                sm.current_stock = s
                sm.load_predict_data()
                print(f"mse[day{args.day}]: {sm.get_mse(args.day)}")
        elif args.action == 'mae':
            for index, s in enumerate(sm.stock_symbols):
                sm.current_stock = s
                sm.load_predict_data()
                print(f"mae[day{args.day}]: {sm.get_mae(args.day)}")
        elif args.action == 'rmse':
            for index, s in enumerate(sm.stock_symbols):
                sm.current_stock = s
                sm.load_predict_data()
                print(f"rmse[day{args.day}]: {sm.get_rmse(args.day)}")
        elif args.action == 'r2':
            for index, s in enumerate(sm.stock_symbols):
                sm.current_stock = s
                sm.load_predict_data()
                print(f"r2[day{args.day}]: {sm.get_r2(args.day)}")
        elif args.action == 'ars':
            for index, s in enumerate(sm.stock_symbols):
                sm.current_stock = s
                sm.load_predict_data()
                print(f"mse[day{args.day}]: {sm.get_direction_rate(args.day)}")
        else:
            raise ValueError("No action is available")
            
            
            