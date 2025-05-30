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
import os, json
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
from Common.AutoNumber import AutoIndex
from typing import overload

class CONFIG(AutoIndex):
    stock = ()
    stock_period = ()
    stock_interval = ()
    window_size = ()
    delay_days = ()
    last_date = ()
    model_file = ()
    train_data_file = ()
    scaler_file = ()
    train_result_file = ()
    predict_days = ()
    pred_data_file = ()
    evaluate_data_file = ()

class FILE_TYPE(AutoIndex):
    MODEL = ()
    SCALER = ()
    TRAIN_DATA = ()
    RESULT_DATA = ()
    PREDICT_DATA = ()
    EVALUATE_DATA = ()

class EVALUATE(AutoIndex):
    mse = ()
    mae = ()
    rmse = ()
    r2 = ()
    acc_score = ()

class MODEL_FUNCTION(AutoIndex):
    retrain_model = ()
    predict_model = ()
    evaluation = ()
    confusion_matrix = ()
    

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
    param[win_size] int: size of data in one step for training
    param[delay_days] int: normally predicted data is slower than real data. this value 
                        is an assumped maximal days to be evaluated. The final value will
                        be given after evaluation.
    param[online] bool: True(default): get stock data directly from finance markt by ticker
                        or download; 
                        False: load stock data from local file
    """
    CLASSES = ["Bull", "Bear"]
    
    def __init__(self, stock_info:list|str, period="10y", interval="1d", win_size=60, delay_days=3):
        if type(stock_info) is str:
            self._config_file = stock_info
            self._stocks = []
            self._online = False
            self._init_from_config()
        elif type(stock_info) is list:
            self._stocks = stock_info
            self._online = True
            self._stock_config = None
            self._current_stock = ''
            self._period = period
            self._interval = interval
            self._window_size = win_size # default 60 days
            self._assumpt_delay_days = delay_days
            self._delay_days = self._assumpt_delay_days
            self._model_name = None
            self._path = None
        else:
            raise ValueError("Error: invalid argument!")
        global SESSION
        if SESSION is None:
            SESSION = create_session()
        self._train_stock_model = None

    def _init_from_config(self):
        with open(self._config_file, 'r') as cfg:
            self._stock_config = json.load(cfg)
        self._stocks.append(self._stock_config[CONFIG.stock.name])
        self._period = self._stock_config[CONFIG.stock_period.name]
        self._interval = self._stock_config[CONFIG.stock_interval.name]
        self._window_size = self._stock_config[CONFIG.window_size.name]
        self._delay_days = self._stock_config[CONFIG.delay_days.name]
        # [STOCK_SYMBOL, INTERVAL, WIN_SIZE, PATH]
        parts = Stock_Model.parse_stock_info_from_model_file(self._config_file)
        if len(parts) == 4:
            self._path = parts[3]
        else:
            self._path = '.'
        self._model_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
        
    #region Property
    @property
    def stock_symbols(self):
        return self._stocks
        
    @property
    def model_basename(self):
        return self._model_name
        
    @property
    def model(self):
        if self._train_stock_model is None:
            return None
        return self._train_stock_model.model
    
    @property
    def features(self):
        if self._train_stock_model is None:
            return None
        return self._train_stock_model.features
        
    @property
    def scaler(self):
        if self._train_stock_model is None:
            return None
        return self._train_stock_model.scaler
        
    @property
    def period(self):
        return self._period
    
    @period.setter
    def period(self, p):
        self._period = p
    
    @property
    def interval(self):
        return self._interval

    @property
    def delay_days(self):
        return self._delay_days

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, ws):
        self._window_size = ws
    #endregion property

    @staticmethod
    def parse_stock_info_from_model_file(model_file_name):
        basename = os.path.basename(model_file_name)
        fn = os.path.splitext(basename)
        model_path = model_file_name.split(basename)
        ## stock symbol, period, interval, window size
        parts = fn[0].split('_')
        if len(parts) < 3:
            raise ValueError("Error: The format of model file name is invalid!")
        parts.append(model_path[0] if len(model_path[0]) > 0 else '.')
        # [STOCK_SYMBOL, INTERVAL, WIN_SIZE, PATH]
        return parts

    def start(self, function:MODEL_FUNCTION=None, nth_day=1):
        if self._online:
            for s in self._stocks:
                self._model_name = f"{s}_{self._interval}_{self._window_size}"
                stock_data = self.load_stock(s)
                if stock_data is None:
                    raise ValueError("Error: cannot load stock data!!")
                output_path = s
                if output_path is not None and not os.path.exists(output_path):
                    os.mkdir(output_path)
                self._path = output_path
                self._train_stock_model = TrainStockModel(self._stock_config, stock_data=stock_data, stock_path=output_path)
                config = self._train_stock_model.train_model_with_actual_data()
                self.export_training_model_config_file(config)
        else:
            if function == MODEL_FUNCTION.predict_model:
                stock_data = self.load_stock(self._stocks[0])
                if stock_data is None:
                    raise ValueError("Error: cannot load stock data!!")
                self._train_stock_model = TrainStockModel(self._stock_config, stock_path=self._path)
                self._train_stock_model.stock_data = stock_data
                self._train_stock_model.load_keras_model(self._stock_config[CONFIG.model_file.name])
                self._train_stock_model.load_scaler()
                data = self._train_stock_model.prepare_data()
                return self._train_stock_model.scale_data(data, create=False, save=False)
            else:
                self._train_stock_model = TrainStockModel(self._stock_config, stock_path=self._path)
                if function == MODEL_FUNCTION.retrain_model:
                    config = self._train_stock_model.train_model_with_loaded_data(model_file=self._config_file[CONFIG.model_file.name])
                    self.export_training_model_config_file(config)
                elif function == MODEL_FUNCTION.confusion_matrix:
                    self._train_stock_model.calculate_confusion_matrix(nth_day)
                elif function == MODEL_FUNCTION.evaluation:
                    self._train_stock_model.evaluate_result(load_data=True, 
                                                            delay_days=self._stock_config[CONFIG.delay_days.name])
                else:
                    print("Error: Undefined model function")

    ### Step 1. obtain stock data from markt
    def load_stock(self, stock):
        ## read stock data from stock markt
        global SESSION
        try:
            stock_data = yf.Ticker(stock, session=SESSION).history(period=self._period, interval=self._interval)
            if stock_data.empty:
                raise ValueError("Return empty data")
            # save date of yesterday
            length = len(stock_data)
            timestamp = stock_data.index[length-2]
            iso_str = timestamp.isoformat()
            iso_date = iso_str.split('T')
            # record last date of training model
            # last_date = f"{iso_date[0]}-{iso_date[1]}-{iso_date[2]}"
            last_date = iso_date[0]
            self._stock_config = {i.name : None for i in CONFIG}
            self._stock_config[CONFIG.stock.name] = stock
            self._stock_config[CONFIG.stock_period.name] = self._period
            self._stock_config[CONFIG.stock_interval.name] = self._interval
            self._stock_config[CONFIG.window_size.name] = self._window_size
            self._stock_config[CONFIG.delay_days.name] = self._delay_days
            self._stock_config[CONFIG.last_date.name] = last_date
        except Exception as e:
            print(f"Exception: {e}")
            stock_data = None
        finally:
            return stock_data

    def export_training_model_config_file(self, config):
        config_file_name = f"{self._model_name}.cfg"
        if self._path is not None:
            path_name = os.path.join(self._path, config_file_name)
        else:
            path_name = config_file_name
        with open(path_name, 'w') as cfg:
            json.dump(config, cfg, indent=4)


class TrainStockModel:
    def __init__(self, stock_config:str, stock_data=None, stock_path='.'):
        if stock_data is not None:
            self._stock_data = stock_data
            self._stock_config = stock_config
            # this is an output path
            self._stock_path = stock_path
            self.__init_vars__()
        else:
            self._stock_config = stock_config
            ## this is an input path
            if not os.path.exists(stock_path):
                raise ValueError(f"Error: cannot open input path: {stock_path}")
            self._stock_path = stock_path
            self.__init_vars__()

    def __init_vars__(self):
        ### default all features
        self._base_features = np.array([['Open', 'High', 'Low', 'Close', 'Volume', "Dividends", "Stock Splits"]])
        self._features = []
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._Y_pred_actual = None
        self._Y_test_actual = None
        self._scaled_data = None
        self._model = None
        self._evaluate_result = {entry.name:None for entry in EVALUATE}
        self._asumpt_delay_days = self._stock_config[CONFIG.delay_days.name]
        self.day_statistic = [0 for i in range(self._asumpt_delay_days*2)]
        self._model_file_name = f"{self._stock_config[CONFIG.stock.name]}_{self._stock_config[CONFIG.stock_interval.name]}_{self._stock_config[CONFIG.window_size.name]}"

    #region Property
    @property
    def stock_data(self):
        return self._stock_data

    @stock_data.setter
    def stock_data(self, s):
        self._stock_data = s

    @property
    def model_config(self):
        return self._stock_config

    @property
    def model(self):
        return self._model
        
    @property
    def scaler(self):
        return self._scaler
        
    @property
    def scaled_data(self):
        return self._scaled_data

    @property
    def tested_data_number(self):
        return self._scaled_data.shape[0]
        
    @property
    def predicted_data(self):
        return self._Y_pred_actual

    @property
    def base_features(self):
        return self._base_features
        
    @property
    def features(self):
        return self._features

    #endregion Property

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
        ws = int(self._stock_config[CONFIG.window_size.name])
        if self.tested_data_number < ws:
            raise ValueError("Too few number of data")
        X, Y = [], []
        for i in range(self.tested_data_number - ws):
            # Input: add all features: Open, High, Low, Close, Volume
            X.append(self._scaled_data[i:i+ws, :]) 
            # Output: predict Close price
            Y.append(self._scaled_data[i+ws, 3]) 
        return np.array(X), np.array(Y)

    ### Step 3. construct data for training
    def create_train_test_data(self, train_data_percentage=0.7, save=True):
        if self.tested_data_number < int(self._stock_config[CONFIG.window_size.name]):
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
        data = self.prepare_data()
        self.scale_data(data)
        self.create_train_test_data()
        self.create_LSTM_model()
        self.compile_model()
        self.train_model()
        self.model_predict()
        self.save_predict_data()
        self.save_training_result()
        self.evaluate_result()
        self.save_evaluation_data()
        return self._stock_config
        
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
        self.save_training_result()
        self.evaluate_result()
        self.save_evaluation_data()
        return self._stock_config
    
    def calculate_confusion_matrix(self, nth_day):
        self.load_predict_data(self._stock_config[CONFIG.pred_data_file.name])
        self.get_confusion_matrix(nth_day)
    
    #endregion Process Methods
        
    #region Save Process Data
    def save_training_result(self, file_path_name=None, sep=':', decimal=','):
        result_y_test = np.array(self._Y_test_actual).reshape(-1, 1)
        result_y_test_pred = np.array(self._Y_pred_actual).reshape(-1,1)
        print(result_y_test.shape, result_y_test_pred.shape)
        result = np.concatenate((result_y_test, result_y_test_pred), axis=1)
        print(result.shape)
        result = pd.DataFrame(result, columns=['real price', 'predicted price'])
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE.RESULT_DATA)
        result.to_csv(path_name, sep=sep, decimal=decimal)
        self._stock_config[CONFIG.train_result_file.name] = path_name

    def save_model(self, model_file=None):
        mfile = self.get_path_file_name(model_file, FILE_TYPE.MODEL)
        self._model.save(mfile, overwrite=True, include_optimizer=False)
        self._stock_config[CONFIG.model_file.name] = mfile

    def save_scaler(self, file_path_name=None):
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE.SCALER)
        dump(self._scaler, path_name)
        self._stock_config[CONFIG.scaler_file.name] = path_name

    def save_train_data(self, file_path_name=None):
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE.TRAIN_DATA)
        np.savez(path_name, x_train=self._X_train, y_train=self._Y_train, x_test=self._X_test, y_test=self._Y_test)
        self._stock_config[CONFIG.train_data_file.name] = path_name
    
    def save_predict_data(self, fname=None):
        path_name = self.get_path_file_name(fname, FILE_TYPE.PREDICT_DATA)
        np.savez(path_name, y_test=self._Y_test_actual, y_pred=self._Y_pred_actual)
        self._stock_config[CONFIG.pred_data_file.name] = path_name
        
    def save_evaluation_data(self, fname=None, sep=':', decimal=','):
        result_arr = [np.array(arr).reshape(-1, 1) for name, arr in self._evaluate_result.items()]
        result = np.concatenate(result_arr, axis=1)
        evl_cols = [i.name for i in EVALUATE]
        result = pd.DataFrame(result, columns=evl_cols)
        path_name = self.get_path_file_name(fname, FILE_TYPE.EVALUATE_DATA)
        result.to_csv(path_name, sep=sep, decimal=decimal)        
        self._stock_config[CONFIG.evaluate_data_file.name] = path_name
#endregion Save Process Data
    
    #region Load Process Data
    def load_keras_model(self, model_file=None, compile=True):
        '''
        load_keras_model
        after load model, process step 5 or 6 depending if model is compiled.
        '''
        mfile = self.get_path_file_name(model_file, FILE_TYPE.MODEL)
        if os.path.exists(mfile):
            basename = os.path.basename(mfile)
            model_name = os.path.splitext(basename)
            self._model_file_name = model_name[0]
        else:
            raise ValueError("The load data file is not found!")
        self._model = load_model(mfile, compile=False)
        if compile:
            self.compile_model()

    def load_predict_data(self, fname=None):
        path_name = self.get_path_file_name(fname, FILE_TYPE.PREDICT_DATA)
        datasets = np.load(path_name)
        self._Y_test_actual, self._Y_pred_actual = datasets["y_test"], datasets["y_pred"]
        
    def load_train_data(self, file_path_name=None):
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE.TRAIN_DATA)
        datasets = np.load(path_name)
        self._X_train, self._Y_train = datasets["x_train"], datasets["y_train"]
        self._X_test, self._Y_test = datasets["x_test"], datasets["y_test"]
        self._features = np.array([self._base_features[0, 0:self._X_train.shape[2]]])

    def load_scaler(self, file_path_name=None):
        path_name = self.get_path_file_name(file_path_name, FILE_TYPE.SCALER)
        self._scaler = load(path_name)
    #endregion Load Process Data
    
    #region Evaluation Data
    def evaluate_result(self, load_data=False, delay_days=None):
        if load_data:
            self.load_predict_data(self._stock_config[CONFIG.pred_data_file.name])
        days = self._asumpt_delay_days + 1
        if delay_days is not None:
            days = delay_days + 1
        else:
            days = delay_days
        self.evaluate_result_data(days)
        self.evaluate_with_direction(days)
        print(f"==> Best result of days: {self.day_statistic}")
        max_day = max(self.day_statistic)
        for i, value in enumerate(self.day_statistic):
            if value == max_day:
                self._delay_days = i
                break
        self._stock_config[CONFIG.delay_days.name] = self._delay_days

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
            if ftype == FILE_TYPE.MODEL:
                file_name = f"{self._model_file_name}.keras"
            elif ftype == FILE_TYPE.SCALER:
                file_name = f"{self._model_file_name}_scaler.joblib"
            elif ftype == FILE_TYPE.PREDICT_DATA:
                file_name = f"{self._model_file_name}_pred.npz"
            elif ftype == FILE_TYPE.RESULT_DATA:
                file_name = f"{self._model_file_name}_result.csv"
            elif ftype == FILE_TYPE.TRAIN_DATA:
                file_name = f"{self._model_file_name}_train.npz"
            elif ftype == FILE_TYPE.EVALUATE_DATA:
                file_name = f"{self._model_file_name}_eval.csv"
            else:
                raise ValueError("Error: false file type")
            path_name = os.path.join(self._stock_path, file_name)
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
    import argparse
    parser = argparse.ArgumentParser(prog='stock_model.py', usage='%(prog)s Stock [options]', description='Train AI-model for one or more stock(s)')
    parser.add_argument('action', help='process action: tmf - train model from finance markt data, tmd - train model from data file, evl - evaluate data, cfm - calculate confusion matrix data, mse - mean squared error, mas - mean absolute error, rmse - sqared mse, r2 - r2 score, ars - accuracy score')
    parser.add_argument('-s', '--stock', action='append', help='stock symbol list to be loaded')
    parser.add_argument('-f', '--file', help='load arguments from file')
    parser.add_argument('-p', '--period', default='10y', help='period of stock to be loaded')
    parser.add_argument('-i', '--interval',default='1d', help='interval of stock data')
    parser.add_argument('-w', '--window-size', default=60, dest='win_size', type=int, help='window size of training model')
    parser.add_argument('-d', '--delay-days', default=3, dest="delay", type=int, help='delay days of predicted data relative to real data')
    parser.add_argument('-n', '--nth-day', dest="day", type=int, default=1, help='Nth-delay day for confusion matrix, mse, mae, rmse, r2 and ars')
    args = parser.parse_args()
    print("--- arguments ---")
    print(f"Stock: {args.stock}")
    print(f"Period: {args.period}")
    print(f"Interval: {args.interval}")
    print(f"Window Size: {args.win_size}")
    print(f"File: {args.file}")
    print(f"Delay Days: {args.delay}")
    print(f"Nth-Day: {args.day}")

    if args.action == 'tmf':
        sm = Stock_Model(args.stock, args.period, interval=args.interval, win_size=args.win_size, delay_days=args.delay)
        sm.start()
    elif args.action == 'tmd':
        sm = Stock_Model(args.file)
        sm.start()
    else:
        sm = Stock_Model(args.file)
        if args.action == 'evl':
            sm.start(function=MODEL_FUNCTION.evaluation, nth_day=args.day)
        elif args.action == 'cfm':
            sm.start(function=MODEL_FUNCTION.confusion_matrix, nth_day=args.day)
        else:
            raise ValueError("No action is available")
            
            
            