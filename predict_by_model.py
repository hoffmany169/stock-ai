
"""
class Model_Prediction
predict trend of stock
NOTE: 新数据应紧接在训练数据之后，保持时间序列连贯性。

"""
import os, json, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from keras.models import load_model
from joblib import load
from stock_model import *
import yfinance as yf
from datetime import datetime, timedelta 
SYMBOL = "IFX.DE"
HISTORY = "3mo"

class Model_Prediction:
    CONFIG = None
    def __init__(self, config_file, predict_days=1, path='.'):
        self._config_file = config_file
        self._predict_days = predict_days
        self._path = path
        self._config = None
        self.load_config()
        self._model_file = self._config[CONFIG.model_file.name]
        self._config[CONFIG.predict_days.name] = self._predict_days
        self._stock_model = Stock_Model(self._config_file)
        self._pred_file_name = f"{self._stock_model.stock_symbols[0]}_{self._stock_model.interval}_{self._stock_model.window_size}_{self._predict_days}"

    #region Property
    @property
    def model_file(self):
        return self._model_file
    
    @property
    def model(self):
        if self._stock_model is None:
            return None
        return self._stock_model.model
        
    @property
    def predicting_days(self):
        return self._predict_days
    
    @predicting_days.setter
    def predicting_days(self, d):
        self._predict_days = d
    #endregion Property

    def load_config(self):
        try:
            with open(self._config_file, 'r') as cfg:
                self._config = json.load(cfg)
        except Exception as e:
            print(f"Load configure file fails: {e}")

    def prepare_prediction_data(self, scaled_data):
        self._prepared_predict_data = np.array(scaled_data[-self._stock_model.window_size:])

    def predict_data(self):
        if self.model is None:
            print("Warning: model is None. It must be created or loaded!")
            return
        predict_data = self.model.predict(self._prepared_predict_data)
        self._Y_pred_actual = self._stock_model.invert_normalized_data(predict_data)
        
    def multi_day_predict(self, days=-1) -> list:
        self._Y_pred_actual = []
        if days > 0:
            self._predict_days = days
        current_sequence = self._prepared_predict_data.copy()

        for _ in range(self._stock_model.delay_days, self._predict_days + self._stock_model.delay_days):
            # 确保输入形状为 (1, window_size, n_features)
            X = np.expand_dims(current_sequence, axis=0)
            next_day_scaled_data = self.model.predict(X, verbose=0)[0][0]
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
            dummy = np.zeros((1, self._stock_model.features.shape[1]))
            dummy[0, 3] = next_day_scaled_data
            predicted_close = self._stock_model.scaler.inverse_transform(dummy)[0, 3]
            self._Y_pred_actual.append(predicted_close)            

    def save_prediction_result(self, sep=':', decimal=','):
        if type(self._Y_pred_actual) is not list:
            result_y_pred_actual = np.array([self._Y_pred_actual]).reshape(-1,1)
        else:    
            result_y_pred_actual = np.array(self._Y_pred_actual).reshape(-1,1)
        row_len = result_y_pred_actual.shape[0]
        last_date = datetime.strptime(self._config[CONFIG.last_date.name], "%Y-%m-%d")
        datetime_column = [last_date + timedelta(days=d) for d in range(row_len)]
        date_str_column = [d.date().isoformat() for d in datetime_column]
        date_column = np.array(date_str_column).reshape(-1, 1)
        result_concat = np.concatenate([date_column, result_y_pred_actual], axis=1)
        result = pd.DataFrame(result_concat, columns=['date', 'predicted price'])
        path_name = os.path.join(self._path, f"{self._pred_file_name}.csv")
        result.to_csv(path_name, sep=sep, decimal=decimal)
        self._config[CONFIG.pred_data_file.name] = path_name

    #region main methods
    def process_prediction(self, period):
        self._stock_model.period = period
        scaled_data = self._stock_model.start(function=MODEL_FUNCTION.predict_model)
        self.prepare_prediction_data(scaled_data)
        if self._stock_model.delay_days + self.predicting_days == 1:
            self.predict_data()
        else:
            self.multi_day_predict()   
        self.save_prediction_result()  
        ## save prediction configure file
        pred_config_file = os.path.join(self._path, f"{self._stock_model.model_basename}_prediction.cfg")
        with open(pred_config_file, 'w') as cfg:
            json.dump(self._config, cfg, indent=4)          
    #endregion main methods

from tkinter import Tk, Label, filedialog, StringVar, messagebox
from tkinter.ttk import Combobox
from Common.CreateGuiElement import CreateGuiElement

PERIOD = {'1mo':'1 month', '3mo':'3 months', '6mo':'6 months'}
class PRED_CONFIG(AutoIndex):
    config_file = ()
    output_dir = ()
    predict_days = ()
    predict_period = ()
class Prediction(Tk):
    CONFIG_FILE = 'config.cfg'
    def __init__(self):
        super().__init__()
        self.title = 'Prediction'
        self.geometry('350x240')
        self.resizable(False, False)
        self._cbox_select = StringVar(self, '3 months')
        self._creator = CreateGuiElement(self)
        self._model_pred = None
        self._config = None
        self._load_config()
        self._create_gui()
        
    def _load_config(self):
        if os.path.exists(Prediction.CONFIG_FILE):
            with open(Prediction.CONFIG_FILE, 'r') as cfg:
                self._config = json.load(cfg)
        else: # create configure file
            self._config = {entry.name : '' for entry in PRED_CONFIG}
            self._config[PRED_CONFIG.config_file.name] = 'config.cfg'
            self._config[PRED_CONFIG.output_dir.name] = '.'
            self._config[PRED_CONFIG.predict_period.name] = '3 months'
            self._config[PRED_CONFIG.predict_days.name] = 7
            with open(Prediction.CONFIG_FILE, 'w') as cfg:
                json.dump(self._config, cfg, indent=4)
            
    def _create_gui(self):
        self._creator.CreateEntry(0, 'Configure File', entry_name='cfg', width_0=14, padx_1=0)\
            .CreateEntry(1, 'Output Folder', entry_name='output', width_0=14, padx_1=0)\
                .CreateEntry(3, 'Prediction Days', entry_name='pred', width_0=14, padx_1=0)
                
        # create combobox
        Label(self, text='Data Period', anchor='e', width=16)\
            .grid(row=2, column=0, padx=10, pady=10)
        self._combobox = Combobox(self, 
                                  values=list(PERIOD.values()),
                                  textvariable=self._cbox_select.get(),
                                  name='period',
                                  width=17)
        self._combobox.grid(row=2, column=1, padx=10, pady=10)
        self._combobox.bind('<<ComboboxSelected>>', self._select_period)
        
        self._creator.CreateButton(4, 0, 'Start', self._start, name='start')\
            .CreateButton(4, 1, 'Exit', self._exit, name='exit')\
                .CreateButton(0, 2, '...', self._select_config, name='sel_cfg', width=2, padx=2)\
                    .CreateButton(1, 2, '...', self._select_output, name='sel_out', width=2, padx=2)
        self._creator.SetEntryText(self._config[PRED_CONFIG.config_file.name], name='cfg')
        self._creator.SetEntryText(self._config[PRED_CONFIG.output_dir.name], name='output')
        self._creator.SetEntryText(self._config[PRED_CONFIG.predict_days.name], name='pred')
        self._creator.SetEntryText(self._config[PRED_CONFIG.predict_period.name], name='period')
    
    def show(self):
        self.mainloop()
    
    def _select_period(self, *args):
        pass
    
    def _select_config(self, *args):
        fn = filedialog.askopenfilename(title='Choose Configure File', initialdir='.', filetypes=[('configure file', '*.cfg')])
        if len(fn) > 0:
            self._creator.SetEntryText(fn, name='cfg')
            
    def _select_output(self, *args):
        dirn = filedialog.askdirectory(title='Choose Output Directory', initialdir='.', mustexist=True)
        if len(dirn) > 0:
            self._creator.SetEntryText(dirn, name='output')
    
    def _start(self):
        cfg_file = self._creator.GetEntryText(name='cfg')
        output_dir = self._creator.GetEntryText(name='output')
        pred_days = self._creator.GetEntryText(name='pred')
        for k, v in PERIOD.items():
            if v == self._cbox_select.get():
                period = k
                break   
        
        self._model_pred = Model_Prediction(cfg_file, predict_days=int(pred_days), path=output_dir)
        self._model_pred.process_prediction(period)
        messagebox.showinfo("Process Prediction", "Process prediction successfully!")
    
    def _exit(self):
        sys.exit()
    

    
if __name__ == "__main__":
    import argparse
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(prog='predict_by_model.py', usage='%(prog)s Stock [options]', description='Train AI-model for one or more stock(s)')
        parser.add_argument("-f", "--file", help='configure file')
        parser.add_argument('-p', '--path', default='.', help='common path of resource and output')
        parser.add_argument('-d', '--predict-days', default=3, dest="predays", type=int, help='delay days of predicted data relative to real data')
        args = parser.parse_args()
        print("--- arguments ---")
        mp = Model_Prediction(args.file, predict_days=int(args.predays), path=args.path)
        print(f"Configure file: {mp._config_file}")
        print(f"Predict days: {mp._predict_days}")
        print(f"Path: {mp._path}")
        mp.process_prediction('3mo')
    else:
        pred = Prediction()
        pred.show()
        

        
