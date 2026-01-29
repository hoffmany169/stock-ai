"""
Docstring for ModelIO
handle with saving or loading model training parameters and model data.
for example:

sm = StockModel('NVDA') # stock model
sm.start_date = '2023-1-1'
sm.end_date = '2023-12-31'
sm.load_historical_data()

sf = StockFeature() # features
ss = LSTMModelTrain(sm, sf.get_features()) #select stock
ss.process_train_data()

mio = ModelSaverLoader('models', ticker_symbol)
mio.set_model_train_data(MODEL_TRAIN_DATA.model, sm.model)
mio.set_model_train_data(MODEL_TRAIN_DATA.scaler, ss.scaler)
mio.set_model_train_data(MODEL_TRAIN_DATA.parameters, ss.create_model_parameters())
readme = mio.create_readme()
mio.set_model_train_data(MODEL_TRAIN_DATA.readme, readme)
mio.set_model_train_data(MODEL_TRAIN_DATA.train_history, ss.train_history)
mio.set_model_train_data(MODEL_TRAIN_DATA.performance, ss.performance)
mio.set_model_train_data(MODEL_TRAIN_DATA.model_summary, ss.get_model_summary())
"""

import pickle
import json, os, sys
import keras
from datetime import datetime
from StockDefine import MODEL_TRAIN_DATA, LTSM_MODEL_PARAM, TICKER_DATA_PARAM

class ModelSaverLoader:
    FILE_NAME_DEFINE = {MODEL_TRAIN_DATA.ticker_data.name: 'ticker_data.csv',
                        MODEL_TRAIN_DATA.ticker_data_params.name: 'ticker_data_params.json', 
                        MODEL_TRAIN_DATA.model.name: '_model.keras', 
                        MODEL_TRAIN_DATA.scaler.name: 'scaler.pkl', 
                        MODEL_TRAIN_DATA.parameters.name: 'params.json',
                        MODEL_TRAIN_DATA.train_history.name: 'history.pkl',
                        MODEL_TRAIN_DATA.performance.name: 'performance.json',
                        MODEL_TRAIN_DATA.readme.name: 'README.md', 
                        MODEL_TRAIN_DATA.model_summary.name: 'model_summary.txt'}
    def __init__(self, directory, ticker_symbol=None, save=True):
        """
        Docstring for __init__
        
        :param self: Description
        :param directory: super directory, in which all data will be saved
        :param ticker_symbol: str
        :param save: save mode or load mode
        """
        self._save_model_mode = save
        self._directory = None
        self._ticker_symbol = None
        self._model_train_data = dict(zip([t for t in MODEL_TRAIN_DATA], [None]*len(MODEL_TRAIN_DATA)))
        self._model_io_functions = dict(zip([t for t in MODEL_TRAIN_DATA], [None]*len(MODEL_TRAIN_DATA)))
        self._init(directory, ticker_symbol)

    @property
    def save_model_mode(self):
        return self._save_model_mode
    @save_model_mode.setter
    def save_model_mode(self, state):
        if self._save_model_mode == state:
            return
        self._save_model_mode = state
        self._init()

    def _init(self, directory=None, ticker_symbol=None):
        if directory is None and ticker_symbol is None:
            self._init_model_functions()
            return
        if self._save_model_mode:
            # in this case, directory is a parent directory, in which ticker data will be saved
            if ticker_symbol is None:
                raise ValueError("Ticker symbol is not set in saving mode.")
            self._ticker_symbol = ticker_symbol
            self.timestamp = datetime.now().strftime('%Y%m%d_%H')
            self._directory = os.path.join(directory, f'{ticker_symbol}_{self.timestamp}')
        else: # in this case, directory is where ticker data are saved
            if ticker_symbol is None:
                # parse ticker name
                base_name = os.path.basename(directory)
                parts = base_name.split('_')
                print(parts)
                self._ticker_symbol = parts[0]
                self._directory = directory
            else:
                if ticker_symbol not in directory:
                    raise ValueError("No ticker symbol is found in directory.")
                self._ticker_symbol = ticker_symbol
                self._directory = directory
        self._init_model_functions()
        self.readme_content = ''

    @property
    def ticker_symbol(self):
        return self._ticker_symbol
    
    @property
    def directory(self):
        return self._directory

    def _parse_parent_directory(self, directory):
        # parse parent directory
        parts = directory.split(os.sep)
        self._directory = os.sep.join(parts[:-1])
        print(f"MIO Directory: {self._directory}")

    def _init_model_functions(self):
        if self._directory is None:
            raise ValueError("Directory is not initialized!")
        # initialize functions
        for data_type in MODEL_TRAIN_DATA:
            if self._save_model_mode:
                print(f'save function name: _save_{data_type.name}')
                self._model_io_functions[data_type] = getattr(self, f'_save_{data_type.name}')
            else:
                print(f'load function name: _load_{data_type.name}')
                self._model_io_functions[data_type] = getattr(self, f'_load_{data_type.name}')
#region save data functions
    def _save_ticker_data(self):
        if len(self._model_train_data[MODEL_TRAIN_DATA.ticker_data]) > 1:
            file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.ticker_data.name])
            data = self._model_train_data[MODEL_TRAIN_DATA.ticker_data]
            data.to_csv(file, compression='zip')
            print(f"✓ 原始数据已保存至: {file}")
            return True
        return False

    def _save_ticker_data_params(self):
        # 2. 保存数据下载参数
        params = self._model_train_data[MODEL_TRAIN_DATA.ticker_data_params]
        if len(params) == len(TICKER_DATA_PARAM):
            params_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.ticker_data_params.name])
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"✓ 数据下载参数已保存至: {params_path}")
            return True
        return False

    def _save_model(self):
        # 1. 保存Keras模型
        model = self._model_train_data[MODEL_TRAIN_DATA.model]
        if model is None:
            print("Model object is invalid.")
            return False
        model_name = f"{self._ticker_symbol}{ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.model.name]}"
        model_save_path = os.path.join(self._directory, model_name)
        keras.saving.save_model(model, model_save_path)
        print(f"✓ 模型已保存至: {model_save_path}")
        return True

    def _save_scaler(self):
        # 2. 保存Scaler
        if self._model_train_data[MODEL_TRAIN_DATA.scaler]:
            scaler_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.scaler.name])
            scaler = self._model_train_data[MODEL_TRAIN_DATA.scaler]
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"✓ Scaler已保存至: {scaler_path}")
            return True
        return False

    def _save_parameters(self):
        # 4. 保存参数
        params = self._model_train_data[MODEL_TRAIN_DATA.parameters]
        if len(params) == len(LTSM_MODEL_PARAM):
            params_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.parameters.name])
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"✓ 参数已保存至: {params_path}")
            return True
        return False

    def _save_train_history(self):
        # 5. 保存训练历史（如果有）
        # if hasattr(self.model, 'history') and self.model.history:
        if self._model_train_data[MODEL_TRAIN_DATA.train_history]:
            history_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.train_history.name])
            history = self._model_train_data[MODEL_TRAIN_DATA.train_history]
            with open(history_path, 'wb') as f:
                pickle.dump(history, f)
            print(f"✓ 训练历史已完整保存至: {history_path}")
            return True
        return False

    def _save_performance(self):
        if self._model_train_data[MODEL_TRAIN_DATA.performance]:
            perf_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.performance.name])
            performance = self._model_train_data[MODEL_TRAIN_DATA.performance]
            with open(perf_path, 'w') as f:
                json.dump(performance, f, indent=2)
            print(f"✓ 评估表现已保存至: {perf_path}")
            return True
        return False

    # 6. 创建README文件
    def _save_readme(self):
        if self._model_train_data[MODEL_TRAIN_DATA.readme]:
            readme_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.readme.name])
            readme = self._model_train_data[MODEL_TRAIN_DATA.readme]
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme)
            print(f"✓ README已保存至: {readme_path}")
            return True
        return False

    def _save_model_summary(self):
        if self._model_train_data[MODEL_TRAIN_DATA.model_summary]:
            summary_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.model_summary.name])
            summary = self._model_train_data[MODEL_TRAIN_DATA.model_summary]
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"✓ 模型总览已保存至: {summary_path}")
            return True
        return False
#endregion save data functions

#region load data functions
    ## load model data from disk
    def _load_ticker_data(self):
        import pandas as pd
        file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.ticker_data.name])
        try:
            if os.path.exists(file):
                self._model_train_data[MODEL_TRAIN_DATA.ticker_data] = pd.read_csv(file, compression='zip')
                print(f"✓ 股票数据已加载")
                return True
            else:
                self._model_train_data[MODEL_TRAIN_DATA.ticker_data] = None
                print(f"[{file}] doesn't exist !")
        except Exception as e:            
            raise(f"股票数据已加载失败: {str(e)}")
        return False

    def _load_ticker_data_params(self):
        # 3. 加载数据下载参数
        params_file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.ticker_data_params.name])
        try:
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.ticker_data_params] = json.load(f)
                print(f"✓ 数据下载参数已加载")
                return True
            else:
                self._model_train_data[MODEL_TRAIN_DATA.parameters] = None
                print(f"[{params_file}] doesn't exist !")
        except Exception as e:            
            raise(f"加载数据下载参数失败: {str(e)}")
        return False

    def _load_model(self):
        # 1. 加载模型
        model_file_name = f"{self._ticker_symbol}{ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.model.name]}"
        model_file = os.path.join(self._directory, model_file_name)
        try:
            if os.path.exists(model_file):
                self._model_train_data[MODEL_TRAIN_DATA.model] = keras.models.load_model(model_file)
                print(f"✓ 模型已加载: {model_file}")
                return True
            else:
                self._model_train_data[MODEL_TRAIN_DATA.model] = None
                print(f"[{model_file}] doesn't exist !")
        except Exception as e:
            print(f"模型已加载失败: {str(e)}")
        return False

    def _load_scaler(self):
        # 2. 加载Scaler
        scaler_file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.scaler.name])
        try:
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.scaler] = pickle.load(f)
                print(f"✓ Scaler已加载")
                return True
            else:
                self._model_train_data[MODEL_TRAIN_DATA.scaler] = None
                print(f"[{scaler_file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load scaler fails: {str(e)}")
        return False

    def _load_parameters(self):
        # 3. 加载参数
        params_file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.parameters.name])
        try:
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.parameters] = json.load(f)
                print(f"✓ 参数已加载")
                return True
            else:
                self._model_train_data[MODEL_TRAIN_DATA.parameters] = None
                print(f"[{params_file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load parameters fails: {str(e)}")
        return False

    def _load_readme(self):
        readme_file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.readme.name])
        try:
            if os.path.exists(readme_file):
                with open(readme_file, 'r', encoding='utf-8') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.readme] = f.read()
                print(f"✓ README已加载")
                return True
            else:
                self._model_train_data[MODEL_TRAIN_DATA.readme] = None
                print(f"[{readme_file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load README fails: {str(e)}")
        return False

    def _load_train_history(self):
        hist_file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.train_history.name])
        try:
            if os.path.exists(hist_file):
                with open(hist_file, 'rb') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.train_history] = pickle.load(f)
                print(f"✓ 训练历史已加载: {hist_file}")
                return True
            else:
                self._model_train_data[MODEL_TRAIN_DATA.train_history] = None
                print(f"[{hist_file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load train history fails: {str(e)}")
        return False

    def _load_performance(self):
        file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.performance.name])
        try:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.performance] = json.load(f)
                print(f"✓ 参数已加载")
                return True
            else:
                self._model_train_data[MODEL_TRAIN_DATA.performance] = None
                print(f"[{file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load performance fails: {str(e)}")
        return False

    def _load_model_summary(self):
        file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE[MODEL_TRAIN_DATA.model_summary.name])
        try:
            if os.path.exists(file):
                with open(file, 'r', encoding='utf-8') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.model_summary] = f.read()
                print(f"✓ README已加载")
                return True
            else:
                self._model_train_data[MODEL_TRAIN_DATA.model_summary] = None
                print(f"[{file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load model summary fails: {str(e)}")
        return False
#endregion load data functions

    def create_readme(self):
        """创建说明文件"""
        params = self._model_train_data[MODEL_TRAIN_DATA.parameters]
        if params is None:
            return ''
        self.readme_content = f"""
# LSTM股票预测模型

## 模型信息
- 保存时间: {params[LTSM_MODEL_PARAM.timestamp.name]}
- 时间步长: {params[LTSM_MODEL_PARAM.lookback.name]}
- 预测天数: {params[LTSM_MODEL_PARAM.future_days.name]}
- 阈值: {params[LTSM_MODEL_PARAM.threshold.name]}
- 特征数量: {params[LTSM_MODEL_PARAM.feature_count.name]}

## 使用方法
```python
from model_loader import ModelLoader
loader = ModelSaverLoader('{self._directory}')
model, scaler, features = loader.load_all()
文件说明
model.h5: Keras模型文件

scaler.pkl: 数据标准化器

features.json: 特征列名

params.json: 训练参数

history.json: 训练历史记录
"""
        return self.readme_content

    def set_model_train_data(self, data_type:MODEL_TRAIN_DATA, data):
        if data_type in MODEL_TRAIN_DATA:
            self._model_train_data[data_type] = data

    def get_model_train_data(self, data_type:MODEL_TRAIN_DATA):
        if data_type in MODEL_TRAIN_DATA:
            return self._model_train_data[data_type]
        else:
            raise ValueError(f"No data type [{data_type}] in MODEL_TRAIN_DATA")

    def save_train_data(self, data_type:MODEL_TRAIN_DATA=None):
        """保存模型及相关组件"""
        import os
        if self._save_model_mode == False:
            raise ValueError("This is not saving model MODE")
        os.makedirs(self._directory, exist_ok=True)
        if data_type:
            if callable(self._model_io_functions[data_type]):
                self._model_io_functions[data_type]()
            else:
                raise ValueError(f"No saving function for [{data_type}]")
        else: # save all      
            for data_type in self._model_io_functions.keys():
                print(f"Saving data {data_type.name} ...")
                if callable(self._model_io_functions[data_type]):
                    self._model_io_functions[data_type]()
                else:
                    raise ValueError(f"No saving function for [{data_type}]")
    
    def load_train_data(self, data_type:MODEL_TRAIN_DATA=None):
        """加载所有组件"""
        from tensorflow import keras
        if self._save_model_mode:
            raise ValueError("This is not a loading model MODE")
        if data_type:
            if callable(self._model_io_functions[data_type]):
                return self._model_io_functions[data_type]()
            else:
                raise ValueError(f"No loading function for [{data_type}]")
        else: ## load all data
            result = dict(zip([m for m in MODEL_TRAIN_DATA], [False]*len(MODEL_TRAIN_DATA)))
            for data_type in self._model_io_functions.keys():
                if data_type == MODEL_TRAIN_DATA.readme:
                    break
                print(f"Loading data {data_type.name} ...")
                if callable(self._model_io_functions[data_type]):
                    result[data_type] = self._model_io_functions[data_type]()
                else:
                    raise ValueError(f"No loading function for [{data_type}]")
            return result

