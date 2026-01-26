"""
Docstring for ModelIO
handle with saving or loading model training parameters and model data.
for example:

sm = StockModel('NVDA') # stock model
sm.start_date = '2023-1-1'
sm.end_date = '2023-12-31'
sm.load_historical_data()

sf = StockFeature() # features
ss = LSTMSelectStock(sm, sf.get_features()) #select stock
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
import json, os
import keras
from datetime import datetime
from stockDefine import MODEL_TRAIN_DATA, LTSM_MODEL_PARAM

class ModelSaverLoader:
    FILE_NAME_DEFINE = {'stock_data': 'stock_data.csv',
                        'model': '_model.h5', 
                        'scaler': 'scaler.pkl', 
                        'features': 'features.json', 
                        'params': 'params.json',
                        'history': 'history.pkl',
                        'performance': 'performance.json',
                        'readme': 'README.md', 
                        'model_summary': 'model_summary.txt'}
    def __init__(self, directory, ticker_symbol=None, save=True):
        """
        Docstring for __init__
        
        :param self: Description
        :param directory: super directory, in which all data will be saved
        :param ticker_symbol: str
        :param save: save mode or load mode
        """
        self._save_model_mode = save
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
            if ticker_symbol is None:
                raise ValueError("Ticker symbol is None")
            self._ticker_symbol = ticker_symbol
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._directory = os.path.join(directory, f'{ticker_symbol}_{self.timestamp}')
        else:
            if ticker_symbol is None:
                # parse ticker name
                base_name = os.path.basename(self._directory)
                parts = base_name.split('_')
                self._ticker_symbol = parts[0]
            else:
                self._ticker_symbol = ticker_symbol
        self._init_model_functions()

    def _init_model_functions(self):
        # initialize functions
        for data_type in MODEL_TRAIN_DATA:
            if self._save_model_mode:
                print(f'save function name: _save_{data_type.name}')
                self._model_io_functions[data_type] = getattr(self, f'_save_{data_type.name}')
            else:
                print(f'load function name: _load_{data_type.name}')
                self._model_io_functions[data_type] = getattr(self, f'_load_{data_type.name}')

    def _save_stock_data(self):
        if self._model_train_data[MODEL_TRAIN_DATA.stock_data]:
            file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['stock_data'])
            data = self._model_train_data[MODEL_TRAIN_DATA.stock_data]
            data.to_csv(data, compression='zip')
            print(f"✓ 原始数据已保存至: {file}")

    def _save_model(self):
        # 1. 保存Keras模型
        model_name = f"{self._ticker_symbol}{ModelSaverLoader.FILE_NAME_DEFINE['model']}"
        model_save_path = os.path.join(self._directory, model_name)
        model = self._model_train_data[MODEL_TRAIN_DATA.model]
        model.save(model_save_path)
        print(f"✓ 模型已保存至: {model_save_path}")

    def _save_scaler(self):
        # 2. 保存Scaler
        if self._model_train_data[MODEL_TRAIN_DATA.scaler]:
            scaler_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['scaler'])
            scaler = self._model_train_data[MODEL_TRAIN_DATA.scaler]
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"✓ Scaler已保存至: {scaler_path}")

    def _save_parameters(self):
        # 4. 保存参数
        params_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['params'])
        params = self._model_train_data[MODEL_TRAIN_DATA.parameters]
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✓ 参数已保存至: {params_path}")

    def _save_train_history(self):
        # 5. 保存训练历史（如果有）
        # if hasattr(self.model, 'history') and self.model.history:
        if self._model_train_data[MODEL_TRAIN_DATA.train_history]:
            history_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['history'])
            history = self._model_train_data[MODEL_TRAIN_DATA.train_history]
            with open(history_path, 'wb') as f:
                pickle.dump(history, f)
            print(f"✓ 训练历史已完整保存至: {history_path}")

    def _save_performance(self):
        if self._model_train_data[MODEL_TRAIN_DATA.performance]:
            perf_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['performance'])
            performance = self._model_train_data[MODEL_TRAIN_DATA.performance]
            with open(perf_path, 'w') as f:
                json.dump(performance, f, indent=2)
            print(f"✓ 评估表现已保存至: {perf_path}")

    # 6. 创建README文件
    def _save_readme(self):
        if self._model_train_data[MODEL_TRAIN_DATA.readme]:
            readme_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['readme'])
            readme = self._model_train_data[MODEL_TRAIN_DATA.readme]
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme)
            print(f"✓ README已保存至: {readme_path}")

    def _save_model_summary(self):
        if self._model_train_data[MODEL_TRAIN_DATA.model_summary]:
            summary_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['model_summary'])
            summary = self._model_train_data[MODEL_TRAIN_DATA.model_summary]
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"✓ 模型总览已保存至: {summary_path}")

    ## load model data from disk
    def _load_stock_data(self):
        pass

    def _load_model(self):
        # 1. 加载模型
        model_file_name = f"{self._ticker_symbol}{ModelSaverLoader.FILE_NAME_DEFINE['model']}"
        model_file = os.path.join(self._directory, model_file_name)
        try:
            if os.path.exists(model_file):
                self._model_train_data[MODEL_TRAIN_DATA.model] = keras.models.load_model(model_file)
                print(f"✓ 模型已加载: {model_file}")
            else:
                raise ValueError(f"[{model_file}] doesn't exist !")
        except Exception as e:
            print(f"Load model fails: {str(e)}")

    def _load_scaler(self):
        # 2. 加载Scaler
        scaler_file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['scaler'])
        try:
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.scaler] = pickle.load(f)
                print(f"✓ Scaler已加载")
            else:
                raise ValueError(f"[{scaler_file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load scaler fails: {str(e)}")

    def _load_parameters(self):
        # 3. 加载参数
        params_file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['params'])
        try:
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.parameters] = json.load(f)
                print(f"✓ 参数已加载")
            else:
                raise ValueError(f"[{params_file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load scaler fails: {str(e)}")

    def _load_readme(self):
        readme_file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['readme'])
        try:
            if os.path.exists(readme_file):
                with open(readme_file, 'r', encoding='utf-8') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.parameters] = f.read()
                print(f"✓ README已加载")
            else:
                raise ValueError(f"[{readme_file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load scaler fails: {str(e)}")

    def _load_train_history(self):
        hist_file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['history'])
        try:
            if os.path.exists(hist_file):
                with open(hist_file, 'rb') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.train_history] = pickle.load(f)
                print(f"✓ 训练历史已加载: {hist_file}")
            else:
                raise ValueError(f"[{hist_file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load scaler fails: {str(e)}")

    def _load_performance(self):
        file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['performance'])
        try:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.performance] = json.load(f)
                print(f"✓ 参数已加载")
            else:
                raise ValueError(f"[{file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load scaler fails: {str(e)}")

    def _load_model_summary(self):
        file = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['model_summary'])
        try:
            if os.path.exists(file):
                with open(file, 'r', encoding='utf-8') as f:
                    self._model_train_data[MODEL_TRAIN_DATA.parameters] = f.read()
                print(f"✓ README已加载")
            else:
                raise ValueError(f"[{file}] doesn't exist !")
        except Exception as e:            
            raise(f"Load scaler fails: {str(e)}")

    def create_readme(self):
        """创建说明文件"""
        params = self._model_train_data[MODEL_TRAIN_DATA.parameters]
        if params is None:
            return ''
        readme_content = f"""
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
        return readme_content

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
            self._model_io_functions[data_type]()
        else: # save all      
            for type, func in self._model_io_functions.items():
                print(f"Saving data {type.name} ...")
                func()
    
    def load_train_data(self, data_type:MODEL_TRAIN_DATA=None):
        """加载所有组件"""
        from tensorflow import keras
        if self._save_model_mode:
            raise ValueError("This is not a loading model MODE")
        if data_type:
            self._model_io_functions[data_type]()
        else: ## load all data
            for type, func in self._model_io_functions.items():
                if type == MODEL_TRAIN_DATA.readme:
                    break
                print(f"Saving data {type.name} ...")
                func()

