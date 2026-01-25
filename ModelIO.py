import pickle
import json, os
import keras
from datetime import datetime
from stockDefine import MODEL_TRAIN_DATA

class ModelSaverLoader:
    FILE_NAME_DEFINE = {'model': '_model.h5', 
                        'scaler': 'scaler.pkl', 
                        'features': 'features.json', 
                        'params': 'params.json',
                        'history': 'history.json',
                        'performance': 'performance.txt',
                        'readme': 'README.md'}
    def __init__(self, directory, ticker_symbol, save=True):
        self._ticker_symbol = ticker_symbol
        self._save_model = save
        self._directory = directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._model_train_data = dict(zip([t for t in MODEL_TRAIN_DATA], [None]*len(MODEL_TRAIN_DATA)))
        self._model_io_functions = dict(zip([t for t in MODEL_TRAIN_DATA], [None]*len(MODEL_TRAIN_DATA)))
        # initialize functions
        for data_type in MODEL_TRAIN_DATA:
            if self._save_model:
                self._model_io_functions[MODEL_TRAIN_DATA.model] = getattr(self, f'_save_{data_type.name}')
            else:
                self._model_io_functions[MODEL_TRAIN_DATA.model] = getattr(self, f'_load_{data_type.name}')

    
    def set_model_train_data(self, data_type:MODEL_TRAIN_DATA, data):
        if data_type in MODEL_TRAIN_DATA:
            self._model_train_data[data_type] = data

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
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"✓ 训练历史已保存至: {history_path}")

    def _save_performance(self):
        if self._model_train_data[MODEL_TRAIN_DATA.performance]:
            perf_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['performance'])
            performance = self._model_train_data[MODEL_TRAIN_DATA.performance]
            with open(perf_path, 'w') as f:
                f.write(performance)
            print(f"✓ 评估表现已保存至: {perf_path}")

    # 6. 创建README文件
    def _save_readme(self):
        if self._model_train_data[MODEL_TRAIN_DATA.readme]:
            readme_path = os.path.join(self._directory, ModelSaverLoader.FILE_NAME_DEFINE['readme'])
            readme = self._model_train_data[MODEL_TRAIN_DATA.readme]
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme)
            print(f"✓ README已保存至: {readme_path}")

    def save_all(self):
        """保存模型及相关组件"""
        import os
        if self._save_model == False:
            raise ValueError("This is not saving model MODE")
        os.makedirs(self._directory, exist_ok=True)
                
        for type, func in self._model_io_functions.items():
            print(f"Saving data {type.name} ...")
            func()
    
    ## load model data from disk
    def _load_model(self):
        # 1. 加载模型
        # parse ticker name
        base_name = os.path.basename(self._directory)
        parts = base_name.split('_')
        self._ticker_symbol = parts[0]
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


    def load_all(self):
        """加载所有组件"""
        from tensorflow import keras
        if self._save_model:
            raise ValueError("This is not a loading model MODE")
        for type, func in self._model_io_functions.items():
            if type == MODEL_TRAIN_DATA.readme:
                break
            print(f"Saving data {type.name} ...")
            func()
