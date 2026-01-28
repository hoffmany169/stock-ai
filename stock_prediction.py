
# os.environ["LANG"] = "C.UTF-8"
# os.environ["LC_ALL"] = "C.UTF-8"
import sys, os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import json
from datetime import datetime
from ModelTrainLSTM import LSTMModelTrain, FEATURE
from StockDefine import TICKER, FEATURE, StockFeature
from TickerManager import TickerManager
from DateRangePicker import DateRangePicker
from Common.AutoNumber import AutoIndex

class ConfigEntry(AutoIndex):
    model_save_path = ()
    ticker_list = ()

class StockPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction GUI")
        self.root.geometry("1200x800")
        self.Gui_Config_Data = {ConfigEntry.model_save_path.name: 'models',
                                ConfigEntry.ticker_list.name:['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
                                                'NVDA', 'META', 'NFLX', 'INTC', 'AMD',
                                                'BABA', 'JD', 'PDD', 'BIDU', 'NTES'],}
        self._gui_config_file_name = 'gui.cfg' # json file
        # 初始化股票列表
        self._processing_stocks = []  # 存储股票代码的列表        
        # 创建Notebook（标签页）
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 初始化管理器
        self.manager = TickerManager()
        self.current_model = None
        self._reload_data = True # decide if reloading data
        self._cur_config = self.Gui_Config_Data
        self.load_gui_config()
        # 特征
        self._stock_features = StockFeature()

        # 创建标签页
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_visualization_tab()
        
        # 加载已保存的模型列表
        self.load_saved_models()

#region properties
    @property
    def reload_data(self):
        return self._reload_data

    @property
    def processing_stocks(self):
        """获取所有选中的股票"""
        return self._processing_stocks.copy()
#endregion properties

    def load_gui_config(self):
        if os.path.exists(self._gui_config_file_name):
            with open(self._gui_config_file_name, 'r') as cfg:
                self._cur_config = json.load(cfg)
        else: # create config
            self.save_gui_config()

    def save_gui_config(self):
        with open(self._gui_config_file_name, 'w+') as cfg:
            json.dump(self._cur_config, cfg)

    def create_training_tab(self):
        """创建训练标签页"""
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Model Training")
        
        # 左侧面板 - 股票管理
        left_frame = ttk.LabelFrame(self.training_frame, text="Stock Management", padding=10)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # 第一行：股票输入和按钮
        input_frame = ttk.Frame(left_frame)
        input_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))        
        # 股票输入标签
        stock_label = tk.Label(input_frame, text="Stock Symbol list:")
        stock_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # 股票代码输入框
        combobox_var = tk.StringVar()
        self.stock_combo = ttk.Combobox(input_frame, width=15, textvariable=combobox_var)
        self.stock_combo.pack(side=tk.LEFT, padx=(0, 5))
        # 设置一些常见的股票代码作为提示
        self.stock_combo['values'] = self.Gui_Config_Data['ticker_list']
        combobox_var.trace('w', lambda *_: self._reload_data)
        
        # 添加按钮
        add_btn = tk.Button(input_frame, text="Add", 
                        command=self.add_stock, width=6)
        add_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # 删除按钮
        remove_btn = tk.Button(input_frame, text="Remove",
                            command=self.remove_stock, width=6)
        remove_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # 清空按钮
        clear_btn = tk.Button(input_frame, text="Clear All",
                            command=self.clear_all_stocks, width=6)
        clear_btn.pack(side=tk.LEFT)

        # 股票列表和滚动条框架
        list_frame = ttk.Frame(left_frame)
        list_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        
        # 股票列表框
        self.stock_listbox = tk.Listbox(list_frame, height=10, width=25,
                                    selectmode=tk.EXTENDED)  # 允许多选
        self.stock_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 滚动条
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.stock_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stock_listbox.config(yscrollcommand=scrollbar.set)

        # 绑定键盘事件
        self.stock_combo.bind('<Return>', lambda event: self.add_stock())  # 按回车添加
        self.stock_listbox.bind('<Delete>', lambda event: self.remove_stock())  # 按Delete删除

        # 日期选择框架
        self.date_frame = ttk.Frame(left_frame)
        self.date_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.train_date_picker = DateRangePicker(self.date_frame)
        self.train_date_picker.pack(anchor=tk.W)
        # Lookback设置
        lookback_frame = ttk.Frame(left_frame)
        lookback_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        
        lookback_label = tk.Label(lookback_frame, text="Lookback Days:")
        lookback_label.pack(side=tk.LEFT)
        
        self.lookback_train = tk.Entry(lookback_frame, width=10)
        self.lookback_train.pack(side=tk.LEFT, padx=(5, 0))
        self.lookback_train.insert(0, "60")

        # path of saving models
        # model_path_label = tk.Label(lookback_frame, text="Saving Moodels:")
        # model_path_label.pack(side=tk.LEFT)
        # self.model_path = tk.Entry(lookback_frame, width=30)
        # self.model_path.pack(side=tk.LEFT, padx=(5, 0))
        # self.model_path.insert(0, "models")
        # self.path_button = tk.Button(lookback_frame, text='...',
        #                              command=self.select_saving_path)
        # self.path_button.pack(side=tk.LEFT, padx=2)
        
        # 右侧面板 - 特征选择
        right_frame = ttk.LabelFrame(self.training_frame, text="Feature Selection", padding=10)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # 创建特征复选框
        self.feature_vars = {}
        for idx, feature in enumerate(FEATURE):
            var = tk.BooleanVar(value=True)
            self.feature_vars[feature] = var
            
            # 获取特征名称
            feature_name = self._stock_features.get_feature_name(feature)
            
            cb = tk.Checkbutton(right_frame, text=feature_name, variable=var,
                                command=lambda f=feature, v=var: self.toggle_feature(f, v))
            cb.grid(row=idx, column=0, sticky=tk.W, pady=2)
        
        btn_frame2 = tk.Frame(self.training_frame)
        btn_frame2.grid(row=1, column=0, columnspan=2, pady=20)
        # 加载数据
        from Common.DropdownButton import DropdownButton
        self.load_btn = DropdownButton(btn_frame2, 
                                   ['Load from Markt', 'Load from Disk'], 
                                    command=self.start_loading)
        self.load_btn.pack(side=tk.LEFT, padx=10)
        
        # 训练按钮
        train_btn = tk.Button(btn_frame2, text="Start Training", 
                            command=self.start_training)
        train_btn.pack(side=tk.LEFT, padx=10)
        
        eval_btn = tk.Button(btn_frame2, text="Evaluate Model",
                            command=self.evaluate_model)
        eval_btn.pack(side=tk.LEFT, padx=10)
        
        # 日志显示
        log_frame = ttk.Frame(self.training_frame)
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        
        log_label = tk.Label(log_frame, text="Training Log:")
        log_label.pack(anchor=tk.W, padx=5, pady=(0, 5))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=100)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 配置网格权重
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(2, weight=1)  # 股票列表框行可扩展
        self.training_frame.grid_columnconfigure(0, weight=1)
        self.training_frame.grid_columnconfigure(1, weight=1)
        self.training_frame.grid_rowconfigure(0, weight=1)
        self.training_frame.grid_rowconfigure(2, weight=1)  # 日志区域可扩展

    def create_prediction_tab(self):
        """创建预测标签页"""
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="Stock Prediction")
        
        # 模型选择
        ttk.Label(self.prediction_frame, text="Select Trained Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_combo = ttk.Combobox(self.prediction_frame, width=30)
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(self.prediction_frame, text="Load Model", command=self.load_model).grid(row=0, column=2, padx=5, pady=5)
        
        # 预测参数
        params_frame = ttk.LabelFrame(self.prediction_frame, text="Prediction Parameters", padding=10)
        params_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        
        self.predict_date_picker = DateRangePicker(params_frame)
        self.predict_date_picker.grid(row=0, column=0, sticky=tk.W, pady=5)
            
        # Lookback天数
        lookback_label_pred = tk.Label(params_frame, text="Lookback Days:")
        lookback_label_pred.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.lookback_pred = tk.Entry(params_frame, width=10)
        self.lookback_pred.grid(row=1, column=1, padx=5, pady=5)
        self.lookback_pred.insert(0, "60")
        
        # 预测阈值
        threshold_label = tk.Label(params_frame, text="Prediction Threshold:")
        threshold_label.grid(row=1, column=2, sticky=tk.W, pady=5, padx=(20,0))
        
        self.pred_threshold = tk.Entry(params_frame, width=10)
        self.pred_threshold.grid(row=1, column=3, padx=5, pady=5)
        self.pred_threshold.insert(0, "0.7")
        
        # 年份提示标签
        hint_label = tk.Label(params_frame, text="Format: YYYY (e.g. 2023)", 
                            font=("Arial", 9), fg="gray")
        hint_label.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        # 按钮
        btn_frame = ttk.Frame(self.prediction_frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        ttk.Button(btn_frame, text="Start Prediction", command=self.start_prediction).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Show Results", command=self.show_prediction_results).pack(side=tk.LEFT, padx=10)
        
        # 预测结果显示
        ttk.Label(self.prediction_frame, text="Prediction Results:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.result_text = scrolledtext.ScrolledText(self.prediction_frame, height=15, width=100)
        self.result_text.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        
        # 配置网格权重
        self.prediction_frame.grid_columnconfigure(0, weight=1)
        self.prediction_frame.grid_rowconfigure(4, weight=1)
    
    def create_visualization_tab(self):
        """创建可视化标签页"""
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text="Data Visualization")
        
        # 控制面板
        control_frame = ttk.Frame(self.visualization_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Show Raw Data", command=self.show_raw_data).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Select Feature:").pack(side=tk.LEFT, padx=(20,5))
        
        # 特征选择下拉框
        self.feature_combo = ttk.Combobox(control_frame, width=25)
        self.feature_combo.pack(side=tk.LEFT, padx=5)
        
        # 填充特征列表
        feature_names = []
        for feature in FEATURE:
            feature_names.append(self._stock_features.get_feature_name(feature))
        self.feature_combo['values'] = feature_names
        if feature_names:
            self.feature_combo.current(0)
        
        ttk.Button(control_frame, text="Show Feature Curve", command=self.show_feature_curve).pack(side=tk.LEFT, padx=5)
        
        # 图表显示区域
        self.figure_frame = ttk.Frame(self.visualization_frame)
        self.figure_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def validate_stock_symbol(self, symbol):
        """验证股票代码格式（增强版）"""
        if not symbol or len(symbol) == 0:
            return False
        
        # 基本验证：只包含字母、数字和点号，长度1-10个字符
        import re
        if not re.match(r'^[A-Z0-9.]{1,10}$', symbol):
            return False
        
        # 常见股票代码后缀验证（可选）
        # 例如：AAPL, GOOGL, BRK.B, BTC-USD 等格式
        
        return True

    def validate_and_fix_stock_symbol(self, symbol):
        """验证并修复股票代码格式"""
        symbol = symbol.strip().upper()
        
        # 常见修复：添加交易所后缀
        if '.' not in symbol and '-' not in symbol:
            # 对于没有后缀的股票，通常需要添加交易所信息
            # 但yfinance通常可以处理大部分常见股票代码
            pass
        
        return symbol

    def open_select_directory(self):
        chosen_path = filedialog.askdirectory(initialdir='.', title='Choose Directory for Saving Model Data:')
        print("Open ", chosen_path)
        self._cur_config[ConfigEntry.model_save_path.name] = chosen_path

    def start_loading(self, selected:str):
        if selected.endswith('Markt'):
            # loading data from yfinance
            self.start_loading_data()
        elif selected.endswith('Disk'):
            # loading data from disk
            self.open_select_directory()
            self.manager.process_load_train_data(self._cur_config[ConfigEntry.model_save_path])

    def add_stock(self):
        """添加股票代码"""
        stock = self.stock_combo.get().strip().upper()
        if not stock:
            messagebox.showwarning("Warning", "Please enter a stock code")
            return
        
        # 验证股票代码格式
        if not self.validate_stock_symbol(stock):
            messagebox.showwarning("Warning", f"Format of Stock is wrong: {stock}\nThey must be capital and number, for example: AAPL, GOOGL")
            return
        
        # 检查是否已存在
        if stock in self._processing_stocks:
            messagebox.showinfo("Hint", f"{stock} exists already!")
            return
        
        # 可选：快速验证股票代码是否存在
        if not self.quick_check_stock_exists(stock):
            response = messagebox.askyesno("Confirmation", 
                f"Stock Symbol {stock} may not exist or invalid\nAdd it?")
            if not response:
                return
        
        # 添加到内部列表
        self._processing_stocks.append(stock)
        
        # 更新Listbox显示
        self.update_stock_listbox()
        
        # 清空输入框
        self.stock_combo.set("")
        
        # 记录日志
        self.log_message(f"Add Stock: [{stock}]")
        
        # 焦点回到输入框
        self.stock_combo.focus_set()
        # 添加到下拉框的历史记录
        # self.add_to_combo_history(stock)
        self._reload_data = True

    def quick_check_stock_exists(self, symbol):
        """快速检查股票代码是否存在"""
        try:
            # 使用简单的API调用检查
            import yfinance as yf
            stock = yf.Ticker(symbol)
            # 尝试获取基本信息
            info = stock.info
            # 如果有基本的公司信息，则认为有效
            if info and len(info) > 0:
                company_name = info.get('longName', info.get('shortName', symbol))
                self.log_message(f"{symbol}: {company_name}")
                return True
            return False
        except Exception as e:
            # 如果出现404错误，说明股票代码无效
            if "404" in str(e):
                return False
            # 其他错误可能只是网络问题，暂时返回True让用户决定
            return True    

    def remove_stock(self):
        """从列表中删除选中的股票"""
        # 获取选中的项目
        selection = self.stock_listbox.curselection()
        
        if not selection:
            messagebox.showwarning("Warning", "Please select a stock to delete")
            return
        
        # 获取选中的股票代码
        index = selection[0]
        stock = self._processing_stocks[index]
        
        # 确认删除
        if messagebox.askyesno("Confirmation", f"Are you sure you want to delete stock '{stock}'?"):
            # 从内部列表删除
            self._processing_stocks.pop(index)
            
            # 更新Listbox显示
            self.update_stock_listbox()
            
            # 记录日志
            self.log_message(f"Deleted stock: {stock}")
            self._reload_data = True
    
    def clear_all_stocks(self):
        """清空所有股票"""
        if not self._processing_stocks:
            messagebox.showinfo("Info", "List of stocks is already empty")
            return
        
        if messagebox.askyesno("Confirmation", f"Are you sure you want to clear all {len(self._processing_stocks)} stocks?"):
            # 清空内部列表
            self._processing_stocks.clear()
            
            # 更新Listbox显示
            self.update_stock_listbox()
            
            # 记录日志
            self.log_message("Cleared all stocks already in the list")
            self._reload_data = True

    def update_stock_listbox(self):
        """更新Listbox显示"""
        # 清空Listbox
        self.stock_listbox.delete(0, tk.END)
        
        # 添加所有股票
        for i, stock in enumerate(self._processing_stocks, 1):
            self.stock_listbox.insert(tk.END, f"{i}. [{stock}]")
        
        # 更新状态显示
        self.update_status()

    def update_status(self):
        """更新状态信息"""
        # 这里可以添加状态栏更新逻辑
        pass

    def toggle_feature(self, feature, var):
        """切换特征启用状态"""
        if var.get():
            self._stock_features.enable_feature(feature)
            self.log_message(f"Enabled feature: {self._stock_features.get_feature_name(feature)}")
        else:
            self._stock_features.disable_feature(feature)
            self.log_message(f"Disabled feature: {self._stock_features.get_feature_name(feature)}")
        self._reload_data = True

    def start_loading_data(self):
        # 在新线程中运行训练，避免GUI冻结
    #     thread = threading.Thread(target=self.run_loading, args=(self._processing_stocks))
    #     thread.daemon = True
    #     thread.start()    

    # def run_loading(self, stocks):
        self.log_message(f"Start training: {len(self._processing_stocks)}")
        self.log_message(f"Stock list: {', '.join(self._processing_stocks)}")
        try:
            # 获取年份并转换为日期
            start_date, end_date = self.train_date_picker.get()
            
            self.log_message(f"Training Period: {start_date} to {end_date}")

            # 初始化管理器
            self.manager.start_date = start_date
            self.manager.end_date = end_date
            self.manager.stock_features = self._stock_features
            
            # 添加股票
            for stock in self._processing_stocks:
                self.manager.add_ticker(stock)
            
            # 加载数据
            self.log_message("Loading stock data...")
            # 在单独的线程中运行数据加载
            def load_data_thread(stocks):
                from ModelIO import ModelSaverLoader
                from StockDefine import MODEL_TRAIN_DATA
                try:
                    # 直接调用修复后的方法
                    no_data = self.manager.load_ticker_data()
                    
                    # 检查数据是否加载成功
                    for ticker in stocks:
                        if ticker in self.manager.tickers:
                            # create folder structure for saving data and save loaded data
                            save_path = self._cur_config[ConfigEntry.model_save_path.name]
                            mio = ModelSaverLoader(save_path, ticker)
                            mio.set_model_train_data(MODEL_TRAIN_DATA.stock_data)
                            mio.save_train_data(MODEL_TRAIN_DATA.stock_data)
                            # keep directory into stock model
                            self.manager.get_stock_model(ticker).ticker_directory = mio.directory
                    
                except Exception as e:
                    error_msg = f"Failure to load data: {str(e)}"
                    self.root.after(0, self.handle_loading_error, error_msg)            
                return no_data
            # 启动数据加载线程
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                # Schedule the function to run
                future = executor.submit(load_data_thread, self._processing_stocks)
                # result() blocks until the thread finishes and returns the value
                result = future.result()
                if len(result) > 0:
                    print(f'Failure downloaded stocks: {result}') 
        except Exception as e:
            self.log_message(f"Error during training: {str(e)}")
            messagebox.showerror("Error", f"Error during training: {str(e)}")

    def handle_loading_error(self, error_msg):
        """处理训练错误"""
        self.log_message(error_msg)
        messagebox.showerror("Error", error_msg)    


    def start_training(self):
        """开始训练模型"""
        if not self._processing_stocks:
            messagebox.showwarning("Warning", "Please add stocks first")
            return
                
        try:
            lookback = int(self.lookback_train.get())
            if lookback <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a valid lookback days")
            return
        
        # 在新线程中运行训练，避免GUI冻结
        thread = threading.Thread(target=self.run_training, args=(self._processing_stocks, lookback))
        thread.daemon = True
        thread.start()    

    def run_training(self, stocks, lookback):
        """运行训练过程"""
        self.log_message(f"Start training: {len(stocks)}")
        self.log_message(f"Stock list: {', '.join(stocks)}")
        
        try:
            # 获取年份并转换为日期
            start_date, end_date = self.train_date_picker.get()
            
            self.log_message(f"Training Period: {start_date} to {end_date}")

            # 初始化管理器
            self.manager.start_date = start_date
            self.manager.end_date = end_date
            self.manager.stock_features = self._stock_features
            
            # 添加股票
            for stock in stocks:
                self.manager.add_ticker(stock)
            
            # 加载数据
            self.log_message("Loading stock data...")
            # 在单独的线程中运行数据加载
            def load_data_thread():
                from ModelIO import ModelSaverLoader
                from StockDefine import MODEL_TRAIN_DATA
                try:
                    # 直接调用修复后的方法
                    no_data = self.manager.load_ticker_data()
                    
                    # 检查数据是否加载成功
                    for ticker in stocks:
                        if ticker in self.manager.tickers:
                            # create folder structure for saving data and save loaded data
                            save_path = self._cur_config[ConfigEntry.model_save_path.name]
                            mio = ModelSaverLoader(save_path, ticker)
                            mio.set_model_train_data(MODEL_TRAIN_DATA.stock_data)
                            mio.save_train_data(MODEL_TRAIN_DATA.stock_data)
                            # keep directory into stock model
                            self.manager.get_stock_model(ticker).ticker_directory = mio.directory

                    # 数据加载成功后继续处理
                    self.root.after(0, self.continue_training, stocks, lookback, start_date, end_date)
                    
                except Exception as e:
                    error_msg = f"Failure to load data: {str(e)}"
                    self.root.after(0, self.handle_training_error, error_msg)            
                return no_data
            
            # 启动数据加载线程
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                # Schedule the function to run
                future = executor.submit(load_data_thread)
                # result() blocks until the thread finishes and returns the value
                result = future.result()
                if len(result) > 0:
                    print(f'Failure downloaded stocks: {result}') 
        except Exception as e:
            self.log_message(f"Error during training: {str(e)}")
            messagebox.showerror("Error", f"Error during training: {str(e)}")

    def continue_training(self, stocks, lookback, start_date, end_date):
            # 处理数据并训练
        try:
            self.log_message("Processing data and training model...")
            self.manager.process_select_stocks()
            
            # 保存模型信息
            # self.save_model_info(stocks, start_date, end_date, lookback)
            
            self.log_message("Training completed!")
            messagebox.showinfo("Success", "Model training completed!")
        except Exception as e:
            self.log_message(f"Error during training: {str(e)}")
            messagebox.showerror("Error", f"Error during training: {str(e)}")
            
    def handle_training_error(self, error_msg):
        """处理训练错误"""
        self.log_message(error_msg)
        messagebox.showerror("Error", error_msg)    

    def evaluate_model(self):
        """评估模型"""
        if not self.manager:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        self.log_message("Start evaluation...")
        
        try:
            # 这里需要根据你的代码调整评估逻辑
            # 假设TickerManager有评估方法
            # 在新线程中运行训练，避免GUI冻结
            def start_eval_func(ticker):
                self.manager.stock_selector.evaluate_model(model=self.manager.tickers[ticker][TICKER.MODEL])
            thread = threading.Thread(target=lambda: [start_eval_func(ticker) for ticker in self.manager.get_all_tickers()])
            thread.daemon = True
            thread.start()
            self.log_message("Evaluation completed!")
            
        except Exception as e:
            self.log_message(f"Error in evaluation: {str(e)}")

    def save_model(self):
        """保存当前模型"""
        if not self.manager:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        try:
            # stocks = self.stock_listbox.get(0, tk.END)
            start_date = self.start_date_train.get_date().strftime("%Y-%m-%d")
            end_date = self.end_date_train.get_date().strftime("%Y-%m-%d")
            lookback = int(self.lookback_train.get())
            
            # self.save_model_info(stocks, start_date, end_date, lookback)
            for ticker in self._processing_stocks:
                self.manager.selector.save_model(ticker, start_date, end_date, lookback)
            messagebox.showinfo("Success", "Model saved successfully!")
            
        except Exception as e:
            self.log_message(f"Error saving model: {str(e)}")
            messagebox.showerror("Error", f"Error saving model: {str(e)}")  

    def load_model(self):
        """加载已保存的模型"""
        model_name = self.model_combo.get()
        if not model_name:
            messagebox.showwarning("Warning", "Please select a model to load")
            return
        
        # 这里需要实现具体的模型加载逻辑
        self.log_message(f"Loading model: {model_name}")
    
    def start_prediction(self):
        """开始预测"""
        try:                            
            lookback = int(self.lookback_pred.get())
            threshold = float(self.pred_threshold.get())
            
            if lookback <= 0 or not (0 <= threshold <= 1):
                raise ValueError
        except ValueError:
            messagebox.showwarning("Warning", "Please enter valid parameters")
            return
            
        self.log_message("Start prediction...", target="result")
        
        # 在新线程中运行预测
        thread = threading.Thread(target=self.run_prediction, 
                                  args=(lookback, threshold))
        thread.daemon = True
        thread.start()
    
    def run_prediction(self, lookback, threshold):
        """运行预测过程"""
        try:
            start_date, end_date = self.predict_date_picker.get()
            self.log_message(f"Prediction Period: {start_date} to {end_date}", target="result")
            
            # 假设TickerManager有预测方法
            if hasattr(self.manager, 'select_stocks'):
                self.manager.select_stocks(date_offset, lookback, threshold)
                selected_stocks = self.manager.get_selected_stocks()
                
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Prediction Period: {start_year} to {end_year}\n")
                self.result_text.insert(tk.END, "Prediction Result:\n")
                self.result_text.insert(tk.END, "="*50 + "\n")
                
                for stock in selected_stocks:
                    self.result_text.insert(tk.END, f"Suggested Stock: {stock}\n")
                
                if not selected_stocks:
                    self.result_text.insert(tk.END, "No stocks meet the criteria\n")
            
            self.log_message("Prediction completed!", target="result")
            
        except Exception as e:
            self.log_message(f"Error in prediction: {str(e)}", target="result")
    
    def show_prediction_results(self):
        """显示预测结果"""
        # 结果已经在result_text中显示了
        pass
    
    def show_raw_data(self):
        """显示原始数据"""
        if not self.manager:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        try:
            # 获取第一个股票的数据
            stocks = self.manager.get_all_tickers()
            if not stocks:
                self.log_message("No available stock data")
                return
            
            ticker = stocks[0]
            data = self.manager.tickers[ticker][TICKER.DATA]
            
            # 绘制价格曲线
            self.ax.clear()
            self.ax.plot(data.index, data['Close'], label='Close Price', linewidth=2)
            self.ax.set_title(f"{ticker} Stock Trends")
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Price")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying data: {str(e)}")
    
    def show_feature_curve(self):
        """显示特征曲线"""
        if not self.manager:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        feature_name = self.feature_combo.get()
        if not feature_name:
            messagebox.showwarning("Warning", "Please select a feature")
            return
        
        try:
            # 获取第一个股票的数据
            stocks = self.manager.get_all_tickers()
            if not stocks:
                self.log_message("No available stock data")
                return
            
            ticker = stocks[0]
            data = self.manager.tickers[ticker][TICKER.DATA]
            
            # 查找对应的特征
            selected_feature = None
            for feature in FEATURE:
                if self._stock_features.get_feature_name(feature) == feature_name:
                    selected_feature = feature
                    break
            
            if not selected_feature:
                messagebox.showwarning("Warning", "Feature does not exist")
                return
            
            # 计算特征值
            selector = LSTMModelTrain()
            selector.ticker = self.manager.tickers[ticker]
            selector.preprocess_data()
            
            # 绘制特征曲线
            self.ax.clear()
            
            if hasattr(data, selected_feature):
                self.ax.plot(data.index, data[selected_feature], label=feature_name, linewidth=2)
                self.ax.set_title(f"{ticker} {feature_name} Curve")
                self.ax.set_xlabel("Date")
                self.ax.set_ylabel("Feature Value")
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
            else:
                self.ax.text(0.5, 0.5, "Feature data unavailable", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=self.ax.transAxes)
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying feature curve: {str(e)}")
    
    def save_model_info(self, stocks, start_date, end_date, lookback):
        """保存模型信息"""
        model_info = {
            "stocks": list(stocks),
            "start_date": start_date,
            "end_date": end_date,
            "lookback": lookback,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 保存到文件
        if not os.path.exists("models"):
            os.makedirs("models")
        
        with open(f"models/{model_name}.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # 保存每个股票的模型文件，为每个股票建立单独的目录
        for ticker in stocks:
            model = self.manager.tickers[ticker][TICKER.MODEL]
            model_dir = f"models/{ticker}"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # 保存特征列表
            features = self.manager.tickers[ticker][TICKER.FEATURES]
            features_file = f"{model_dir}/{model_name}_features.joblib"
            json.dump(features, features_file)

            # 保存缩放器文件
            scaler = self.manager.tickers[ticker][TICKER.SCALER]
            scaler_file = f"{model_dir}/{model_name}_scaler.joblib"
            json.dump(scaler, scaler_file)

            # 保存模型文件
            model.save(f"{model_dir}/{model_name}.h5")

        # 更新模型列表
        self.load_saved_models()
    
    def load_saved_models(self):
        """加载已保存的模型列表"""
        if not os.path.exists("models"):
            return
        
        model_files = [f for f in os.listdir("models") if f.endswith(".json")]
        model_names = [f.replace(".json", "") for f in model_files]
        # list in prediction tab
        self.model_combo['values'] = model_names
    
    def log_message(self, message, target="training"):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        if target == "training":
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
        elif target == "result":
            self.result_text.insert(tk.END, log_entry)
            self.result_text.see(tk.END)

def main():

    root = tk.Tk()
    # root.tk.call("encoding", "system", "utf-8")
    app = StockPredictionGUI(root)
    root.mainloop()
        
if __name__ == "__main__":
    main()
