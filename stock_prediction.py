import subprocess
import sys, os

# os.environ["LANG"] = "C.UTF-8"
# os.environ["LC_ALL"] = "C.UTF-8"
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font
from tkcalendar import DateEntry
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import json
from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from select_stock import LSTM_Select_Stock, FEATURE
from stock import TICKER
from TickerManager import TickerManager

class StockPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction GUI")
        self.root.geometry("1200x800")
        # 初始化股票列表
        self.stocks = []  # 存储股票代码的列表        
        # 创建Notebook（标签页）
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 初始化管理器
        self.manager = TickerManager()
        self.current_model = None
        
        # 创建标签页
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_visualization_tab()
        
        # 加载已保存的模型列表
        self.load_saved_models()
        self.set_default_years()

    def get_stored_stocks(self):
        """获取所有选中的股票"""
        return self.stocks.copy()

    def set_default_years(self):
        """设置默认年份"""
        current_year = datetime.now().year
        last_year = current_year - 1
        
        # 设置训练标签页的默认年份
        if hasattr(self, 'start_year_var'):
            self.start_year_var.set(str(last_year - 3))  # 3年前
        if hasattr(self, 'end_year_var'):
            self.end_year_var.set(str(last_year))  # 去年
        
        # 设置预测标签页的默认年份
        if hasattr(self, 'start_year_pred_var'):
            self.start_year_pred_var.set(str(last_year))
        if hasattr(self, 'end_year_pred_var'):
            self.end_year_pred_var.set(str(current_year))


    def validate_year_input(self, input_text):
        """验证年份输入，只允许输入数字，且长度为4位"""
        if input_text == "":  # 允许清空
            return True
        
        # 检查是否只包含数字
        if not input_text.isdigit():
            return False
        
        # 检查长度是否不超过4位
        if len(input_text) > 4:
            return False
        
        # 检查是否在合理范围内（1000-9999）
        if len(input_text) == 4:
            year = int(input_text)
            if year < 1000 or year > 9999:
                return False
        
        return True

    def validate_year_range(self):
        """验证年份范围是否合理"""
        try:
            start_year = self.start_year_var.get().strip()
            end_year = self.end_year_var.get().strip()
            
            if not start_year or not end_year:
                return False, "请填写开始年份和结束年份"
            
            start_year_int = int(start_year)
            end_year_int = int(end_year)
            
            if start_year_int > end_year_int:
                return False, "开始年份不能晚于结束年份"
            
            current_year = datetime.now().year
            if start_year_int > current_year or end_year_int > current_year:
                return False, "年份不能晚于当前年份"
            
            if start_year_int < 1900 or end_year_int < 1900:
                return False, "年份不能早于1900年"
            
            return True, ""
            
        except ValueError:
            return False, "年份格式不正确"
        except Exception as e:
            return False, f"验证错误: {str(e)}"

    def get_start_date_from_year(self, year_str):
        """从年份字符串获取开始日期（YYYY-01-01）"""
        try:
            year = int(year_str)
            # 验证年份范围
            if year < 1900 or year > datetime.now().year:
                return None
            return f"{year:04d}-01-01"
        except:
            return None

    def get_end_date_from_year(self, year_str):
        """从年份字符串获取结束日期（YYYY-12-31）"""
        try:
            year = int(year_str)
            # 验证年份范围
            if year < 1900 or year > datetime.now().year:
                return None
            return f"{year:04d}-12-31"
        except:
            return None    

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
        self.stock_combo['values'] = ('AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
                                     'NVDA', 'META', 'NFLX', 'INTC', 'AMD',
                                     'BABA', 'JD', 'PDD', 'BIDU', 'NTES')
        combobox_var.trace('w', lambda *_: self.set_reload_data())
        
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
        date_frame = ttk.Frame(left_frame)
        date_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # 开始年份
        start_date_frame = ttk.Frame(date_frame)
        start_date_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        start_label = tk.Label(start_date_frame, text="开始年份:")
        start_label.pack(anchor=tk.W)
        
        # 年份输入框 - 只接受年份输入
        self.start_year_var = tk.StringVar()
        self.start_year_entry = tk.Entry(
            start_date_frame, 
            width=10, 
            textvariable=self.start_year_var,
            validate="key",  # 按键时验证
            validatecommand=(self.root.register(self.validate_year_input), '%P')
        )
        self.start_year_entry.pack(fill=tk.X, pady=(2, 0))
        self.start_year_entry.insert(0, "2020")  # 默认开始年份
        
        # 添加年份输入提示
        year_hint_label = tk.Label(start_date_frame, text="格式: YYYY", 
                                font=("Arial", 9), fg="gray")
        year_hint_label.pack(anchor=tk.W)
        
        # 结束年份
        end_date_frame = ttk.Frame(date_frame)
        end_date_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        end_label = tk.Label(end_date_frame, text="结束年份:")
        end_label.pack(anchor=tk.W)
        
        # 年份输入框 - 只接受年份输入
        self.end_year_var = tk.StringVar()
        self.end_year_entry = tk.Entry(
            end_date_frame, 
            width=10, 
            textvariable=self.end_year_var,
            validate="key",  # 按键时验证
            validatecommand=(self.root.register(self.validate_year_input), '%P')
        )
        self.end_year_entry.pack(fill=tk.X, pady=(2, 0))
        self.end_year_entry.insert(0, "2023")  # 默认结束年份
        
        # 添加年份输入提示
        year_hint_label2 = tk.Label(end_date_frame, text="格式: YYYY", 
                                font=("Arial", 9), fg="gray")
        year_hint_label2.pack(anchor=tk.W)
                        
        # Lookback设置
        lookback_frame = ttk.Frame(left_frame)
        lookback_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 15))
        
        lookback_label = tk.Label(lookback_frame, text="Lookback Days:")
        lookback_label.pack(side=tk.LEFT)
        
        self.lookback_train = tk.Entry(lookback_frame, width=10)
        self.lookback_train.pack(side=tk.LEFT, padx=(5, 0))
        self.lookback_train.insert(0, f"{self.manager.lookback}")
        
        # 右侧面板 - 特征选择
        right_frame = ttk.LabelFrame(self.training_frame, text="Feature Selection", padding=10)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # 创建特征复选框
        self.feature_vars = {}
        for idx, feature in enumerate(FEATURE):
            var = tk.BooleanVar(value=True)
            self.feature_vars[feature] = var
            
            # 获取特征名称
            feature_name = LSTM_Select_Stock.get_feature_name(feature)
            
            cb = tk.Checkbutton(right_frame, text=feature_name, variable=var,
                                command=lambda f=feature, v=var: self.toggle_feature(f, v))
            cb.grid(row=idx, column=0, sticky=tk.W, pady=2)
        
        # 训练按钮
        btn_frame2 = tk.Frame(self.training_frame)
        btn_frame2.grid(row=1, column=0, columnspan=2, pady=20)
        
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

    def set_reload_data(self):
        """设置重新加载数据标志"""
        if self.manager:
            self.manager.reload_data = True

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
        
        # 开始年份
        start_label_pred = tk.Label(params_frame, text="Start Year:")
        start_label_pred.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.start_year_pred_var = tk.StringVar()
        self.start_year_pred_entry = tk.Entry(
            params_frame, 
            width=10, 
            textvariable=self.start_year_pred_var,
            validate="key",
            validatecommand=(self.root.register(self.validate_year_input), '%P')
        )
        self.start_year_pred_entry.grid(row=0, column=1, padx=5, pady=5)
        self.start_year_pred_entry.insert(0, "2023")
        
        # 结束年份
        end_label_pred = tk.Label(params_frame, text="End Year:")
        end_label_pred.grid(row=0, column=2, sticky=tk.W, pady=5, padx=(20,0))
        
        self.end_year_pred_var = tk.StringVar()
        self.end_year_pred_entry = tk.Entry(
            params_frame, 
            width=10, 
            textvariable=self.end_year_pred_var,
            validate="key",
            validatecommand=(self.root.register(self.validate_year_input), '%P')
        )
        self.end_year_pred_entry.grid(row=0, column=3, padx=5, pady=5)
        self.end_year_pred_entry.insert(0, "2024")
            
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
            feature_names.append(LSTM_Select_Stock.get_feature_name(feature))
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
    
    def add_stock(self):
        """添加股票代码"""
        stock = self.stock_combo.get().strip().upper()
        if not stock:
            messagebox.showwarning("Warning", "Please enter a stock code")
            return
        
        # 检查股票代码格式（简单验证）
        if len(stock) < 1 or len(stock) > 5:
            if not messagebox.askyesno("Confirmation", f"Stock '{stock}' length is uncommon, do you want to add it?"):
                return
        
        # 检查是否已存在
        if stock in self.stocks:
            messagebox.showinfo("Hint", f"Stock '{stock}' already exists")
            return
        
        # 添加到内部列表
        self.stocks.append(stock)
        
        # 更新Listbox显示
        self.update_stock_listbox()
        
        # 清空输入框
        self.stock_combo.set("")
        
        # 记录日志
        self.log_message(f"Add Stock: {stock}")
        
        # 焦点回到输入框
        self.stock_combo.focus_set()
    
    def remove_stock(self):
        """从列表中删除选中的股票"""
        # 获取选中的项目
        selection = self.stock_listbox.curselection()
        
        if not selection:
            messagebox.showwarning("Warning", "Please select a stock to delete")
            return
        
        # 获取选中的股票代码
        index = selection[0]
        stock = self.stocks[index]
        
        # 确认删除
        if messagebox.askyesno("Confirmation", f"Are you sure you want to delete stock '{stock}'?"):
            # 从内部列表删除
            self.stocks.pop(index)
            
            # 更新Listbox显示
            self.update_stock_listbox()
            
            # 记录日志
            self.log_message(f"Deleted stock: {stock}")
    
    def clear_all_stocks(self):
        """清空所有股票"""
        if not self.stocks:
            messagebox.showinfo("Info", "List of stocks is already empty")
            return
        
        if messagebox.askyesno("Confirmation", f"Are you sure you want to clear all {len(self.stocks)} stocks?"):
            # 清空内部列表
            self.stocks.clear()
            
            # 更新Listbox显示
            self.update_stock_listbox()
            
            # 记录日志
            self.log_message("Cleared all stocks already in the list")

    def update_stock_listbox(self):
        """更新Listbox显示"""
        # 清空Listbox
        self.stock_listbox.delete(0, tk.END)
        
        # 添加所有股票
        for i, stock in enumerate(self.stocks, 1):
            self.stock_listbox.insert(tk.END, f"{i}. {stock}")
        
        # 更新状态显示
        self.update_status()

    def update_status(self):
        """更新状态信息"""
        # 这里可以添加状态栏更新逻辑
        pass

    def toggle_feature(self, feature, var):
        """切换特征启用状态"""
        if var.get():
            LSTM_Select_Stock.enable_feature(feature)
            self.log_message(f"Enabled feature: {LSTM_Select_Stock.get_feature_name(feature)}")
        else:
            LSTM_Select_Stock.disable_feature(feature)
            self.log_message(f"Disabled feature: {LSTM_Select_Stock.get_feature_name(feature)}")

    def start_training(self):
        """开始训练模型"""
        stocks = list(self.stock_listbox.get(0, tk.END))
        if not stocks:
            messagebox.showwarning("Warning", "Please add stocks first")
            return
        
        # 验证年份范围
        is_valid, error_msg = self.validate_year_range()
        if not is_valid:
            messagebox.showwarning("Warning", error_msg)
            return
        
        try:
            lookback = int(self.lookback_train.get())
            if lookback <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a valid lookback days")
            return
        
        # 在新线程中运行训练，避免GUI冻结
        thread = threading.Thread(target=self.run_training, args=(stocks, lookback))
        thread.daemon = True
        thread.start()    

    def run_training(self, stocks, lookback):
        """运行训练过程"""
        self.log_message(f"Start training: {len(stocks)}")
        self.log_message(f"Stock list: {', '.join(stocks)}")
        
        try:
            # 获取年份并转换为日期
            start_year = self.start_year_var.get().strip()
            end_year = self.end_year_var.get().strip()
            
            start_date = self.get_start_date_from_year(start_year)
            end_date = self.get_end_date_from_year(end_year)
            
            if not start_date or not end_date:
                self.log_message("Warning: Format of years is incorrect or out of range (1900-current year)")
                messagebox.showerror("Warning", "Format of years is incorrect or out of range (1900-current year)")
                return

            self.log_message(f"Training Period: {start_date} to {end_date}")

            # 初始化管理器
            self.manager = TickerManager(start_date, end_date, lookback)
            
            # 添加股票
            for stock in stocks:
                self.manager.add_ticker(stock)
            
            # 加载数据
            self.log_message("Loading stock data...")
            self.manager.load_ticker_data()
            
            # 处理数据并训练
            self.log_message("Processing data and training model...")
            self.manager.process_select_stocks()
            
            # 保存模型信息
            self.save_model_info(stocks, start_date, end_date, lookback)
            
            self.log_message("Training completed!")
            messagebox.showinfo("Success", "Model training completed!")
            
        except Exception as e:
            self.log_message(f"Error during training: {str(e)}")
            messagebox.showerror("Error", f"Error during training: {str(e)}")
    
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
            stocks = self.stock_listbox.get(0, tk.END)
            start_date = self.start_date_train.get_date().strftime("%Y-%m-%d")
            end_date = self.end_date_train.get_date().strftime("%Y-%m-%d")
            lookback = int(self.lookback_train.get())
            
            self.save_model_info(stocks, start_date, end_date, lookback)
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
        if not self.current_model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        # 验证年份范围
        try:
            start_year = self.start_year_pred_var.get().strip()
            end_year = self.end_year_pred_var.get().strip()
            
            if not start_year or not end_year:
                messagebox.showwarning("Warning", "Please enter start year and end year")
                return
            
            start_year_int = int(start_year)
            end_year_int = int(end_year)
            
            if start_year_int > end_year_int:
                messagebox.showwarning("Warning", "Start year cannot be later than end year")
                return
                
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
                                  args=(start_year_int, end_year_int, lookback, threshold))
        thread.daemon = True
        thread.start()
    
    def run_prediction(self, start_year, end_year, lookback, threshold):
        """运行预测过程"""
        try:
            # 将年份转换为日期
            start_date = f"{start_year:04d}-01-01"
            end_date = f"{end_year:04d}-12-31"
            self.log_message(f"Prediction Period: {start_date} to {end_date}", target="result")

            # 这里需要根据你的代码调整预测逻辑
            date_offset = 180  # 假设的偏移天数
            
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
                if LSTM_Select_Stock.get_feature_name(feature) == feature_name:
                    selected_feature = feature
                    break
            
            if not selected_feature:
                messagebox.showwarning("Warning", "Feature does not exist")
                return
            
            # 计算特征值
            selector = LSTM_Select_Stock()
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
