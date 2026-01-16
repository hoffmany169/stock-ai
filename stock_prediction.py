import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from matplotlib import font_manager
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import json
import os
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
        self.root.title("股票预测系统")

        # 设置字体
        self.setup_fonts()
        self.root.geometry("1200x800")
        
        # 创建Notebook（标签页）
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 初始化管理器
        self.manager = None
        self.current_model = None
        
        # 创建标签页
        self.create_training_tab()
        self.create_prediction_tab()
        self.create_visualization_tab()
        
        # 加载已保存的模型列表
        self.load_saved_models()

    def setup_fonts(self):
        """设置中文字体"""
        try:
            # 尝试设置Tkinter字体
            font_names = [
                "WenQuanYi Zen Hei MONO",  # 文泉驿正黑
                "WenQuanYi Micro Hei MONO",  # 文泉驿微米黑
                "Noto Sans Mono CJK SC",  # Google Noto字体
                "DejaVu Sans",  # 备选英文字体
                "Ubuntu"  # Ubuntu系统字体
            ]
            
            # 为Tkinter设置默认字体
            for font_name in font_names:
                try:
                    default_font = tk.font.nametofont("TkDefaultFont")
                    default_font.configure(family=font_name, size=10)
                    print(f"Tkinter font set to: {font_name}")
                    break
                except:
                    continue
            
            # 为ttk设置样式
            style = ttk.Style()
            style.configure('.', font=(font_names[0], 10))
            
            # 设置Matplotlib字体
            matplotlib_fonts = [
                "WenQuanYi Zen Hei",
                "WenQuanYi Micro Hei Mono",
                "Noto Sans CJK SC",
                "DejaVu Sans",
                "Arial"
            ]
            
            for font_name in matplotlib_fonts:
                if font_name in [f.name for f in font_manager.fontManager.ttflist]:
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"Matplotlib font set to: {font_name}")
                    break
            
        except Exception as e:
            print(f"Font setup error: {e}")
            # 如果中文字体设置失败，使用默认英文字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
                
    def create_training_tab(self):
        """创建训练标签页"""
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="模型训练")
        
        # 左侧面板 - 股票管理
        left_frame = ttk.LabelFrame(self.training_frame, text="股票管理", padding=10)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # 股票输入
        ttk.Label(left_frame, text="股票代码:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.stock_combo = ttk.Combobox(left_frame, width=20)
        self.stock_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # 股票列表
        ttk.Label(left_frame, text="股票列表:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.stock_listbox = tk.Listbox(left_frame, height=10, width=25)
        self.stock_listbox.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # 按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="添加", command=self.add_stock).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="删除", command=self.remove_stock).pack(side=tk.LEFT, padx=5)
        
        # 日期选择
        ttk.Label(left_frame, text="开始日期:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.start_date_train = DateEntry(left_frame, width=18, background='darkblue',
                                         foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.start_date_train.grid(row=4, column=1, padx=5, pady=5)
        
        ttk.Label(left_frame, text="结束日期:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.end_date_train = DateEntry(left_frame, width=18, background='darkblue',
                                       foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.end_date_train.grid(row=5, column=1, padx=5, pady=5)
        
        # Lookback设置
        ttk.Label(left_frame, text="Lookback天数:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.lookback_train = ttk.Entry(left_frame, width=10)
        self.lookback_train.grid(row=6, column=1, padx=5, pady=5)
        self.lookback_train.insert(0, "60")
        
        # 右侧面板 - 特征选择
        right_frame = ttk.LabelFrame(self.training_frame, text="特征选择", padding=10)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # 创建特征复选框
        self.feature_vars = {}
        for idx, feature in enumerate(FEATURE):
            var = tk.BooleanVar(value=True)
            self.feature_vars[feature] = var
            
            # 获取特征名称
            feature_name = LSTM_Select_Stock.get_feature_name(feature)
            
            cb = ttk.Checkbutton(right_frame, text=feature_name, variable=var,
                                command=lambda f=feature, v=var: self.toggle_feature(f, v))
            cb.grid(row=idx, column=0, sticky=tk.W, pady=2)
        
        # 训练按钮
        btn_frame2 = ttk.Frame(self.training_frame)
        btn_frame2.grid(row=1, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame2, text="开始训练", command=self.start_training).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame2, text="评估模型", command=self.evaluate_model).pack(side=tk.LEFT, padx=10)
        
        # 日志显示
        ttk.Label(self.training_frame, text="训练日志:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.log_text = scrolledtext.ScrolledText(self.training_frame, height=10, width=100)
        self.log_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # 配置网格权重
        self.training_frame.grid_columnconfigure(0, weight=1)
        self.training_frame.grid_columnconfigure(1, weight=1)
        self.training_frame.grid_rowconfigure(3, weight=1)
    
    def create_prediction_tab(self):
        """创建预测标签页"""
        self.prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text="股票预测")
        
        # 模型选择
        ttk.Label(self.prediction_frame, text="选择训练好的模型:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_combo = ttk.Combobox(self.prediction_frame, width=30)
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(self.prediction_frame, text="加载模型", command=self.load_model).grid(row=0, column=2, padx=5, pady=5)
        
        # 预测参数
        params_frame = ttk.LabelFrame(self.prediction_frame, text="预测参数", padding=10)
        params_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        
        ttk.Label(params_frame, text="开始日期:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.start_date_pred = DateEntry(params_frame, width=18, background='darkblue',
                                        foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.start_date_pred.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="结束日期:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(20,0))
        self.end_date_pred = DateEntry(params_frame, width=18, background='darkblue',
                                      foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.end_date_pred.grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Lookback天数:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.lookback_pred = ttk.Entry(params_frame, width=10)
        self.lookback_pred.grid(row=1, column=1, padx=5, pady=5)
        self.lookback_pred.insert(0, "60")
        
        ttk.Label(params_frame, text="预测阈值:").grid(row=1, column=2, sticky=tk.W, pady=5, padx=(20,0))
        self.pred_threshold = ttk.Entry(params_frame, width=10)
        self.pred_threshold.grid(row=1, column=3, padx=5, pady=5)
        self.pred_threshold.insert(0, "0.7")
        
        # 按钮
        btn_frame = ttk.Frame(self.prediction_frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        ttk.Button(btn_frame, text="开始预测", command=self.start_prediction).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="显示结果", command=self.show_prediction_results).pack(side=tk.LEFT, padx=10)
        
        # 预测结果显示
        ttk.Label(self.prediction_frame, text="预测结果:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.result_text = scrolledtext.ScrolledText(self.prediction_frame, height=15, width=100)
        self.result_text.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        
        # 配置网格权重
        self.prediction_frame.grid_columnconfigure(0, weight=1)
        self.prediction_frame.grid_rowconfigure(4, weight=1)
    
    def create_visualization_tab(self):
        """创建可视化标签页"""
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text="数据可视化")
        
        # 控制面板
        control_frame = ttk.Frame(self.visualization_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="显示原始数据", command=self.show_raw_data).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="选择特征:").pack(side=tk.LEFT, padx=(20,5))
        
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
        
        ttk.Button(control_frame, text="显示特征曲线", command=self.show_feature_curve).pack(side=tk.LEFT, padx=5)
        
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
            messagebox.showwarning("警告", "请输入股票代码")
            return
        
        if stock in self.stock_listbox.get(0, tk.END):
            messagebox.showinfo("提示", f"{stock} 已存在")
            return
        
        self.stock_listbox.insert(tk.END, stock)
        self.stock_combo.set("")
        self.log_message(f"添加股票: {stock}")
    
    def remove_stock(self):
        """删除选中的股票代码"""
        selection = self.stock_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请选择要删除的股票")
            return
        
        stock = self.stock_listbox.get(selection[0])
        self.stock_listbox.delete(selection[0])
        self.log_message(f"删除股票: {stock}")
    
    def toggle_feature(self, feature, var):
        """切换特征启用状态"""
        if var.get():
            LSTM_Select_Stock.enable_feature(feature)
            self.log_message(f"启用特征: {LSTM_Select_Stock.get_feature_name(feature)}")
        else:
            LSTM_Select_Stock.disable_feature(feature)
            self.log_message(f"禁用特征: {LSTM_Select_Stock.get_feature_name(feature)}")
    
    def start_training(self):
        """开始训练模型"""
        stocks = self.stock_listbox.get(0, tk.END)
        if not stocks:
            messagebox.showwarning("警告", "请先添加股票")
            return
        
        try:
            lookback = int(self.lookback_train.get())
            if lookback <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("警告", "请输入有效的Lookback天数")
            return
        
        # 在新线程中运行训练，避免GUI冻结
        thread = threading.Thread(target=self.run_training, args=(stocks, lookback))
        thread.daemon = True
        thread.start()
    
    def run_training(self, stocks, lookback):
        """运行训练过程"""
        self.log_message("开始训练模型...")
        
        try:
            start_date = self.start_date_train.get_date().strftime("%Y-%m-%d")
            end_date = self.end_date_train.get_date().strftime("%Y-%m-%d")
            
            # 初始化管理器
            self.manager = TickerManager(start_date, end_date, lookback)
            
            # 添加股票
            for stock in stocks:
                self.manager.add_ticker(stock)
            
            # 加载数据
            self.log_message("加载股票数据...")
            self.manager.load_ticker_data()
            
            # 处理数据并训练
            self.log_message("处理数据并训练模型...")
            self.manager.process_select_stocks()
            
            # 保存模型信息
            self.save_model_info(stocks, start_date, end_date, lookback)
            
            self.log_message("训练完成！")
            messagebox.showinfo("成功", "模型训练完成！")
            
        except Exception as e:
            self.log_message(f"训练出错: {str(e)}")
            messagebox.showerror("错误", f"训练过程中出错: {str(e)}")
    
    def evaluate_model(self):
        """评估模型"""
        if not self.manager:
            messagebox.showwarning("警告", "请先训练模型")
            return
        
        self.log_message("开始评估模型...")
        
        try:
            # 这里需要根据你的代码调整评估逻辑
            # 假设TickerManager有评估方法
            self.log_message("评估完成！")
            
        except Exception as e:
            self.log_message(f"评估出错: {str(e)}")
    
    def load_model(self):
        """加载已保存的模型"""
        model_name = self.model_combo.get()
        if not model_name:
            messagebox.showwarning("警告", "请选择要加载的模型")
            return
        
        # 这里需要实现具体的模型加载逻辑
        self.log_message(f"加载模型: {model_name}")
    
    def start_prediction(self):
        """开始预测"""
        if not self.current_model:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        try:
            lookback = int(self.lookback_pred.get())
            threshold = float(self.pred_threshold.get())
            
            if lookback <= 0 or not (0 <= threshold <= 1):
                raise ValueError
        except ValueError:
            messagebox.showwarning("警告", "请输入有效的参数")
            return
        
        self.log_message("开始预测...", target="result")
        
        # 在新线程中运行预测
        thread = threading.Thread(target=self.run_prediction, args=(lookback, threshold))
        thread.daemon = True
        thread.start()
    
    def run_prediction(self, lookback, threshold):
        """运行预测过程"""
        try:
            start_date = self.start_date_pred.get_date().strftime("%Y-%m-%d")
            end_date = self.end_date_pred.get_date().strftime("%Y-%m-%d")
            
            # 这里需要根据你的代码调整预测逻辑
            date_offset = 180  # 假设的偏移天数
            
            # 假设TickerManager有预测方法
            if hasattr(self.manager, 'select_stocks'):
                self.manager.select_stocks(date_offset, lookback, threshold)
                selected_stocks = self.manager.get_selected_stocks()
                
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "预测结果:\n")
                self.result_text.insert(tk.END, "="*50 + "\n")
                
                for stock in selected_stocks:
                    self.result_text.insert(tk.END, f"推荐股票: {stock}\n")
                
                if not selected_stocks:
                    self.result_text.insert(tk.END, "没有符合条件的股票\n")
            
            self.log_message("预测完成！", target="result")
            
        except Exception as e:
            self.log_message(f"预测出错: {str(e)}", target="result")
    
    def show_prediction_results(self):
        """显示预测结果"""
        # 结果已经在result_text中显示了
        pass
    
    def show_raw_data(self):
        """显示原始数据"""
        if not self.manager:
            messagebox.showwarning("警告", "请先训练模型")
            return
        
        try:
            # 获取第一个股票的数据
            stocks = self.manager.get_all_tickers()
            if not stocks:
                self.log_message("没有可用的股票数据")
                return
            
            ticker = stocks[0]
            data = self.manager.tickers[ticker][TICKER.DATA]
            
            # 绘制价格曲线
            self.ax.clear()
            self.ax.plot(data.index, data['Close'], label='收盘价', linewidth=2)
            self.ax.set_title(f"{ticker} 股价走势")
            self.ax.set_xlabel("日期")
            self.ax.set_ylabel("价格")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("错误", f"显示数据出错: {str(e)}")
    
    def show_feature_curve(self):
        """显示特征曲线"""
        if not self.manager:
            messagebox.showwarning("警告", "请先训练模型")
            return
        
        feature_name = self.feature_combo.get()
        if not feature_name:
            messagebox.showwarning("警告", "请选择特征")
            return
        
        try:
            # 获取第一个股票的数据
            stocks = self.manager.get_all_tickers()
            if not stocks:
                self.log_message("没有可用的股票数据")
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
                messagebox.showwarning("警告", "特征不存在")
                return
            
            # 计算特征值
            selector = LSTM_Select_Stock()
            selector.ticker = self.manager.tickers[ticker]
            selector.preprocess_data()
            
            # 绘制特征曲线
            self.ax.clear()
            
            if hasattr(data, selected_feature):
                self.ax.plot(data.index, data[selected_feature], label=feature_name, linewidth=2)
                self.ax.set_title(f"{ticker} {feature_name} 曲线")
                self.ax.set_xlabel("日期")
                self.ax.set_ylabel("特征值")
                self.ax.legend()
                self.ax.grid(True, alpha=0.3)
            else:
                self.ax.text(0.5, 0.5, "特征数据不可用", 
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=self.ax.transAxes)
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("错误", f"显示特征曲线出错: {str(e)}")
    
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
        
        # 更新模型列表
        self.load_saved_models()
    
    def load_saved_models(self):
        """加载已保存的模型列表"""
        if not os.path.exists("models"):
            return
        
        model_files = [f for f in os.listdir("models") if f.endswith(".json")]
        model_names = [f.replace(".json", "") for f in model_files]
        
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
    app = StockPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()