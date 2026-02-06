import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from Common.AutoNumber import AutoIndex
from tkinter import Menu

class StockChartPlotter:
    """
    股票图表绘制类
    功能：
    1. 绘制股价走势图（主图）
    2. 绘制交易量柱状图（副图）
    3. 鼠标悬停显示数据点详细信息
    4. 支持自定义样式和配置
    """
    class CONTEXT_MENU(AutoIndex):
        MA_5 = ()
        MA_20 = ()
        MA_90 = ()
        Seperator = ()
        remove = ()
        remove_all = ()

    def __init__(self, symbol, stock_data, figsize=(14, 10)):
        """
        初始化StockChartPlotter
        
        参数:
        ----------
        stock_data : DataFrame
            股票数据，必须包含以下列：
            - 'Date': 日期 (datetime格式)
            - 'Close': 收盘价
            - 'Volume': 交易量
            可选列:
            - 'Open': 开盘价
            - 'High': 最高价
            - 'Low': 最低价
        
        figsize : tuple
            图表大小，默认(14, 10)
        """
        self.symbol = symbol
        self.stock_data = stock_data.copy()
        self.figsize = figsize
        self.fig = None
        self.ax_price = None  # 股价图坐标轴
        self.ax_volume = None  # 交易量图坐标轴
        self.moving_average_lines = []
        
        # 确保日期为datetime格式
        if not pd.api.types.is_datetime64_any_dtype(self.stock_data['Date']):
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        
        # 将日期转换为matplotlib格式
        self.dates_mpl = mdates.date2num(self.stock_data['Date'])
        
        # 配置样式
        self.setup_styles()
        
        # 初始化交互元素
        self.hover_line = None
        self.price_annotation = None
        self.volume_annotation = None
        self.create_plot()
        self._create_context_menu_commands()
        # We'll create the menu dynamically each time        
        self.fig.canvas.mpl_connect('button_press_event', self.on_right_click)
        
    def setup_styles(self):
        """设置图表样式"""
        self.styles = {
            # 颜色配置
            'colors': {
                'price_line': '#1f77b4',        # 股价线颜色
                'price_up': '#2ecc71',          # 上涨颜色
                'price_down': '#e74c3c',        # 下跌颜色
                'volume_up': 'lightgreen',      # 上涨交易量颜色
                'volume_down': 'lightcoral',    # 下跌交易量颜色
                'hover_line': '#e74c3c',        # 悬停线颜色
                'annotation_bg': 'white',       # 注释框背景
                'grid_color': '#ecf0f1',        # 网格颜色
            },
            # 线型配置
            'linewidths': {
                'price_line': 2,
                'hover_line': 1,
            },
            # 透明度
            'alphas': {
                'volume': 0.7,
                'hover_line': 0.5,
            },
            # 字体大小
            'font_sizes': {
                'title': 16,
                'axis_label': 12,
                'annotation': 10,
                'tick_label': 10,
            }
        }
        
    def calculate_price_change(self):
        """计算价格涨跌，用于确定颜色"""
        if 'Open' in self.stock_data.columns and 'Close' in self.stock_data.columns:
            # 如果有开盘价，用收盘价与开盘价比较
            price_change = self.stock_data['Close'] >= self.stock_data['Open']
        else:
            # 否则用收盘价与前一日收盘价比较
            price_change = self.stock_data['Close'] >= self.stock_data['Close'].shift(1)
            price_change.iloc[0] = True  # 第一天默认为上涨
        
        # 计算涨跌颜色
        price_colors = np.where(price_change, 
                                self.styles['colors']['price_up'], 
                                self.styles['colors']['price_down'])
        volume_colors = np.where(price_change,
                                 self.styles['colors']['volume_up'],
                                 self.styles['colors']['volume_down'])
        
        return price_colors, volume_colors
    
    def create_plot(self):
        """创建主图表"""
        # 创建图形和坐标轴
        self.fig, (self.ax_price, self.ax_volume) = plt.subplots(
            2, 1, 
            figsize=self.figsize,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )
        if self.fig is None:
            raise ValueError("fig is None")
        self.root = self.fig.canvas.manager.window
        # 计算涨跌颜色
        price_colors, volume_colors = self.calculate_price_change()
        
        # 绘制股价图
        self.plot_price_chart(price_colors)
        
        # 绘制交易量图
        self.plot_volume_chart(volume_colors)
        
        # 配置图表格式
        self.format_chart()
        
        # 添加交互功能
        self.add_interactive_features()
        
        # 调整布局
        plt.tight_layout()
        
    def plot_price_chart(self, price_colors):
        """绘制股价图"""
        # 绘制收盘价折线
        self.price_line, = self.ax_price.plot(
            self.dates_mpl, 
            self.stock_data['Close'],
            color=self.styles['colors']['price_line'],
            linewidth=self.styles['linewidths']['price_line'],
            label='Close Price',
            zorder=5
        )
        
        # 如果有高低价数据，绘制价格区间
        if all(col in self.stock_data.columns for col in ['High', 'Low']):
            # 绘制价格区间（阴影）
            self.ax_price.fill_between(
                self.dates_mpl,
                self.stock_data['Low'],
                self.stock_data['High'],
                alpha=0.2,
                color='gray',
                label='Price Range'
            )
        
        # 设置股价图标题和标签
        self.ax_price.set_title(f'{self.symbol}: Stock Price Trend', 
                               fontsize=self.styles['font_sizes']['title'],
                               fontweight='bold',
                               pad=20)
        self.ax_price.set_ylabel('Price (€)', 
                                 fontsize=self.styles['font_sizes']['axis_label'])
        self.ax_price.legend(loc='upper left')
        self.ax_price.grid(True, alpha=0.3, linestyle='--', 
                          color=self.styles['colors']['grid_color'])
        
        # 添加网格
        self.ax_price.grid(True, alpha=0.3, linestyle='--')
        
    def plot_volume_chart(self, volume_colors):
        """绘制交易量图"""
        # 绘制交易量柱状图
        self.ax_volume.bar(
            self.dates_mpl,
            self.stock_data['Volume'],
            color=volume_colors,
            alpha=self.styles['alphas']['volume'],
            width=0.8,  # 柱状图宽度
            edgecolor='black',
            linewidth=0.5
        )
        
        # 设置交易量图标签
        self.ax_volume.set_ylabel('Volume', 
                                  fontsize=self.styles['font_sizes']['axis_label'])
        self.ax_volume.set_xlabel('Date', 
                                  fontsize=self.styles['font_sizes']['axis_label'])
        self.ax_volume.grid(True, alpha=0.3, linestyle='--',
                           color=self.styles['colors']['grid_color'])
        
        # 格式化y轴标签（使用K/M/B表示大数字）
        self.ax_volume.yaxis.set_major_formatter(
            plt.FuncFormatter(self.format_large_numbers)
        )
        
    def format_large_numbers(self, x, pos):
        """格式化大数字显示（如1000显示为1K）"""
        if x >= 1e9:
            return f'{x/1e9:.1f}B'
        elif x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.0f}'
    
    def format_chart(self):
        """格式化图表"""
        # 设置x轴日期格式
        date_formatter = mdates.DateFormatter('%Y-%m-%d')
        
        # 根据数据长度设置合适的刻度间隔
        num_dates = len(self.dates_mpl)
        if num_dates <= 30:
            # 少于30天，每周显示一个刻度
            locator = mdates.WeekdayLocator(interval=1)
        elif num_dates <= 90:
            # 少于90天，每两周显示一个刻度
            locator = mdates.WeekdayLocator(interval=2)
        else:
            # 更多数据，每月显示一个刻度
            locator = mdates.MonthLocator()
        
        # 应用格式到交易量图（股价图共享x轴）
        self.ax_volume.xaxis.set_major_locator(locator)
        self.ax_volume.xaxis.set_major_formatter(date_formatter)
        
        # 自动旋转日期标签
        plt.setp(self.ax_volume.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 设置y轴格式
        self.ax_price.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'¥{x:.2f}')
        )
    
    def add_interactive_features(self):
        """添加交互功能"""
        # 创建悬停线（垂直虚线）
        self.hover_line = self.ax_price.axvline(
            x=self.dates_mpl[0],
            color=self.styles['colors']['hover_line'],
            linewidth=self.styles['linewidths']['hover_line'],
            alpha=self.styles['alphas']['hover_line'],
            visible=False,
            zorder=10
        )
        
        # 创建股价注释框
        self.price_annotation = self.ax_price.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=self.styles['colors']['annotation_bg'],
                edgecolor='black',
                alpha=0.9
            ),
            fontsize=self.styles['font_sizes']['annotation'],
            zorder=11
        )
        self.price_annotation.set_visible(False)
        
        # 创建交易量注释框
        self.volume_annotation = self.ax_volume.annotate(
            "",
            xy=(0, 0),
            xytext=(20, -30),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=self.styles['colors']['annotation_bg'],
                edgecolor='black',
                alpha=0.9
            ),
            fontsize=self.styles['font_sizes']['annotation'],
            zorder=11
        )
        self.volume_annotation.set_visible(False)
        
        # 连接鼠标移动事件
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        
        # 连接鼠标离开事件
        self.fig.canvas.mpl_connect("axes_leave_event", self.on_leave)
    
    def on_hover(self, event):
        """鼠标悬停事件处理"""
        # 检查鼠标是否在图表区域内
        if event.inaxes in [self.ax_price, self.ax_volume]:
            # 找到最近的日期点
            idx = np.abs(self.dates_mpl - event.xdata).argmin()
            
            # 更新悬停线位置
            self.hover_line.set_xdata([self.dates_mpl[idx]])
            self.hover_line.set_visible(True)
            
            # 获取数据
            date_str = mdates.num2date(self.dates_mpl[idx]).strftime('%Y-%m-%d')
            close_price = self.stock_data['Close'].iloc[idx]
            volume = self.stock_data['Volume'].iloc[idx]
            
            # 获取其他可选数据
            extra_info = ""
            if 'Open' in self.stock_data.columns:
                open_price = self.stock_data['Open'].iloc[idx]
                extra_info += f"Open: €{open_price:.2f}\n"
            if 'High' in self.stock_data.columns:
                high_price = self.stock_data['High'].iloc[idx]
                extra_info += f"High: €{high_price:.2f}\n"
            if 'Low' in self.stock_data.columns:
                low_price = self.stock_data['Low'].iloc[idx]
                extra_info += f"Low: €{low_price:.2f}\n"
            
            # 计算涨跌幅
            if idx > 0:
                prev_close = self.stock_data['Close'].iloc[idx-1]
                change = close_price - prev_close
                change_pct = (change / prev_close) * 100
                change_symbol = "+" if change >= 0 else ""
                change_str = f"Up/Down: {change_symbol}{change:.2f} ({change_symbol}{change_pct:.2f}%)"
            else:
                change_str = "Up/Down: N/A"
            
            # 更新股价注释框
            price_text = f"Date: {date_str}\n" \
                        f"Close: €{close_price:.2f}\n" \
                        f"{change_str}"
            
            if extra_info:
                price_text += f"\n{extra_info}"
            
            self.price_annotation.set_text(price_text)
            self.price_annotation.xy = (self.dates_mpl[idx], close_price)
            self.price_annotation.set_visible(True)
            
            # 更新交易量注释框
            volume_text = f"Date: {date_str}\n" \
                         f"Volume: {volume:,.0f}"
            self.volume_annotation.set_text(volume_text)
            self.volume_annotation.xy = (self.dates_mpl[idx], volume)
            self.volume_annotation.set_visible(True)
            
            # 重绘图形
            self.fig.canvas.draw_idle()
    
    def on_leave(self, event):
        """鼠标离开图表区域事件处理"""
        # 隐藏注释框和悬停线
        self.hover_line.set_visible(False)
        self.price_annotation.set_visible(False)
        self.volume_annotation.set_visible(False)
        self.fig.canvas.draw_idle()
    
    def show(self):
        """显示图表"""
        if self.fig is None:
            self.create_plot()
        plt.show()
    
    def save(self, filename, dpi=300):
        """保存图表为文件
        
        参数:
        ----------
        filename : str
            保存的文件名，支持格式: .png, .jpg, .pdf, .svg
        dpi : int
            图像分辨率，默认300
        """
        if self.fig is None:
            self.create_plot()
        
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
    
    def add_moving_average(self, window=20, color='orange', label=None):
        """添加移动平均线
        
        参数:
        ----------
        window : int
            移动平均窗口大小
        color : str
            线条颜色
        label : str
            图例标签，如果为None则使用'MA{window}'
        """
        if self.ax_price is None:
            raise ValueError("请先调用create_plot()或show()方法")
        
        # 计算移动平均
        ma_label = label if label else f'MA{window}'
        ma_data = self.stock_data['Close'].rolling(window=window).mean()
        
        # 绘制移动平均线
        line = self.ax_price.plot(
            self.dates_mpl,
            ma_data,
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=ma_label
        )
        self.moving_average_lines.append(line)
        # 更新图例
        self.ax_price.legend(loc='upper left')
        
        if self.fig is not None:
            self.fig.canvas.draw_idle()
    
    def set_custom_colors(self, price_up=None, price_down=None, 
                          volume_up=None, volume_down=None):
        """设置自定义颜色
        
        参数:
        ----------
        price_up : str
            股价上涨颜色
        price_down : str
            股价下跌颜色
        volume_up : str
            上涨交易量颜色
        volume_down : str
            下跌交易量颜色
        """
        if price_up:
            self.styles['colors']['price_up'] = price_up
        if price_down:
            self.styles['colors']['price_down'] = price_down
        if volume_up:
            self.styles['colors']['volume_up'] = volume_up
        if volume_down:
            self.styles['colors']['volume_down'] = volume_down

    def _create_context_menu_commands(self):
        self.context_menu = Menu(self.root, tearoff=0)
        for e in StockChartPlotter.CONTEXT_MENU:
            if e.name.startswith('Seperator'):
                self.context_menu.add_separator()
            else:
                self.context_menu.add_command(label=' '.join(e.name.split('_')), command=self.dummy_command)
                self.menu_items[e.name] = self.context_menu.index("end")

    def dummy_command(self):
        pass

    def remove(self, all=False):
        line = self.moving_average_lines.pop()
        line.remove()
        self.fig.canvas.draw()


    def remove_all(self):
        self.remove(True)

    def on_right_click(self, event):
        if event.button == 3:  # Right-click in axes
            # self.last_click_coords = (event.xdata, event.ydata)
            # self.fig._last_right_click = (event.xdata, event.ydata, event.inaxes)
            for key in StockChartPlotter.CONTEXT_MENU:
                # trick here: set key as a default argument to prevent of late binding of variables in lambdas inside a loop!!!
                if key.name.startswith('MA'):
                    x = key.name.split('_')[1]
                    self.context_menu.entryconfig(key.value, command=lambda item=x: self.add_moving_average(item))
                else:
                    self.context_menu.entryconfig(key.value, command=lambda: getattr(self, key.name)())
            try:
                x_tk = self.root.winfo_pointerx()
                y_tk = self.root.winfo_pointery()
                self.context_menu.tk_popup(x_tk, y_tk)
            except:
                # Fallback
                self.context_menu.post(self.last_click_coords)

# ==================== 使用示例 ====================

def create_sample_stock_data(days=100, start_price=100):
    """创建示例股票数据"""
    dates = pd.date_range('2024-01-01', periods=days, freq='D')
    
    # 生成随机价格数据（模拟股价走势）
    returns = np.random.randn(days) * 0.02  # 日收益率
    prices = start_price * (1 + returns).cumprod()
    
    # 生成高低价（基于收盘价）
    highs = prices * (1 + np.random.rand(days) * 0.03)
    lows = prices * (1 - np.random.rand(days) * 0.03)
    opens = prices * (1 + np.random.randn(days) * 0.01)
    
    # 生成交易量数据
    base_volume = np.random.randint(1000000, 5000000, days)
    # 交易量与价格波动相关
    volume = base_volume * (1 + np.abs(returns) * 10)
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volume
    })
    
    return data


def load_stock_data_from_csv(filename):
    """从CSV文件加载股票数据"""
    data = pd.read_csv(filename)
    
    # 确保列名正确（不区分大小写）
    column_mapping = {}
    for col in data.columns:
        col_lower = col.lower()
        if 'date' in col_lower:
            column_mapping[col] = 'Date'
        elif 'close' in col_lower:
            column_mapping[col] = 'Close'
        elif 'volume' in col_lower:
            column_mapping[col] = 'Volume'
        elif 'open' in col_lower:
            column_mapping[col] = 'Open'
        elif 'high' in col_lower:
            column_mapping[col] = 'High'
        elif 'low' in col_lower:
            column_mapping[col] = 'Low'
    
    data = data.rename(columns=column_mapping)
    
    # 确保Date列是datetime格式
    data['Date'] = pd.to_datetime(data['Date'])
    
    return data


if __name__ == "__main__":
    print("股票图表可视化类 - 使用示例")
    print("=" * 50)
    
    # 示例1: 使用示例数据
    print("示例1: 使用随机生成的示例数据")
    sample_data = create_sample_stock_data(days=60)
    
    # 创建图表对象
    plotter = StockChartPlotter(sample_data, figsize=(12, 8))
    
    # 添加移动平均线
    plotter.add_moving_average(window=5, color='red', label='5日均线')
    plotter.add_moving_average(window=20, color='blue', label='20日均线')
    
    # 显示图表
    plotter.show()
    
    # 示例2: 从CSV文件加载数据
    # print("\n示例2: 从CSV文件加载数据")
    # csv_data = load_stock_data_from_csv('your_stock_data.csv')
    # plotter2 = StockChartPlotter(csv_data)
    # plotter2.set_custom_colors(price_up='green', price_down='red')
    # plotter2.show()
    
    # 保存图表
    # plotter.save('stock_chart.png', dpi=300)