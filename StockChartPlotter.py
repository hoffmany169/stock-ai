import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from Common.AutoNumber import AutoIndex
from tkinter import Menu
from plot_style import PlotStyle, PLOT_ELEMENT, STYLE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class StockVisualData:
    class TYPE(AutoIndex):
        axes = ()
        artist = ()
        property = ()
    def __init__(self, fig=None):
        self._fig = fig
        self.visual_data = {}
        # self.visual_data.artist = {}
        # self.visual_data.property = {}

    @property
    def fig(self):
        return self._fig
    @fig.setter
    def fig(self, value):
        self._fig = value

    def add_stock_visual_data(self, data_type:TYPE, data:any, axes_name=None, name=None):
        """添加股票数据到图表
        
        参数:
        ----------
        data_type : StockVisualData
            数据类型，包含以下属性:
            - axes: 图表坐标轴对象
            - artist: 图表元素对象（如线条、柱状图等）
            - property: 数据属性（如价格、交易量等）
        data : any
            数据内容，具体格式取决于数据类型
        """
        # 根据数据类型添加数据到图表
        if data_type == StockVisualData.TYPE.axes:
            if (type(data) == list and type(name) != list):
                raise ValueError("如果data是列表，name必须也是列表，且长度与data相同")
            if type(data) == list and name is not None and  len(data) != len(name):
                raise ValueError("data和name列表长度不一致")
            if type(data) == list:
                for i, ax in enumerate(data):
                    if name is None:
                        name = f"axes_{len(self.visual_data.get('axes', []))}"
                    self.visual_data[name[i]] = ax
                    self.visual_data[name[i]].artists = {}
                    self.visual_data[name[i]].properties = {}
            else:
                if name is None:
                    name = f"axes_{len(self.visual_data.get('axes', []))}"
                self.visual_data[name] = data
                self.visual_data[name].artists = {}
                self.visual_data[name].properties = {}
        else:
            if axes_name is None:
                raise ValueError("axes_name must be provided for artist data")
            if self.visual_data.get(axes_name) is None:
                raise ValueError(f"axes_name '{axes_name}' does not exist in visual_data")
            if data_type == StockVisualData.TYPE.artist:
                if name is None:
                    name = f"artist_{len(self.visual_data[axes_name].artists)}"
                # if name already exists, overwrite it
                self.visual_data[axes_name].artists[name] = data
            if data_type == StockVisualData.TYPE.property:
                if name is None:
                    name = f"property_{len(self.visual_data[axes_name].properties)}"
                self.visual_data[axes_name].properties[name] = data
            else:
                # 其他类型的数据处理（如价格、交易量等）
                pass
        
        # 重绘图表
        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def get_stock_visual_data(self, data_type:TYPE, axes_name, name=None):
        """获取图表中的股票数据
        
        参数:
        ----------
        data_type : StockVisualData
            数据类型，包含以下属性:
            - axes: 图表坐标轴对象
            - artist: 图表元素对象（如线条、柱状图等）
            - property: 数据属性（如价格、交易量等
        axes_name : str
            坐标轴名称，用于定位数据所在的坐标轴
        name : str
            数据名称，用于定位具体的数据项
        返回:
        any
            返回指定的数据项内容
        """
        if self.visual_data.visual_data.get(axes_name) is None:
            raise ValueError(f"axes_name '{axes_name}' does not exist in visual_data")
        if data_type == StockVisualData.TYPE.property:
            return self.visual_data.visual_data[axes_name].properties.get(name)
        elif data_type == StockVisualData.TYPE.artist:
            return self.visual_data.visual_data[axes_name].artists.get(name)
        elif data_type == StockVisualData.TYPE.axes:
            return self.visual_data.visual_data.get(axes_name)

    def popup_stock_visual_data(self, data_type:TYPE, axes_name, name=None):
        """从图表中移除指定的股票数据
        
        参数:
        ----------
        data_type : StockVisualData
            数据类型，包含以下属性:
            - axes: 图表坐标轴对象
            - artist: 图表元素对象（如线条、柱状图等）
            - property: 数据属性（如价格、交易量等）
        axes_name : str
            坐标轴名称，用于定位数据所在的坐标轴
        name : str
            数据名称，用于定位具体的数据项
        """
        if self.visual_data.visual_data.get(axes_name) is None:
            raise ValueError(f"axes_name '{axes_name}' does not exist in visual_data")
        if data_type == StockVisualData.TYPE.property:
            if name in self.visual_data.visual_data[axes_name].properties:
                return self.visual_data.visual_data[axes_name].properties.pop(name)
        elif data_type == StockVisualData.TYPE.artist:
            if name in self.visual_data.visual_data[axes_name].artists:
                return self.visual_data.visual_data[axes_name].artists.pop(name)    
        elif data_type == StockVisualData.TYPE.axes:
            if axes_name in self.visual_data.visual_data:
                return self.visual_data.visual_data.pop(axes_name)
        return None    

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
        ma_5 = ()
        ma_20 = ()
        ma_90 = ()
        seperator = ()
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

        图形存储数据结构:
        self.fig: matplotlib.figure.Figure对象，主图表对象
        self.visual_data: 存储图表配置和元素的对象，包含以下属性:
            - axes: list [{ax1 : [artist1, {property1 : data}, ...]}, 
                          {ax2 : [artist2, {property1 : data}, ...]}
                          ...
                         ]
        """
        self.symbol = symbol
        self.stock_data = stock_data.copy()
        self.figsize = figsize
        self.visual_data = StockVisualData()
        self.plot_styles = PlotStyle()
        
        # 确保日期为datetime格式
        if not pd.api.types.is_datetime64_any_dtype(self.stock_data['Date']):
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        
        # 将日期转换为matplotlib格式
        self.dates_mpl = mdates.date2num(self.stock_data['Date'])
        
        # 配置样式
        # self.setup_styles()
        
        # 初始化交互元素
        self.hover_line = None
        self.price_annotation = None
        self.volume_annotation = None
        # valid fig and ax are available after plot is created.
        self.create_plot()

    # after plot is created, create window controls, e.g. context menu
    def set_backend_window(self, parent):
        self.parent = parent
        # add plot canvas of figure to tkinter window
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        # get root of window, which is top level window of canvas, also sub-window of parent tkinter window 
        self.tk_root = self.canvas.get_tk_widget() # root of this figure canvas
        self._create_context_menu_commands()
        # We'll create the menu dynamically each time. add event to figure's canvas        
        self.fig_canvas.mpl_connect('button_press_event', self.on_right_click)
        return (self.fig, self.root) # root is canvas of parent figure

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
        colors = np.where(price_change, 
                                self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.price_up), 
                                self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.price_down))        
        return colors
    
    # plot stock chart
    def create_plot(self):
        """创建主图表"""
        # 创建图形和坐标轴
        fig, (ax_price, ax_volume) = plt.subplots(
            2, 1, 
            figsize=self.figsize,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )
        if fig is None:
            raise ValueError("fig is None")
        self.visual_data.fig = fig
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.axes, [ax_price, ax_volume])
        # 计算涨跌颜色
        colors = self.calculate_price_change()
        self.visaul_config.add_visual_data(StockVisualData.TYPE.property, colors, axes_name='axes_0', name='price_change')
        self.visaul_config.add_visual_data(StockVisualData.TYPE.property, colors, axes_name='axes_1', name='price_change')
        
        # 绘制股价图
        self.plot_price_chart(ax_price)
        
        # 绘制交易量图
        self.plot_volume_chart(ax_volume, self.visaul_config.volume_colors)
        
        # 配置图表格式
        self.format_chart_price(ax_price)
        self.format_chart_bar(ax_volume)
        
        # 添加交互功能
        self.add_interactive_features(ax_price, ax_price=True)
        self.add_interactive_features(ax_volume, ax_price=False)
        self.visaul_config.axes = [ax_price, ax_volume]
        
        # 调整布局
        plt.tight_layout()
        
    def plot_price_chart(self, ax_price):
        """绘制股价图"""
        # 绘制收盘价折线
        price_line, = ax_price.plot(
            self.dates_mpl, 
            self.stock_data['Close'],
            color=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.price_line),
            linewidth=self.plot_styles.get_setting(STYLE.line_widths, PLOT_ELEMENT.price_line),
            label='Close Price',
            zorder=5
        )
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artist, price_line, name='price_line', axes_name='axes_0')
        # 如果有高低价数据，绘制价格区间
        if all(col in self.stock_data.columns for col in ['High', 'Low']):
            # 绘制价格区间（阴影）
            ax_price.fill_between(
                self.dates_mpl,
                self.stock_data['Low'],
                self.stock_data['High'],
                alpha=0.2,
                color='gray',
                label='Price Range'
            )
        
        # 设置股价图标题和标签
        ax_price.set_title(f'{self.symbol}: Stock Price Trend', 
                               fontsize=self.plot_styles.get_setting(STYLE.font_sizes, PLOT_ELEMENT.title),
                               fontweight='bold',
                               pad=20)
        ax_price.set_ylabel('Price (€)', 
                                 fontsize=self.plot_styles.get_setting(STYLE.font_sizes, PLOT_ELEMENT.axis_label))
        ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3, linestyle='--', 
                          color=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.grid_color))
        
        # 添加网格
        ax_price.grid(True, alpha=0.3, linestyle='--')
        
    def plot_volume_chart(self, ax_volume, volume_colors):
        """绘制交易量图"""
        # 绘制交易量柱状图
        bar = ax_volume.bar(
            self.dates_mpl,
            self.stock_data['Volume'],
            color=volume_colors,
            alpha=self.plot_styles.get_setting(STYLE.alphas, PLOT_ELEMENT.volume),
            width=0.8,  # 柱状图宽度
            edgecolor='black',
            linewidth=0.5
        )
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artist, bar, name='volume_bar', axes_name='axes_1')
        # 设置交易量图标签
        ax_volume.set_ylabel('Volume', 
                                  fontsize=self.plot_styles.get_setting(STYLE.font_sizes, PLOT_ELEMENT.axis_label))
        ax_volume.set_xlabel('Date', 
                                  fontsize=self.plot_styles.get_setting(STYLE.font_sizes, PLOT_ELEMENT.axis_label))
        ax_volume.grid(True, alpha=0.3, linestyle='--',
                           color=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.grid_color))
        
        # 格式化y轴标签（使用K/M/B表示大数字）
        ax_volume.yaxis.set_major_formatter(
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

    def format_chart_price(self, ax):
        # 设置y轴格式
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'€{x:.2f}')
        )

    def format_chart_bar(self, ax):
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
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(date_formatter)
        
        # 自动旋转日期标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
    def add_interactive_features(self, ax, ax_price=True):
        """添加交互功能"""
        if ax_price:
            self._add_interactive_features_to_price(ax)
        else:
            self._add_interactive_features_to_volume(ax)

        # 连接鼠标移动事件
        fig = self.visual_data.fig
        fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        
        # 连接鼠标离开事件
        fig.canvas.mpl_connect("axes_leave_event", self.on_leave)

    def _add_interactive_features_to_price(self, ax):
        # 创建悬停线（垂直虚线）
        hover_line = ax.axvline(
            x=self.dates_mpl[0],
            color=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.hover_line),
            linewidth=self.plot_styles.get_setting(STYLE.line_widths, PLOT_ELEMENT.hover_line),
            alpha=self.plot_styles.get_setting(STYLE.alphas, PLOT_ELEMENT.hover_line),
            visible=False,
            zorder=10
        )
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artist, hover_line, name='hover_line', axes_name='axes_0')
        # 创建股价注释框
        price_annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.annotation_bg),
                edgecolor='black',
                alpha=0.9
            ),
            fontsize=self.plot_styles.get_setting(STYLE.font_sizes, PLOT_ELEMENT.annotation),
            zorder=11
        )
        price_annotation.set_visible(False)
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artist, price_annotation, name='price_annotation', axes_name='axes_0')

    def _add_interactive_features_to_volume(self, ax):
        # 创建交易量注释框
        volume_annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, -30),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.annotation),
                edgecolor='black',
                alpha=0.9
            ),
            fontsize=self.plot_styles.get_setting(STYLE.font_sizes, PLOT_ELEMENT.annotation),
            zorder=11
        )
        volume_annotation.set_visible(False)
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artist, volume_annotation, name='volume_annotation', axes_name='axes_1')
    
    def on_hover(self, event):
        """鼠标悬停事件处理"""
        # 检查鼠标是否在图表区域内
        if event.inaxes in self.visual_data.axes:
            # 找到最近的日期点
            idx = np.abs(self.dates_mpl - event.xdata).argmin()
            
            # 更新悬停线位置
            hover_line = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artist, 'hover_line', axes_name='axes_0')
            hover_line.set_xdata([self.dates_mpl[idx]])
            hover_line.set_visible(True)
            
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
            
            price_annotation = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artist, 'price_annotation', axes_name='axes_0')  
            price_annotation.set_text(price_text)
            price_annotation.xy = (self.dates_mpl[idx], close_price)
            price_annotation.set_visible(True)
            
            # 更新交易量注释框
            volume_annotation = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artist, 'volume_annotation', axes_name='axes_1')
            volume_text = f"Date: {date_str}\n" \
                         f"Volume: {volume:,.0f}"
            volume_annotation.set_text(volume_text)
            volume_annotation.xy = (self.dates_mpl[idx], volume)
            volume_annotation.set_visible(True)
            
            # 重绘图形
            self.visual_data.fig.canvas.draw_idle()
    
    def on_leave(self, event):
        """鼠标离开图表区域事件处理"""
        # 隐藏注释框和悬停线
        hover_line = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artist, 'hover_line', axes_name='axes_0')
        hover_line.set_visible(False)
        price_annotation = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artist, 'price_annotation', axes_name='axes_0')  
        price_annotation.set_visible(False)
        volume_annotation = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artist, 'volume_annotation', axes_name='axes_1')
        volume_annotation.set_visible(False)
        self.visual_data.fig.canvas.draw_idle()
    
    def show(self):
        """显示图表"""
        if self.visual_data.fig is None:
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
        if self.visual_data.fig is None:
            self.create_plot()
        
        self.visual_data.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
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
        if self.axes[0] is None:
            raise ValueError("请先调用create_plot()或show()方法")
        
        # 计算移动平均
        ma_label = label if label else f'MA{window}'
        ma_data = self.stock_data['Close'].rolling(window=window).mean()
        
        # 绘制移动平均线
        axes0 = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.axes, 'axes_0')
        line = axes0.plot(
            self.dates_mpl,
            ma_data,
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=ma_label
        )
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artist, line, name='moving_average', axes_name='axes_0')
        # 更新图例
        axes0.legend(loc='upper left')
        
        if self.visual_data.fig is not None:
            self.visual_data.fig.canvas.draw_idle()
    
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
            self.plot_styles.set_setting(STYLE.colors, PLOT_ELEMENT.price_up, price_up)
        if price_down:
            self.plot_styles.set_setting(STYLE.colors, PLOT_ELEMENT.price_down, price_down)
        if volume_up:
            self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.volume_up, volume_up)
        if volume_down:
            self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.volume_down, volume_down)

    def _create_context_menu_commands(self):
        self.context_menu = Menu(self.tk_root, tearoff=0)
        self.menu_items = {}
        for e in StockChartPlotter.CONTEXT_MENU:
            if e.name == 'seperator':
                self.context_menu.add_separator()
            else:
                self.context_menu.add_command(label=e.name, command=self.dummy_command)
            self.menu_items[e.name] = self.context_menu.index("end")

    def dummy_command(self):
        pass

    def remove(self):
        moving_average_lines = self.visual_data.popup_stock_visual_data(StockVisualData.TYPE.artist, 'moving_average', axes_name='axes_0')
        if moving_average_lines is None:
            return False
        if len(moving_average_lines) == 0:
            return False
        line = moving_average_lines.pop()
        if line:
            for artist in line:
                if artist:
                    artist.remove()
        self.visual_data.fig.canvas.draw_idle()
        return True

    def remove_all(self):
        moving_average_lines = self.visual_data.popup_stock_visual_data(StockVisualData.TYPE.artist, 'moving_average', axes_name='axes_0')
        if moving_average_lines is None:
            return
        if len(moving_average_lines) == 0:
            return
        for line in moving_average_lines:
            if line:
                for artist in line:
                    if artist:
                        artist.remove()
        self.visual_data.fig.canvas.draw()

    def on_right_click(self, event):
        if event.button == 3:  # Right-click in axes
            # self.last_click_coords = (event.xdata, event.ydata)
            # self.fig._last_right_click = (event.xdata, event.ydata, event.inaxes)
            for key, idx in self.menu_items.items():
                # trick here: set key as a default argument to prevent of late binding of variables in lambdas inside a loop!!!
                if key == 'seperator':
                    continue
                elif key.startswith('ma'):
                    x = int(key.split('_')[1])
                    self.context_menu.entryconfig(idx, command=lambda item=x: self.add_moving_average(item, label=key))
                else:
                    self.context_menu.entryconfig(idx, command=lambda: getattr(self, key)())
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