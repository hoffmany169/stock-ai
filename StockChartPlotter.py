from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplcursors
from matplotlib.patches import Rectangle
from plot_style import PlotStyle, PLOT_ELEMENT, STYLE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Common.AutoNumber import AutoIndex
from StockModel import StockModel

class StockVisualData:
    class TYPE(AutoIndex):
        ax = ()
        artists = ()
        properties = ()

    AX_PRICE = 'ax_price'
    AX_VOLUME = 'ax_volume'
    def __init__(self, fig=None):
        # 存储图表配置和元素的对象，包含以下属性:
        # - fig: plot figure
        # - visual_data: dict {
        #                      {name1: {TYPE.ax : ax1, 
        #                              TYPE.artists: {artist1 : data, ...}, 
        #                              TYPE.properties: {property1 : data, ...}
        #                      ,
        #                      {name2: {TYPE.ax : ax2, 
        #                               TYPE.artists: {artist2 : data, ...},
        #                               TYPE.properties: {property1 : data, ...}
        #                      }
        #                      ...
        #                      }
        self._fig = fig
        self.visual_data = {}
        # instance of mplcursors
        self.curve_cursor = None

    @property
    def fig(self):
        return self._fig
    @fig.setter
    def fig(self, value):
        self._fig = value

    @property
    def plot_data(self):
        return self.visual_data

    def add_stock_visual_data(self, data_type:TYPE, data:any, data_name, axes_name=None):
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
        name : str
            数据名称，用于定位具体的数据项
        axes_name : str
            数据区域名称，用于加入或获取数据，如果数据类型是artist或property，axes_name必须提供，用于定位数据所在的坐标轴
        """
        # 根据数据类型添加数据到图表
        if data_name is None:
            raise ValueError("data_name must be provided for all data types")
        if data_type == StockVisualData.TYPE.ax:
            if (type(data) == list and (data_name is not None and type(data_name) != list) and  len(data) != len(data_name)):
                raise ValueError("if data is a list, data_name must be a list, and length of data must be same.")
            if type(data) == list:
                for i, name in enumerate(data_name):
                    self.plot_data[name] = {
                        StockVisualData.TYPE.ax : data[i],
                        StockVisualData.TYPE.artists : {}, 
                        StockVisualData.TYPE.properties : {}
                    }
            else:
                self.plot_data[data_name] = {
                    StockVisualData.TYPE.ax : data,
                    StockVisualData.TYPE.artists : {},
                    StockVisualData.TYPE.properties : {}
                }
        else:
            if axes_name is None:
                raise ValueError("axes_name must be provided for artist data except for axes data")
            if self.plot_data.get(axes_name) is None:
                raise ValueError(f"axes_name '{axes_name}' does not exist in visual_data")
            self.plot_data[axes_name][data_type][data_name] = data
        
        # 重绘图表
        if self._fig is not None:
            self._fig.canvas.draw_idle()

    def set_stock_visual_data(self, data_type:TYPE, data:any, axes_name, data_name=None):
        """设置图表中的股票数据
        
        参数:
        ----------
        data_type : StockVisualData
            数据类型，包含以下属性:
            - axes: 图表坐标轴对象
            - artist: 图表元素对象（如线条、柱状图等）
            - property: 数据属性（如价格、交易量等）
        data : any
            数据内容，具体格式取决于数据类型
        name : str
            数据名称，用于定位具体的数据项
        axes_name : str
            数据区域名称，用于定位数据所在的坐标轴，如果数据类型是artist或property，axes_name必须提供，用于定位数据所在的坐标轴
        """
        if axes_name is None:
            raise ValueError("axes_name must be provided for artist data except for axes data")
        
        if data_type == StockVisualData.TYPE.ax:
            if type(data) == list:
                for i, name in enumerate(axes_name):
                    if self.plot_data.get(name) is None:
                        raise ValueError(f"axes_name '{name}' does not exist in visual_data")
                    if name in self.plot_data.keys():
                        self.plot_data[name][data_type] = data[i]
                    else:
                        self.add_stock_visual_data(data_type, data[i], name)
            else:
                if self.plot_data.get(axes_name) is None:
                    raise ValueError(f"axes_name '{axes_name}' does not exist in visual_data")
                if axes_name in self.plot_data.keys():
                    self.plot_data[axes_name][data_type] = data
                else:
                    self.add_stock_visual_data(data_type, data, axes_name)
        else: 
            self.plot_data[axes_name][data_type][data_name] = data

    def get_stock_visual_data(self, data_type:TYPE, axes_name, data_name=None):
        """获取图表中的股票数据
        
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
        返回:
        any
            返回指定的数据项内容
        """
        if self.plot_data.get(axes_name) is None:
            print(f"axes_name '{axes_name}' does not exist in visual data")
            return None
        if data_type == StockVisualData.TYPE.ax:
            return self.plot_data[axes_name].get(data_type, None)
        else:
            if data_name is None:
                return self.plot_data[axes_name][data_type] # return all artists dict
            else:
                return self.plot_data[axes_name].get(data_type, {}).get(data_name, None)

    def remove_stock_visual_data(self, data_type:TYPE, axes_name, data_name):
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
        return: element which is removed
        """
        if self.plot_data.get(axes_name) is None:
            raise ValueError(f"axes_name '{axes_name}' does not exist in visual_data")
        if data_type == StockVisualData.TYPE.ax:
            raise ValueError("Remove ax data self, please, use function remove_visual_ax_data")
        else:
            if data_name is None:
                raise ValueError(f"data name is None. Cannot remove data {data_type.name}")
            if data_name in self.plot_data[axes_name][data_type]:
                return self.plot_data[axes_name][data_type].pop(data_name)
        return None    

    def remove_artists(self, ax_name, data_name=None):
        if data_name is None: # remove all artists
            for name, artist in self.plot_data[ax_name][StockVisualData.TYPE.artists].items():
                if artist:
                    artist.remove()
                    print(f"remove artist: {name}")
            self.plot_data[ax_name][StockVisualData.TYPE.artists].clear()
        else: 
             for name, artist in self.plot_data[ax_name][StockVisualData.TYPE.artists].items():
                if name == data_name and artist:
                    artist.remove()
                    print(f"remove artist: {name}")
                    self.plot_data[ax_name][StockVisualData.TYPE.artists][name] = None

    def remove_visual_ax_data(self, ax_name, only_chart=True):
        for data_type, data in self.plot_data[ax_name].items():
            if data_type == self.TYPE.ax:
                if not only_chart:
                    self.plot_data[ax_name][data_type].clear()
                    print(f"removed ax data {ax_name}")
            elif data_type == self.TYPE.artists:
                self.remove_artists(ax_name)
            else:
                self.plot_data[ax_name][data_type].clear()
                print(f"removed data in {data_type.name}")

    def clear_ax(self, ax_name):
        ax = self.get_stock_visual_data(self.TYPE.ax, ax_name)
        if ax:
            ax.clear()

class PointData:

    def __init__(self, selection):
        self._x = selection.target[0]
        self._value = selection.target[1]
        self._index = selection.index
        self.date_str = self.get_date_from_x(self._x) ## mdates.num2date(self.x).strftime('%Y-%m-%d')

    def __str__(self):
        return f"Date: {self.date_str}\nPrice: {self._value:.2f}\nIndex: {round(self._index)}"

    def __repr__(self):
        return f"Date='{self.date_str}' PointData(x='{self._x}', price={self._value:.2f}, index={round(self._index)})"
    
    def __eq__(self, other):
        if isinstance(other, PointData):
            return self.date_str == other.date_str
        return False

    @staticmethod
    def get_date_from_x(x)->str:
        return mdates.num2date(x).strftime('%Y-%m-%d')

    @staticmethod
    def get_datetime_from_date_string(ds)->datetime:
        return datetime.strptime(ds, '%Y-%m-%d')

    @property
    def Value(self):
        return self._value
    
    @property
    def Coordinate(self):
        return (self._x, self._value)

    @property
    def Index(self):
        return self._index
    
    @property
    def Date(self):
        return self.date_str
    
    def compare_with(self, other, to_string=False) -> str|dict:
        if self.date_str == other.date_str:
            return "Same point selected"
        x_diff = other.x - self.x
        price_diff = other.price - self._value
        percentage_change = (price_diff / self._value * 100) if self._value != 0 else float('inf')
        tangent = (price_diff / x_diff) if x_diff != 0 else float('inf')
        date_diff = self.get_datetime_from_date_string(other.date_str) - self.get_datetime_from_date_string(self.date_str) # datetime.strptime(other.date_str, '%Y-%m-%d') - datetime.strptime(self.date_str, '%Y-%m-%d') 
    
        result = {
            'price_diff': price_diff,
            'percentage_change': percentage_change,
            'tangent': tangent,
            'date_span': date_diff
        }
        if to_string:
            return f"price change: {result['price_diff']:.2f}({result['percentage_change']:.2f}%)\ntangent: {result['tangent']:.2f}\ndate span: {result['date_span'].days} days"
        return result

    def compare_feature_value(self, other, stock_data, feature) -> str:
        if self.date_str == other.date_str:
            return "Same point selected"
        if feature in stock_data.columns:
            date_diff = datetime.strptime(other.date_str, '%Y-%m-%d') - datetime.strptime(self.date_str, '%Y-%m-%d') 
            value = self.get_feature_value(stock_data, feature)
            other_value = other.get_feature_value(stock_data, feature)
            value_diff = other_value - value
            percentage_change = (value_diff / value * 100) if value != 0 else float('inf')
            return f"Days:{date_diff.days}\n{feature} change: {value_diff:.2f}({percentage_change:.2f}%)"

    def sum_feature_value(self, other, stock_data, feature):
        if self.date_str == other.date_str:
            return "Same point selected"
        sum_value = 0
        if feature in stock_data.columns:
            date_diff = datetime.strptime(other.date_str, '%Y-%m-%d') - datetime.strptime(self.date_str, '%Y-%m-%d') 
            feature_column = stock_data[feature].fillna(0)  # Ensure feature column has no NaN values
            for d in range(0, other.index - self._index + 1):
                current_index = self._index + d
                sum_value += feature_column.iloc[int(current_index)]
            return f"Days:{date_diff.days}\nsum_value: {sum_value:.2f}"
        return ""

class StockChartPlotter(ABC):
    """
    股票图表绘制类
    功能：
    1. 绘制股价走势图（主图）
    2. 绘制交易量柱状图（副图）
    3. 鼠标悬停显示数据点详细信息
    4. 支持自定义样式和配置
    """
    def __init__(self, stock_model, figsize=(14, 10)):
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
        self.symbol = stock_model.ticker_symbol
        self.stock_model = stock_model
        self.stock_data = stock_model.loaded_data
        self.figsize = figsize
        self.visual_data = StockVisualData()
        self.plot_styles = PlotStyle()
        #parent window of tkinter
        self.parent = None
        # canvas for tkinter from plot figure, original figure is saved in self.visual_data.fig
        self.fig_canvas = None 
        # tk widget obtained from figure canvas
        self.tk_root = None 
        
        # 确保日期为datetime格式
        # if not pd.api.types.is_datetime64_any_dtype(self.stock_data['Date']):
        #     self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        
        # # 将日期转换为matplotlib格式
        # self.dates_mpl = mdates.date2num(self.stock_data['Date'])
        self.convert_date_to_matplotlib_format()

        # feature used by plotting chart
        self._feature = 'Close'

        # mplcursors interactive
        self.ax_cursor = None
        # valid fig and ax are available after plot is created.
        self.create_plot()

#region properties
    @property
    def feature(self):
        return self._feature
    @feature.setter
    def feature(self, feat):
        self._feature = feat

    @property
    def ticker_symbol(self):
        return self.stock_model.ticker_symbol
#endregion properties

    def convert_date_to_matplotlib_format(self):
        # 确保日期为datetime格式
        if not pd.api.types.is_datetime64_any_dtype(self.stock_data['Date']):
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        self.dates_mpl = mdates.date2num(self.stock_data['Date'])

    # after plot is created, create window controls, e.g. context menu
    def set_backend_window(self, parent):
        self.parent = parent
        # add plot canvas of figure to tkinter window
        self.fig_canvas = FigureCanvasTkAgg(self.visual_data.fig, master=parent)
        # get root of window, which is top level window of canvas, also sub-window of parent tkinter window 
        self.tk_root = self.fig_canvas.get_tk_widget() # root of this figure canvas
        self.create_context_menu_commands()
        # We'll create the menu dynamically each time. add event to figure's canvas        
        self.fig_canvas.mpl_connect('button_press_event', self.on_right_click)
        return (self.visual_data.fig, self.tk_root) # root is canvas of parent figure

    def switch_mplcursors(self, ax, on:bool, hover_opt:int=2):
        # hover_opt: 2 - Transient hover mode: the annotation appears when the mouse is near a point and disappears when the mouse moves away, with a delay of 2 seconds before disappearing. This mode is useful for providing information about points without requiring a click,
        #                while also ensuring that the annotation does not linger on the screen for too long.
        self.ax_cursor = mplcursors.cursor(ax, hover=hover_opt)
        if on:
            self.ax_cursor.connect("add", self.on_add)
            self.ax_cursor.connect("remove", self.on_remove)
        else:
            self.ax_cursor.disconnect("add", self.on_add)
            self.ax_cursor.disconnect("remove", self.on_remove)

    def calculate_price_change(self, column_1='Open', column_2='Close'):
        """计算价格涨跌，用于确定颜色"""
        if column_1 in self.stock_data.columns and column_2 in self.stock_data.columns:
            # 如果有开盘价，用收盘价与开盘价比较
            price_change = self.stock_data[column_2] >= self.stock_data[column_1]
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
    @abstractmethod
    def create_plot(self):
        pass

    @abstractmethod    
    def plot(self):
        """绘制股价图"""
        pass

    @abstractmethod
    def on_hover_info(self, *args):
        """获取悬停信息"""
        pass

    @abstractmethod
    def on_leave_info(self, *args):
        """获取鼠标离开信息"""
        pass

    def format_large_numbers(self, x, pos)->str:
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
            
    def add_mouse_hover_event(self, ax):
        """添加交互功能"""
        # 连接鼠标移动hover事件
        fig = self.visual_data.fig
        if fig is None:
            return
        fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        
        # 连接鼠标离开事件
        fig.canvas.mpl_connect("axes_leave_event", self.on_leave)

    def on_add(self, sel):
        """event callback for mplcursors.connect("add", callback)"""
        pass

    def on_remove(self, sel):
        """event callback for mplcursors.connect("remove", callback)"""
        pass

    def on_leave(self, event):
        """鼠标离开图表区域事件处理"""
        self.on_leave_info()
        self.visual_data.fig.canvas.draw_idle()
    
    def on_hover(self, event):
        """鼠标悬停事件处理"""
        # 检查鼠标是否在图表区域内
        if event.inaxes in [self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE), self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, 'ax_volume')]:
            # 找到最近的日期点
            idx = np.abs(self.dates_mpl - event.xdata).argmin()
            
            # 更新悬停线位置
            hover_line = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, StockVisualData.AX_PRICE, data_name='hover_line')
            hover_line.set_xdata([self.dates_mpl[idx]])
            hover_line.set_visible(True)
            
            self.on_hover_info(idx)
            
            # 重绘图形
            self.visual_data.fig.canvas.draw_idle()
    
    def highlight_peaks_valleys(self, ax_name:str, feature:str,
                                window=5, 
                                peak_color='green', 
                                valley_color='red',
                                peak_marker='v',
                                valley_marker='^',
                                peak_label='Local Highest Point',
                                valley_label='Local Lowest Point',
                                artist_names=('peaks', 'valleys'),
                                start_date: str|None=None,
                                end_date:str|None=None):
        """
        标记局部高点和低点
        
        参数:
        ----------
        window : int
            用于识别局部极值的窗口大小
        peak_color : str
            高点标记颜色
        valley_color : str
            低点标记颜色
        """
        ax_main = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, ax_name)
        if ax_main is None:
            raise ValueError("请先调用create_plot()或show()方法")
        from StockModel import StockModel
        if feature not in [n.name for n in StockModel.FEATURE]:
            raise ValueError(f"{feature} does not exist in stock features!")
        feature_data = self.stock_data[feature].values
        
        peaks, valleys = self.find_peaks_valleys(feature, window, start_date=start_date, end_date=end_date)

        # 绘制高点标记
        if peaks and len(peaks) > 0:
            artist1 = ax_main.scatter(
                        self.dates_mpl[peaks],
                        feature_data[peaks],
                        marker=peak_marker,  # 向下三角表示高点（因为要下跌）
                        color=peak_color,
                        s=120,
                        zorder=10,
                        # edgecolors='white',
                        linewidth=1.5,
                        label=peak_label
                    )
            self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, artist1, artist_names[0], axes_name=StockVisualData.AX_PRICE)
        
        # 绘制低点标记
        if valleys and len(valleys) > 0:
            artist2 = ax_main.scatter(
                        self.dates_mpl[valleys],
                        feature_data[valleys],
                        marker=valley_marker,  # 向上三角表示低点（因为要上涨）
                        color=valley_color,
                        s=120,
                        zorder=10,
                        # edgecolors='white',
                        linewidth=1.5,
                        label=valley_label
                    )
            self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, artist2, artist_names[1], axes_name=StockVisualData.AX_PRICE)
        ax_main.legend(loc='upper left')
        if self.parent is None:
            self.visual_data.fig.canvas.draw_idle()
        else:
            self.tk_root.update()
        return (peaks, valleys)

    def find_peaks_valleys(self,
                           feature,
                           window=2,
                           start_date=None,
                           end_date=None):
        if feature not in [f.name for f in StockModel.FEATURE]:
            return None

        start_idx, end_idx = self.stock_model.get_data_absolute_index_by_date_range(start_date, end_date, window)
        feature_data = self.stock_data[feature]
        if len(feature_data) <= window:
            print("Window size is too big")
            return None
        # 仅在指定范围内识别局部高点
        peaks = []
        for i in range(start_idx, end_idx + 1):
            # 检查是否为局部高点
            left_window = feature_data.iloc[max(0, i-window):i]
            right_window = feature_data.iloc[i+1:min(len(feature_data), i+window+1)]
            
            if len(left_window) > 0 and len(right_window) > 0:
                if i == end_idx: # treat last point as a valid point
                    if feature_data.iloc[i] >= np.max(left_window):
                        peaks.append(i)
                else:
                    if feature_data.iloc[i] >= np.max(left_window) and feature_data.iloc[i] >= np.max(right_window):
                        peaks.append(i)

        valleys = []
        for i in range(start_idx, end_idx + 1):
            left_window = feature_data.iloc[max(0, i-window):i]
            right_window = feature_data.iloc[i+1:min(len(feature_data), i+window+1)]
            
            if len(left_window) > 0 and len(right_window) > 0:
                if i == end_idx: # treat last point as a valid point
                    if feature_data.iloc[i] <= np.min(left_window):
                        valleys.append(i)
                else:
                    if feature_data.iloc[i] <= np.min(left_window) and feature_data.iloc[i] <= np.min(right_window):
                        valleys.append(i)
        return (peaks, valleys)

    def remove_highlighted_peaks_valleys(self, ax_name=StockVisualData.AX_PRICE, artist_names=('peaks', 'valleys')):
        self.remove_artist(ax_name, artist_names[0])
        self.remove_artist(ax_name, artist_names[1])

    def remove_artist(self, axes_name, data_name):
        artist = self.visual_data.remove_stock_visual_data(StockVisualData.TYPE.artists, axes_name, data_name)
        if artist is None:
            return
        if type(artist) is tuple or type(artist) is list:
            for art in artist:
                if art:
                    art.remove()
        else:
            if artist:
                artist.remove()
        self.visual_data.fig.canvas.draw_idle()

    def remove_all_artists(self, ax_name):
        artists = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, ax_name)
        for name in artists.keys():
            artist = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, ax_name, data_name=name)
            if artist is None:
                continue
            artist.remove()
            print(f"Removed artist: {name}")
        # keep data structure
        self.visual_data.plot_data[ax_name]['artists'].clear()
        self.visual_data.fig.canvas.draw_idle()

    def clear_all_plot_data(self):
        """清空所有图形对象"""
        # 移除价格线
        for ax_name, ax in self.visual_data.plot_data.items():
            print(f"removing all ax {ax_name}")
            for name, values in ax.items():
                print(f"- removing {name}")
                if len(values) == 0:
                    continue
                if name == 'artists':
                    self.remove_all_artists(ax_name)
                else:
                    self.visual_data.plot_data[ax_name][name].clear()

    def show(self):
        """显示图表"""
        if self.visual_data.fig is None:
            self.create_plot()
        # if parent is a tkinter widget, don't call show() function.
        if self.parent is None:
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

         # 确保布局正确再保存
        self.visual_data.fig.tight_layout()
        
        # 保存时使用bbox_inches='tight'
        self.visual_data.fig.savefig(
            filename, 
            dpi=dpi, 
            bbox_inches='tight',
            pad_inches=0.1  # 添加一点内边距
        )       
        # self.visual_data.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
    
    def create_context_menu_commands(self):
        pass

    def dummy_command(self):
        pass

    def on_right_click(self, event):
        """右键点击事件处理，显示上下文菜单"""
        pass
    
    def diagnose_and_fix_layout(self, fig):
        """诊断并修复布局问题"""
        
        print("=" * 50)
        print("布局诊断")
        print("=" * 50)
        
        # 获取当前布局信息
        try:
            # 尝试使用tight_layout
            print("\n尝试使用 tight_layout...")
            fig.tight_layout()
            print("✅ tight_layout 应用成功")
        except Exception as e:
            print(f"❌ tight_layout 失败: {e}")
        
        # 获取当前边距
        try:
            bbox = fig.get_tightbbox()
            print(f"\n当前图形边界: {bbox}")
        except:
            pass
        
        # 显示当前边距设置
        print(f"\n当前边距设置:")
        print(f"  left: {fig.subplotpars.left}")
        print(f"  bottom: {fig.subplotpars.bottom}")
        print(f"  right: {fig.subplotpars.right}")
        print(f"  top: {fig.subplotpars.top}")
        print(f"  hspace: {fig.subplotpars.hspace}")
        print(f"  wspace: {fig.subplotpars.wspace}")
        
        # 提供修复建议
        print("\n修复建议:")
        print("1. 使用 fig.tight_layout()")
        print("2. 使用 fig.subplots_adjust() 手动调整")
        print("3. 创建图形时使用 constrained_layout=True")
        print("4. 保存时使用 bbox_inches='tight'")
        
        print("\n示例手动调整:")
        print("fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)")
        
        return fig

    # # 使用示例
    # fig, ax = plt.subplots()
    # ax.plot([1, 2, 3], [1, 2, 3])
    # ax.set_title('测试标题', fontsize=20)
    # ax.set_xlabel('X轴标签', fontsize=15)
    # ax.set_ylabel('Y轴标签', fontsize=15)

    # # 诊断并修复
    # fig = diagnose_and_fix_layout(fig)

    # plt.show()

    @staticmethod
    def diagnose_matplotlib_toolbar(self):
        """诊断matplotlib工具栏问题"""
        import matplotlib
        
        print("=" * 50)
        print("MATPLOTLIB 工具栏诊断")
        print("=" * 50)
        
        # 1. 检查版本
        print(f"\n1. Matplotlib版本: {matplotlib.__version__}")
        
        # 2. 检查后端
        backend = matplotlib.get_backend()
        print(f"\n2. 当前后端: {backend}")
        
        # 3. 检查工具栏设置
        toolbar_setting = plt.rcParams.get('toolbar', '未设置')
        print(f"\n3. 工具栏设置: {toolbar_setting}")
        
        # 4. 检查交互模式
        is_interactive = plt.isinteractive()
        print(f"\n4. 交互模式: {'开启' if is_interactive else '关闭'}")
        
        # 5. 检查是否在IPython环境中
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                print(f"\n5. 运行环境: IPython ({ip.__class__.__name__})")
                if 'ZMQInteractiveShell' in str(type(ip)):
                    print("   - 这是Jupyter notebook/实验室环境")
                    print("   - 建议使用 %matplotlib qt 或 %matplotlib widget")
            else:
                print(f"\n5. 运行环境: 标准Python ({sys.executable})")
        except ImportError:
            print(f"\n5. 运行环境: 标准Python ({sys.executable})")
        
        # 6. 检查可用后端
        print(f"\n6. 可用后端:")
        for backend_name in matplotlib.rcsetup.all_backends:
            if 'Agg' not in backend_name:  # 只显示交互式后端
                print(f"   - {backend_name}")
        
        # 7. 测试创建图形
        print(f"\n7. 测试创建图形...")
        try:
            # 尝试设置工具栏
            plt.rcParams['toolbar'] = 'toolbar2'
            
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            
            # 检查是否有工具栏
            canvas = fig.canvas
            if hasattr(canvas, 'toolbar'):
                toolbar = canvas.toolbar
                if toolbar is not None:
                    print(f"   ✅ 工具栏存在: {type(toolbar).__name__}")
                    print(f"   - 工具栏可见: {getattr(toolbar, 'visible', 'N/A')}")
                else:
                    print("   ❌ 工具栏对象为None")
            else:
                print("   ❌ canvas没有toolbar属性")
                print(f"   canvas类型: {type(canvas).__name__}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"   ❌ 创建图形时出错: {e}")
        
        print("\n" + "=" * 50)
        print("诊断完成")
        print("=" * 50)

    ########### 运行诊断 ###########
    # diagnose_matplotlib_toolbar()

    # # 提供修复建议
    # print("\n🔧 修复建议:")
    # print("-" * 30)
    # print("1. 如果使用非交互式后端，在代码开头添加:")
    # print("   import matplotlib")
    # print("   matplotlib.use('Qt5Agg')  # 或 'TkAgg'")
    # print("   import matplotlib.pyplot as plt")
    # print()
    # print("2. 如果在Jupyter中，使用:")
    # print("   %matplotlib qt  # 弹出独立窗口")
    # print("   # 或")
    # print("   %matplotlib widget  # 安装ipympl后在notebook内交互")
    # print()
    # print("3. 如果在PyCharm中:")
    # print("   取消 Settings -> Tools -> Python Scientific -> Show plots in toolwindow")
    # print()
    # print("4. 检查matplotlib安装:")
    # print("   pip install --upgrade matplotlib")
    # print("   # 对于Qt支持:")
    # print("   pip install pyqt5  # 或 pyside2")

