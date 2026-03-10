from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        # - visual_data: dict {name1: {TYPE.ax : ax1, 
        #                              TYPE.artists: {artist1 : data, ...}, 
        #                              TYPE.properties: {property1 : data, ...}
        #                      },
        #                      {name2: {TYPE.ax : ax2, 
        #                               TYPE.artists: {artist2 : data, ...},
        #                               TYPE.properties: {property1 : data, ...}
        #                      }
        #                      ...
        self._fig = fig
        self.visual_data = {}

    @property
    def fig(self):
        return self._fig
    @fig.setter
    def fig(self, value):
        self._fig = value

    @property
    def data(self):
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
                    self.visual_data[name] = {
                        StockVisualData.TYPE.ax : data[i],
                        StockVisualData.TYPE.artists : {}, 
                        StockVisualData.TYPE.properties : {}
                    }
            else:
                self.visual_data[data_name] = {
                    StockVisualData.TYPE.ax : data,
                    StockVisualData.TYPE.artists : {},
                    StockVisualData.TYPE.properties : {}
                }
        else:
            if axes_name is None:
                raise ValueError("axes_name must be provided for artist data except for axes data")
            if self.visual_data.get(axes_name) is None:
                raise ValueError(f"axes_name '{axes_name}' does not exist in visual_data")
            self.visual_data[axes_name][data_type][data_name] = data
        
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
                    if self.visual_data.get(name) is None:
                        raise ValueError(f"axes_name '{name}' does not exist in visual_data")
                    if name in self.visual_data.keys():
                        self.visual_data[name][data_type] = data[i]
                    else:
                        self.add_stock_visual_data(data_type, data[i], name)
            else:
                if self.visual_data.get(axes_name) is None:
                    raise ValueError(f"axes_name '{axes_name}' does not exist in visual_data")
                if axes_name in self.visual_data.keys():
                    self.visual_data[axes_name][data_type] = data
                else:
                    self.add_stock_visual_data(data_type, data, axes_name)
        else: 
            self.visual_data[axes_name][data_type][data_name] = data

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
        if self.visual_data.get(axes_name) is None:
            print(f"axes_name '{axes_name}' does not exist in visual data")
            return None
        if data_type == StockVisualData.TYPE.ax:
            return self.visual_data[axes_name].get(data_type, None)
        else:
            if data_name is None:
                return self.visual_data[axes_name][data_type] # return all artists dict
            else:
                return self.visual_data[axes_name].get(data_type, {}).get(data_name, None)

    def remove_stock_visual_data(self, data_type:TYPE, axes_name, data_name=None):
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
        if self.visual_data.get(axes_name) is None:
            raise ValueError(f"axes_name '{axes_name}' does not exist in visual_data")
        if data_type == StockVisualData.TYPE.ax:
            if axes_name in self.visual_data:
                return self.visual_data.pop(axes_name)
        else:
            if data_name in self.visual_data[axes_name][data_type]:
                return self.visual_data[axes_name][data_type].pop(data_name)
        return None    

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
        if not pd.api.types.is_datetime64_any_dtype(self.stock_data['Date']):
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        
        # 将日期转换为matplotlib格式
        self.dates_mpl = mdates.date2num(self.stock_data['Date'])

        # feature used by plotting chart
        self._feature = 'Close'

        # valid fig and ax are available after plot is created.
        self.create_plot()

    @property
    def feature(self):
        return self._feature
    @feature.setter
    def feature(self, feat):
        self._feature = feat

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

    def add_custom_markers(self, name, indices=None, dates=None, marker='^', color='red', size=100):
        """
        在特定位置添加自定义标记
        
        参数:
        ----------
        indices : list
            要标记的数据点索引列表
        dates : list
            或者用日期列表指定要标记的点
        marker : str
            标记样式 ('^' 向上三角, 'v' 向下三角, 'o' 圆形, 's' 方形, '*' 星形)
        color : str
            标记颜色
        size : int
            标记大小
        """
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        if ax is None:
            raise ValueError("Please, call create_plot() or show() at first.")
        
        if indices is not None:
            # 使用索引标记
            x_coords = self.dates_mpl[indices]
            y_coords = self.stock_data['Close'].iloc[indices]
        elif dates is not None:
            # 使用日期标记
            date_indices = []
            for date in dates:
                # 找到最接近的日期索引
                date_num = mdates.date2num(pd.to_datetime(date))
                idx = np.abs(self.dates_mpl - date_num).argmin()
                date_indices.append(idx)
            
            x_coords = self.dates_mpl[date_indices]
            y_coords = self.stock_data['Close'].iloc[date_indices]
        else:
            raise ValueError("必须提供indices或dates参数")
        
        # 绘制标记
        markers = ax.scatter(
            x_coords, 
            y_coords,
            marker=marker,
            color=color,
            s=size,  # 标记大小
            zorder=10,  # 确保标记显示在最上层
            edgecolors='white',
            linewidth=1,
            label=f'Markers ({marker})'
        )
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE, markers, name, StockVisualData.AX_PRICE)
        # 更新图例
        ax.legend(loc='upper left')
        
        if self.fig is not None:
            self.fig.canvas.draw_idle()

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
            artist = self.visual_data.remove_stock_visual_data(StockVisualData.TYPE.artists, ax_name, data_name=name)
            if artist is None:
                continue
            artist.remove()
            print(f"Removed artist: {name}")
        self.visual_data.fig.canvas.draw_idle()

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
        
        self.visual_data.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
    
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
            self.plot_styles.set_setting(STYLE.colors, PLOT_ELEMENT.volume_up, volume_up)
        if volume_down:
            self.plot_styles.set_setting(STYLE.colors, PLOT_ELEMENT.volume_down, volume_down)

    def create_context_menu_commands(self):
        pass

    def dummy_command(self):
        pass

    def on_right_click(self, event):
        """右键点击事件处理，显示上下文菜单"""
        pass
    
