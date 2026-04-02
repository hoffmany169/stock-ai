from matplotlib import pyplot as plt
from plot_style import PlotStyle, PLOT_ELEMENT, STYLE
from StockChartPlotter import StockChartPlotter, StockVisualData
import matplotlib.dates as mdates
from tkinter import Menu
from Common.AutoNumber import AutoIndex

class PriceVolumePlotter(StockChartPlotter):
    class CONTEXT_MENU(AutoIndex):
        ma_5 = ()
        ma_20 = ()
        ma_90 = ()
        seperator = ()
        remove_artist = ()
        remove_all_artists = ()

    def __init__(self):
        super().__init__()

    def create_gui(self):
        pass

    def create_plot(self, stock_model=None, feature='Close', figsize=(14, 10)):
        """创建主图表"""
        super().create_plot(stock_model, feature, figsize)
        # 创建图形和坐标轴,使用constrained_layout（最简单）
        fig, (ax_price, ax_volume) = plt.subplots(
            2, 1, 
            figsize=self.figsize,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
            constrained_layout=True  # 添加这个参数保证图形完整显示在窗口中
        )
        if fig is None:
            raise ValueError("fig is None")
        # 3. 最后再微调
        # self.fig.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95)
        self.visual_data.fig = fig
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.ax, 
                                               [ax_price, ax_volume],
                                               [StockVisualData.AX_PRICE, StockVisualData.AX_VOLUME])
        # 计算涨跌颜色
        colors = self.calculate_price_change()
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.properties, colors, 'price_change', axes_name=StockVisualData.AX_PRICE)
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.properties, colors, 'price_change', axes_name=StockVisualData.AX_VOLUME)
        
        
        # 添加交互功能
        self.add_mouse_hover_event(ax_price)
        self._add_interactive_features_to_volume(ax_volume)
        super().add_mouse_hover_event(ax_volume)

        self.visual_data.set_stock_visual_data(StockVisualData.TYPE.ax, [ax_price, ax_volume], 
                                               [StockVisualData.AX_PRICE, StockVisualData.AX_VOLUME])
        
        # Check current toolbar setting， default: toolbar2
        # current_toolbar = plt.rcParams['toolbar']
        # print(f"Current toolbar: {current_toolbar}")
        # List all available rcParams
        # print(plt.rcParams.keys())        
        # plt.rcParams['toolbar'] = 'toolmanager'
        # 手动调整方案，但不工作
        # plt.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.8)
        self.plot()
        # 调整布局
        plt.tight_layout()

    def plot(self):
        """绘制图表"""
        # 绘制股价图
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        self.plot_price_chart(ax, self._feature)
        
        # 绘制交易量图
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_VOLUME)
        if ax:
            self.plot_volume_chart(ax, self.visual_data.get_stock_visual_data(StockVisualData.TYPE.properties, 'ax_volume', data_name='price_change'))
        
    def update_data_dynamically(self, new_stock_model, feature=None):
        """
        动态更新数据而不重新创建图形对象（最高效）
        """
        from tkinter import messagebox
        if not new_stock_model.is_data_loaded:
            messagebox.showwarning("Warning", f"Stock data of {new_stock_model.ticker_symbol} is not loaded, yet!")
            return
        # 更新内部数据
        self.stock_model = new_stock_model
        self.stock_data = self.stock_model.loaded_data
        self.symbol = self.stock_model.ticker_symbol
        if feature is not None:
            self._feature = feature
        # new_dates_mpl = mdates.date2num(self.stock_data['Date'])
        # self.dates_mpl = mdates.date2num(self.stock_data['Date'])
        self.convert_date_to_matplotlib_format()
        
        # 更新价格线的数据
        ax_price = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax,
                                                          StockVisualData.AX_PRICE)
        price_line = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists,
                                                          StockVisualData.AX_PRICE,
                                                          data_name='price_line')
        if price_line:
            price_line.set_data(self.dates_mpl, self.stock_data[self.feature])
        
        # 更新x轴范围
        ax_price.set_xlim(self.dates_mpl[0], self.dates_mpl[-1])
        
        # 更新y轴范围
        y_min = self.stock_data[self.feature].min() * 0.95
        y_max = self.stock_data[self.feature].max() * 1.05
        ax_price.set_ylim(y_min, y_max)
        
        # 如果有交易量图，也需要更新
        ax_volume = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax,
                                                          StockVisualData.AX_VOLUME)
        if ax_volume:
            # 清除旧的柱状图
            ax_volume.clear()
            
            # 重新绘制交易量图
            volume_colors = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.properties, 
                                                                   StockVisualData.AX_VOLUME,
                                                                   'price_change')
            # self.calculate_price_change()
            self.plot_volume_chart(ax_volume, volume_colors)
            
            # 更新x轴范围
            ax_volume.set_xlim(self.dates_mpl[0], self.dates_mpl[-1])
        
        # 重绘
        self.fig_canvas.draw_idle()

    def plot_price_chart(self, ax_price, feature):
        """绘制股价图"""
        # 绘制收盘价折线
        price_line, = ax_price.plot(
            self.dates_mpl, 
            self.stock_data[feature],
            color=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.price_line),
            linewidth=self.plot_styles.get_setting(STYLE.line_widths, PLOT_ELEMENT.price_line),
            label=f'{feature} Price',
            zorder=5
        )
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, price_line, 'price_line', axes_name=StockVisualData.AX_PRICE)
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
        # 配置图表格式
        self.format_chart_price(ax_price)
    
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
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, bar, 'volume_bar', axes_name='ax_volume')
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
        # 配置图表格式
        self.format_chart_bar(ax_volume)

    def add_mouse_hover_event(self, ax):
        # 创建悬停线（垂直虚线）
        hover_line = ax.axvline(
            x=self.dates_mpl[0],
            color=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.hover_line),
            linewidth=self.plot_styles.get_setting(STYLE.line_widths, PLOT_ELEMENT.hover_line),
            alpha=self.plot_styles.get_setting(STYLE.alphas, PLOT_ELEMENT.hover_line),
            visible=False,
            zorder=10
        )
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, hover_line, 'hover_line', axes_name=StockVisualData.AX_PRICE)
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
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, price_annotation, 'price_annotation', axes_name=StockVisualData.AX_PRICE)
        super().add_mouse_hover_event(ax)

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
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, volume_annotation, 'volume_annotation', axes_name='ax_volume')
    
    def on_hover_info(self, *args):
        date_idx = args[0]
        # 获取数据
        date_str = mdates.num2date(self.dates_mpl[date_idx]).strftime('%Y-%m-%d')
        close_price = self.stock_data['Close'].iloc[date_idx]
        volume = self.stock_data['Volume'].iloc[date_idx]
        
        # 获取其他可选数据
        extra_info = ""
        if 'Open' in self.stock_data.columns:
            open_price = self.stock_data['Open'].iloc[date_idx]
            extra_info += f"Open: €{open_price:.2f}\n"
        if 'High' in self.stock_data.columns:
            high_price = self.stock_data['High'].iloc[date_idx]
            extra_info += f"High: €{high_price:.2f}\n"
        if 'Low' in self.stock_data.columns:
            low_price = self.stock_data['Low'].iloc[date_idx]
            extra_info += f"Low: €{low_price:.2f}\n"
        
        # 计算涨跌幅
        if date_idx > 0:
            prev_close = self.stock_data['Close'].iloc[date_idx-1]
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
        
        price_annotation = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, StockVisualData.AX_PRICE, data_name='price_annotation')  
        price_annotation.set_text(price_text)
        price_annotation.xy = (self.dates_mpl[date_idx], close_price)
        price_annotation.set_visible(True)
        
        # 更新交易量注释框
        volume_annotation = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, 'ax_volume', data_name='volume_annotation')
        volume_text = f"Date: {date_str}\n" \
                        f"Volume: {volume:,.0f}"
        volume_annotation.set_text(volume_text)
        volume_annotation.xy = (self.dates_mpl[date_idx], volume)
        volume_annotation.set_visible(True)

    def on_leave_info(self, *args):
        # 隐藏注释框和悬停线
        hover_line = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, StockVisualData.AX_PRICE, data_name='hover_line')
        hover_line.set_visible(False)
        price_annotation = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, StockVisualData.AX_PRICE, data_name='price_annotation')  
        price_annotation.set_visible(False)
        volume_annotation = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, 'ax_volume', data_name='volume_annotation')
        volume_annotation.set_visible(False)

    def add_moving_average(self, ax_name, window=20, color='orange', label=None):
        """添加移动平均线
        
        参数:
        ----------
        ax_name : str
            要添加移动平均线的坐标轴名称
        window : int
            移动平均窗口大小
        color : str
            线条颜色
        label : str
            图例标签，如果为None则使用'MA{window}'
        """
        if self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, ax_name) is None:
            raise ValueError("请先调用create_plot()或show()方法")
        
        # 计算移动平均
        ma_label = label if label else f'MA{window}'
        ma_data = self.stock_data['Close'].rolling(window=window).mean()
        
        # 绘制移动平均线
        axes0 = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, ax_name)
        line = axes0.plot(
            self.dates_mpl,
            ma_data,
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=ma_label
        )
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, line, 'moving_average', axes_name=ax_name)
        # 更新图例
        axes0.legend(loc='upper left')
        
        if self.visual_data.fig is not None:
            self.visual_data.fig.canvas.draw_idle()
    
    def create_context_menu_commands(self):
        self.context_menu = Menu(self.tk_root, tearoff=0)
        self.menu_items = {}
        for e in PriceVolumePlotter.CONTEXT_MENU:
            if e.name == 'seperator':
                self.context_menu.add_separator()
            else:
                self.context_menu.add_command(label=e.name, command=self.dummy_command)
            self.menu_items[e.name] = self.context_menu.index("end")

    def remove_artist(self, ax_name, data_name):
        super().remove_artist(ax_name, data_name)

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
                x_tk = self.tk_root.winfo_pointerx()
                y_tk = self.tk_root.winfo_pointery()
                self.context_menu.tk_popup(x_tk, y_tk)
            except:
                # Fallback
                self.context_menu.post()

# ==================== 使用示例 ====================
import numpy as np
import pandas as pd

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

from tkinter import Tk
from tkinter import Frame, BOTH
if __name__ == "__main__":
    print("股票图表可视化类 - 使用示例")
    print("=" * 50)
    
    # 示例1: 使用示例数据
    print("示例1: 使用随机生成的示例数据")
    sample_data = create_sample_stock_data(days=60)
    root = Tk()
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=True)
    # 创建图表对象
    plotter = PriceVolumePlotter('test', sample_data)
    fig, canvas = plotter.set_backend_window(frame)
    canvas.pack(fill=BOTH, expand=True)
    
    # 添加移动平均线
    plotter.add_moving_average(StockVisualData.AX_PRICE, window=5, color='red', label='MA_5')
    plotter.add_moving_average(StockVisualData.AX_PRICE, window=20, color='blue', label='MA_20')
    
    # 显示图表
    # plotter.show()
    root.mainloop()
    
    # 示例2: 从CSV文件加载数据
    # print("\n示例2: 从CSV文件加载数据")
    # csv_data = load_stock_data_from_csv('your_stock_data.csv')
    # plotter2 = StockChartPlotter(csv_data)
    # plotter2.set_custom_colors(price_up='green', price_down='red')
    # plotter2.show()
    
    # 保存图表
    # plotter.save('stock_chart.png', dpi=300)