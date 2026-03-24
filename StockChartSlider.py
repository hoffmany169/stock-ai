
import datetime
from tkinter import messagebox
from matplotlib import pyplot as plt
import pandas as pd
from StockChartPlotter import StockChartPlotter, StockVisualData
import matplotlib.dates as mdates

class StockChartSlider(StockChartPlotter):
    def __init__(self, stock_model, figsize=(14, 10)):
        super().__init__(stock_model, figsize)
        self._show_start_date = self.stock_model.start_date
        self._show_end_date = self.stock_model.end_date
        self._old_dates = self.dates_mpl.copy()

    @property
    def show_start_date(self):
        return self._show_start_date
    @show_start_date.setter
    def show_start_date(self, dt):
        min_date = datetime.date.fromisoformat(self.stock_model.start_date)
        new_date = datetime.date.fromisoformat(dt)
        if new_date < min_date:
            messagebox.showwarning("Warning", "Date is out of range")
            return
        self._show_start_date = dt

    @property
    def show_end_date(self):
        return self._show_end_date
    @show_end_date.setter
    def show_end_date(self, dt):
        max_date = datetime.date.fromisoformat(self.stock_model.end_date)
        new_date = datetime.date.fromisoformat(dt)
        if new_date > max_date:
            messagebox.showwarning("Warning", "Date is out of range")
            return
        self._show_end_date = dt

    def create_plot(self):
        # 绘制收盘价折线
        fig, ax = plt.subplots(figsize=self.figsize)
        if fig is None:
            raise ValueError("fig is None")
        self.visual_data.fig = fig
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.ax, ax, StockVisualData.AX_PRICE)
        # 计算涨跌颜色
        colors = self.calculate_price_change()
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.properties, colors, 'price_change', axes_name=StockVisualData.AX_PRICE)
        self.plot()      
        # use mplcursors to show points on the curve.  
        self.switch_mplcursors(ax, on=True)
        
        # 添加交互功能, not activate interactive features for volume subplot for now, as it may cause some performance issue and the interaction on price plot is more intuitive and useful
        # self.add_mouse_hover_event(ax)
        
        # 调整布局
        plt.tight_layout()

    def plot(self):
        super().plot()

    def update_to_date_range(self):
        start_num = mdates.date2num(pd.to_datetime(self.show_start_date))
        end_num = mdates.date2num(pd.to_datetime(self.show_end_date))
        
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax,
                                                    StockVisualData.AX_PRICE)
        ax.set_xlim(start_num, end_num)
        
        # 自动调整y轴
        mask = (self.stock_data['Date'] >= self.show_start_date) & (self.stock_data['Date'] <= self.show_end_date)
        if mask.any():
            y_min = self.stock_data.loc[mask, self.feature].min() * 0.95
            y_max = self.stock_data.loc[mask, self.feature].max() * 1.05
            ax.set_ylim(y_min, y_max)
        
        self.fig_canvas.draw_idle()

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
        
        # 重绘
        self.fig_canvas.draw_idle()

    def on_hover_info(self, *args):
        return super().on_hover_info(*args)
    
    def on_leave_info(self, *args):
        return super().on_leave_info(*args)