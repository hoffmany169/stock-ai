
from matplotlib import pyplot as plt
from StockChartPlotter import StockChartPlotter, StockVisualData


class StockChartSlider(StockChartPlotter):
    def __init__(self, stock_model, figsize=...):
        super().__init__(stock_model, figsize)
        self.show_start_date = ''
        self.show_end_date = ''

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

