
import datetime
from tkinter import font, messagebox, ttk, Button, BOTH, X, LEFT
from matplotlib import pyplot as plt
import pandas as pd
from StockChartPlotter import StockChartPlotter, StockVisualData
import matplotlib.dates as mdates
from Common.Util import AutoIndex
from plot_style import PLOT_ELEMENT, STYLE


class StockChartSlider(StockChartPlotter):
    class CONTROLS(AutoIndex):
        control_frame = ()
        slide_model_combo = ()
        slide_company_label = ()
        slide_feature_combo = ()
        control_frame_2 = ()
        start_date_entry = ()
        end_date_entry = ()
        show_slide_button = ()
        back_slide_button = ()
        slide_change_number_entry = ()
        interval_label = ()
        forward_slide_button = ()
        slide_figure_frame = ()

    def __init__(self):
        self.controls = {}
        super().__init__()
        # self._show_start_date = self.stock_model.start_date
        # self._show_end_date = self.stock_model.end_date
        # self._old_dates = self.dates_mpl.copy()

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

    def get_control(self, control):
        return self.controls.get(control, None)

    def create_gui(self):
        self.plot_frame = ttk.Frame()
        control_frame = ttk.Frame(self.plot_frame)
        self.controls[self.CONTROLS.control_frame] = control_frame
        self.get_control(self.CONTROLS.control_frame).pack(fill=X, padx=5, pady=5)
        ttk.Label(control_frame, text="Select Ticker:").pack(side=LEFT, padx=(20,5))
        self.controls[self.CONTROLS.slide_model_combo] = ttk.Combobox(control_frame, width=10)
        self.get_control(self.CONTROLS.slide_model_combo).pack(side=LEFT, padx=5)
        self.controls[self.CONTROLS.slide_company_label] = ttk.Label(control_frame, text="Company", relief="sunken", width=60)
        self.get_control(self.CONTROLS.slide_company_label).pack(side=LEFT, padx=5)
        ttk.Label(control_frame, text="Select to be shown Feature:").pack(side=LEFT, padx=(10,5))
        # 特征选择下拉框
        self.controls[self.CONTROLS.slide_feature_combo] = ttk.Combobox(control_frame, width=10)
        self.controls[self.CONTROLS.slide_feature_combo].pack(side=LEFT, padx=5)

        control_frame_2 = ttk.Frame(self.plot_frame)
        self.controls[self.CONTROLS.control_frame_2] = control_frame_2
        self.get_control(self.CONTROLS.control_frame_2).pack(fill=X, padx=5, pady=5)
        ttk.Label(control_frame_2, text='Start Date').pack(side=LEFT, padx=(20,5))
        self.controls[self.CONTROLS.start_date_entry] = ttk.Entry(control_frame_2, width=12, justify='center')
        self.get_control(self.CONTROLS.start_date_entry ).pack(side=LEFT, padx=5)
        ttk.Label(control_frame_2, text='End Date').pack(side=LEFT, padx=5)
        self.controls[self.CONTROLS.end_date_entry] = ttk.Entry(control_frame_2, width=12, justify='center')
        self.get_control(self.CONTROLS.end_date_entry).pack(side=LEFT, padx=5)
        self.controls[self.CONTROLS.show_slide_button] = Button(control_frame_2, text="Show Slide")
        self.get_control(self.CONTROLS.show_slide_button).pack(side=LEFT, padx=5)  
        from PIL import Image, ImageTk   
        # 1. Open the image file
        back_img = Image.open("rsc/backwords.jpg")  
        # 2. Resize it (optional but recommended for buttons)
        resized_back_img = back_img.resize((20, 20))
        # 3. Convert to a Tkinter-compatible photo object
        back_icon = ImageTk.PhotoImage(resized_back_img)
        self.controls[self.CONTROLS.back_slide_button] = Button(control_frame_2, image=back_icon, width=20)
        self.get_control(self.CONTROLS.back_slide_button).pack(side=LEFT, padx=(10, 5))
        self.controls[self.CONTROLS.slide_change_number_entry] = ttk.Entry(control_frame_2, width=3)
        self.get_control(self.CONTROLS.slide_change_number_entry).pack(side=LEFT, padx=5)
        self.controls[self.CONTROLS.interval_label] = ttk.Label(control_frame_2, width=6, justify='center')
        self.get_control(self.CONTROLS.interval_label).pack(side=LEFT, padx=5)
        # 1. Open the image file
        forward_img = Image.open("rsc/forwords.jpg")  
        # 2. Resize it (optional but recommended for buttons)
        resized_forward_img = forward_img.resize((20, 20))
        # 3. Convert to a Tkinter-compatible photo object
        forward_icon = ImageTk.PhotoImage(resized_forward_img)
        self.controls[self.CONTROLS.forward_slide_button] = Button(control_frame_2, image=forward_icon, width=20)
        self.get_control(self.CONTROLS.forward_slide_button).pack(side=LEFT, padx=(5, 20))
        # 图表显示区域
        self.controls[self.CONTROLS.slide_figure_frame] = ttk.Frame(self.plot_frame)
        self.get_control(self.CONTROLS.slide_figure_frame).pack(fill=BOTH, expand=True, padx=5, pady=5)
        return self.plot_frame
    
    def create_plot(self, stock_model=None, feature=None):
        super().create_plot(stock_model, feature)
        # 绘制收盘价折线
        fig, ax = plt.subplots(figsize=self.figsize)
        if fig is None:
            raise ValueError("fig is None")
        self.visual_data.fig = fig
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.ax, ax, StockVisualData.AX_PRICE)
        # 计算涨跌颜色
        self.plot()      
        
        # 添加交互功能, not activate interactive features for volume subplot for now, as it may cause some performance issue and the interaction on price plot is more intuitive and useful
        # self.add_mouse_hover_event(ax)
        
        # 调整布局
        plt.tight_layout()

    def plot(self):
        if self.stock_data is None:
            return
        self._show_start_date = self.stock_model.start_date
        self._show_end_date = self.stock_model.end_date
        self._old_dates = self.dates_mpl.copy()
        ax_price = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        # 绘制收盘价折线
        price_line, = ax_price.plot(
            self.dates_mpl, 
            self.stock_data[self.feature],
            color=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.price_line),
            linewidth=self.plot_styles.get_setting(STYLE.line_widths, PLOT_ELEMENT.price_line),
            label=f'{self.feature} Price',
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
        # use mplcursors to show points on the curve.  
        self.switch_mplcursors(ax_price, on=True)

    def update_to_date_range(self):
        start_num = mdates.date2num(pd.to_datetime(self.show_start_date))
        end_num = mdates.date2num(pd.to_datetime(self.show_end_date))
        
        ax_price = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax,
                                                    StockVisualData.AX_PRICE)        
        # 自动调整y轴
        mask = (self.stock_data['Date'] >= self.show_start_date) & (self.stock_data['Date'] <= self.show_end_date)
        if not mask.any():
            print("所选时间段无数据")
            messagebox.showinfo("Info", "No data in selected period")
            return     
        # 获取子集
        subset = self.stock_data[mask]
        new_dates = mdates.date2num(subset['Date'])
        new_feature_data = subset[self.feature].values
        # 更新价格线
        price_line = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists,
                                                            StockVisualData.AX_PRICE,
                                                            'price_line')
        if hasattr(self, 'price_line'):
            price_line.set_data(new_dates, new_feature_data)

        # 重新计算涨跌颜色
        # price_change = subset[self.feature] >= subset[self.feature].shift(1).fillna(subset[self.feature])

        # 更新坐标轴范围
        ax_price.set_xlim(new_dates[0], new_dates[-1])
        
        y_min = subset[self.feature].min() * 0.95
        y_max = subset[self.feature].max() * 1.05
        ax_price.set_ylim(y_min, y_max)

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
        # if shown data are only part of all data, this stock data are not identical with loaded_data
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
        # 设置股价图标题和标签
        ax_price.set_title(f'{self.symbol}: Stock Price Trend', 
                               fontsize=self.plot_styles.get_setting(STYLE.font_sizes, PLOT_ELEMENT.title),
                               fontweight='bold',
                               pad=20)
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

    def reset_view(self):
        """重置为完整数据视图"""
        if self.visual_data.fig is None:
            return
        
        # 原始 x 轴范围（全部日期）
        x_min = self.dates_mpl[0]
        x_max = self.dates_mpl[-1]
        
        # 原始 y 轴范围（股价整体范围，可稍留边距）
        y_min = self.stock_data[self.feature].min() * 0.95
        y_max = self.stock_data[self.feature].max() * 1.05
        
        # 应用到股价图
        ax_price = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax,
                                                          StockVisualData.AX_PRICE)
        ax_price.set_xlim(x_min, x_max)
        ax_price.set_ylim(y_min, y_max)
        
        # 如果存在交易量图且共享 x 轴，也同步
        # if hasattr(self, 'ax_volume'):
        #     self.ax_volume.set_xlim(x_min, x_max)
        
        # 重绘
        self.fig_canvas.draw_idle()

    def on_hover_info(self, *args):
        return super().on_hover_info(*args)
    
    def on_leave_info(self, *args):
        return super().on_leave_info(*args)
    
    def on_add(self, sel):
        from StockChartPlotter import PointData
        from StockModel import StockModel
        self.current_point = PointData(sel)
        idx = int(sel.index)
        x, y = sel.target
        info_text = ''
        # 1. value of this point
        info_text += f'{str(self.current_point)}\n'
        # 2. volume value
        Volume = self.stock_model.get_feature_value(StockModel.FEATURE.Volume, idx)
        info_text += f'Volume: {self.format_large_numbers(Volume, 0)}\n'
        # change of volume
        changed_volume = self.stock_model.get_ext_feature(StockModel.ExtendFeature.volume_change, idx) 
        sym = "$\\uparrow$" if changed_volume > 0 else "$\\downarrow$"
        info_text += f'{sym}{self.format_large_numbers(abs(changed_volume), 0)}\n'
        # percentage of volume relative with maximum volume 
        max_volume = self.stock_model.get_ext_feature(StockModel.ExtendFeature.max_volume)
        vol_perc = Volume / max_volume * 100 if max_volume is not None and max_volume != 0 else 0
        info_text += f'{vol_perc:.1f}%\n'
        # price position between low-high price range
        range = self.stock_model.get_ext_feature(StockModel.ExtendFeature.high_low_range, idx)
        low = self.stock_model.get_feature_value(StockModel.FEATURE.Low, idx)
        price_perc = (y - low) / range * 100 if range is not None and range != 0 else 0
        info_text += f'price / range: {price_perc:.1f}'
        # define annotation and show it.
        sel.annotation.set_text(f'{info_text}' )
        sel.annotation.set(ha='left')
        sel.annotation.get_bbox_patch().set_alpha(0.9)
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        # show a small red point
        highlight = ax.plot(x, y, color='red', marker='o', markersize=5)
        # add it to extras so it is removed on deselection
        sel.extras.append(highlight[0])

