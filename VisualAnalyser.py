import sys
from tkinter import StringVar, filedialog, messagebox, IntVar
import matplotlib
from AxisRatioCalculator import AxisRatioCalculator
from StockChartPlotter import PointData
from PriceVolumePlotter import PriceVolumePlotter, StockVisualData
matplotlib.use('TkAgg')  # Use Tk backend
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from Common.AutoNumber import AutoIndex
import mplcursors
from Common.Util import CreateChildWindow

class ElementLayer(AutoIndex):
    MARKER = ()
    ANNOTATION = ()
    GUIDELINE = ()
    ORIGINAL = ()

class CONTEXT_COMMAND(AutoIndex):
    #   zoom_in = ()
    #   zoom_out = ()
    #   seperator_1 = ()
    set_first_point = ()
    compare_point = ()
    seperator_2 = ()
    wave_begin = ()
    wave_end = ()
    seperator_3 = ()
    draw_horizontal_line = ()
    remove_horizontal_line = ()
    seperator_4 = ()
    first_trend_point = ()
    draw_trend_line = ()
    remove_trend_line = ()
    


class FILE_MENU_COMMAND(AutoIndex):
      open_file = ()
      save_image = ()
      save_image_as = ()
      export_pdf = ()
      seperator_1 = ()
      references = ()
      seperator_2 = ()
      exit = ()

class ACTION_MENU_COMMAND(AutoIndex):
      highlight_peaks_and_valleys = ()
      remove_peaks_and_valleys = ()
#       show_comparison_2_points = ()
#       draw_line = ()
#       remove_point = ()
#       remove_line = ()
#       clear_all_markers = ()
#       clear_all_lines = ()
#       reset_view = ()

class MARKER_STYLE(AutoIndex):
    red_circle = ()
    green_cross = ()
    blue_triangle = ()

class LINE_STYLE(AutoIndex):
    solid_line = ()
    dashed_line = ()
    dash_dot_line = ()
    dotted_line = ()

class PROPERTY_2_POINTS(AutoIndex):
    distance = ()
    value_difference = ()
    percentage_of_value_change = ()
    tangent_of_line = ()
    
class VisualAnalyser(PriceVolumePlotter):
    CONTEXT_MENU_TEXT = ['label', 'command']
    def __init__(self, stock_model, figsize=(14,8)):
        """
        Docstring for __init__
        two scenarios:
        1. pass in fig and ax from outside
        :param fig: figure object in plotting
        :param ax: axis object in plotting
        2. create fig and ax inside
        :param figsize: figure size
        """
        super().__init__(stock_model, figsize=figsize)
        self.axis_ratio_calculator = AxisRatioCalculator(self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)  )
        # if activate real-time search
        self.current_point = None
        # first point for comparing points
        self.first_point = None
        # first point for trend line
        self.first_trend_point = None

    def set_backend_window(self, parent):
        super().set_backend_window(parent)
        self._init_plot_window_()
        return (self.visual_data.fig, self.tk_root) # root is canvas of parent figure

    def _init_plot_window_(self):
        self._create_menu_bar()
        self._create_context_menu_commands()
                
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
        self.curve_cursor = mplcursors.cursor(self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, 
                                                                                     StockVisualData.AX_PRICE, 
                                                                                     data_name='price_line'),
                                              hover=2) # Transient hover mode: the annotation appears when the mouse is near a point and disappears when the mouse moves away, with a delay of 2 seconds before disappearing. This mode is useful for providing information about points without requiring a click, while also ensuring that the annotation does not linger on the screen for too long.
        self.curve_cursor.connect("add", self.on_add)
        # disable remove event to prevent right-click conflict with hover event, as they may trigger at the same time when user right-clicks on a point, which can cause the context menu to not show up
        self.curve_cursor.connect("remove", None)
        # 配置图表格式
        self.format_chart_price(ax)
        
        # 添加交互功能, not activate interactive features for volume subplot for now, as it may cause some performance issue and the interaction on price plot is more intuitive and useful
        # self.add_mouse_hover_event(ax)
        
        # 调整布局
        plt.tight_layout()

    # def update_chart(self, new_stock_model):
    #     """
    #     清空当前图形并绘制新数据
    #     """
    #     # 清空整个坐标轴
    #     self.clear_all_plot_data()
        
    #     # 更新数据
    #     self.stock_data = new_data
    #     self.dates_mpl = mdates.date2num(self.stock_data['Date'])
        
    #     # 重新绘制
    #     self.plot_price_chart()
        
    #     # 重新格式化
    #     self.format_chart()
        
    #     # 重绘
    #     self.fig.canvas.draw_idle()

    def plot(self):
        super().plot()
        # ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        # self.plot_price_chart(ax. self._feature)

    def _create_menu_bar(self):
        self.menubar = tk.Menu(self.parent)
        self.parent.config(menu=self.menubar)
        # File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='File', menu=self.file_menu)
        self._add_command_to_menu(self.file_menu, FILE_MENU_COMMAND)
        # Action menu
        self.action_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='Action', menu=self.action_menu)
        self._add_command_to_menu(self.action_menu, ACTION_MENU_COMMAND)

    def _add_command_to_menu(self, menu, commands):
        for e in commands:
            label_text = ' '.join(e.name.split('_'))
            if e.name.startswith('seperator'):
                menu.add_separator()
            else:
                cmd = getattr(self, f'on_{e.name}')
                menu.add_command(label=label_text, 
                                command=cmd)

    def _create_context_menu_commands(self):
        self.context_menu = tk.Menu(self.tk_root, tearoff=0)
        self.menu_items = {}
        for e in CONTEXT_COMMAND:
            if e.name.startswith('seperator'):
                self.context_menu.add_separator()
            else:
                self.context_menu.add_command(label=' '.join(e.name.split('_')), command=self.dummy_command)
                self.menu_items[e.name] = self.context_menu.index("end")

    # def fill_between(self, alpha=0.2):
    #     """Fill area under the curve with specified alpha transparency."""
    #     self.ax.fill_between(self.x, self.y, alpha=alpha)
    #     self.fig_canvas.draw()

    def set_labels(self, xlabel='', ylabel=''):
        """Set x and y axis labels."""
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.fig_canvas.draw()

    def set_legend(self, location='upper right', shadow=True, fsize='x-small'):
        """Set legend on the plot."""
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        ax.legend(loc=location, shadow=shadow, fontsize=fsize)
        self.fig_canvas.draw()
        
#region ### event callback functions    
    def on_right_click(self, event):
        if event.button == 3 and event.inaxes == self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, axes_name='ax_price'):  # Right-click in axes
            self.last_click_coords = (event.xdata, event.ydata)
            self.visual_data.fig._last_right_click = (event.xdata, event.ydata, event.inaxes)
            for key, index in self.menu_items.items():
                # trick here: set key as a default argument to prevent of late binding of variables in lambdas inside a loop!!!
                self.context_menu.entryconfig(index, command=lambda item=key: getattr(self, f'on_{item}')(self.last_click_coords))
            try:
                x_tk = self.tk_root.winfo_pointerx()
                y_tk = self.tk_root.winfo_pointery()
                self.context_menu.tk_popup(x_tk, y_tk)
            except:
                # Fallback
                self.context_menu.post(self.last_click_coords)
            
    def on_add(self, sel):
        from StockModel import StockModel
        self.current_point = PointData(sel)
        idx = int(sel.index)
        # Volume = self.stock_data['Volume'].fillna(0)  # Ensure Volume column has no NaN values
        Volume = self.stock_model.get_feature_value(StockModel.FEATURE.Volume, idx)
        max_volume = self.stock_model.get_ext_feature(StockModel.ExtendFeature.max_volume)
        vol_perc = Volume / max_volume * 100 if max_volume is None or max_volume > 0 else 0
        range = self.stock_model.get_ext_feature(StockModel.ExtendFeature.high_low_range, idx)
        # calculate changed volume
        changed_volume = self.stock_model.get_ext_feature(StockModel.ExtendFeature.volume_change, idx) 
        sym = '↑' if changed_volume > 0 else '↓'
        if changed_volume < 0:
            changed_volume = changed_volume * (-1) 
        sel.annotation.set(ha='left', text=f"{str(self.current_point)}\nVolume:\n ° {self.format_large_numbers(Volume, 0)}\n ° {vol_perc:.1f}%\n ° {sym}{self.format_large_numbers(changed_volume, 0)}\nHL Range: {range:.0f}")
        sel.annotation.get_bbox_patch().set_alpha(0.9)
        # draw selected point marker
        x, y = sel.target
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        highlight = ax.plot(x, y, color='red', marker='o', markersize=5)
        # add it to extras so it is removed on deselection
        sel.extras.append(highlight[0])

    def on_zoom_in(self, coords):
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0]*0.8, xlim[1]*0.8)
        ax.set_ylim(ylim[0]*0.8, ylim[1]*0.8)
        self.fig_canvas.draw()
    
    def on_zoom_out(self, coords):
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0]*1.2, xlim[1]*1.2)
        ax.set_ylim(ylim[0]*1.2, ylim[1]*1.2)
        self.fig_canvas.draw()
    
    # set selected first point for comparison
    def on_set_first_point(self, *args):
        self.first_point = self.current_point

    # compare current point with the first point and show the result in a dialog
    def on_compare_point(self, *args):
        if self.first_point is None:
            messagebox.showinfo("Info", "Please select the first point before comparing.")
            return
        else:
            result = self.first_point.compare_with(self.current_point, to_string=True)
            from Common.Util import CreateChildWindow
            def create_comparison_dialog():
                top = CreateChildWindow(self.tk_root, "Comparison Result", "300x200", XClose=True)
                tk.Label(top, text=result,justify=tk.LEFT).pack(pady=20, padx=20)
            create_comparison_dialog()

    def on_wave_begin(self, *args):
        self.first_wave_point = self.current_point

    def on_wave_end(self, *args):
        if self.first_wave_point is None:
            messagebox.showinfo("Info", "Please select the wave start-point before comparing.")
            return
        else:
            x0 = self.first_wave_point
            x1 = self.current_point
            pk_color = 'limegreen'
            pk_marker = self.plot_styles.get_marker('plus')
            vl_color = 'firebrick'
            vl_marker = self.plot_styles.get_marker('cross')
            from Common.Util import CreateChildWindow, CloseChildWindow
            def create_wave_dialog():
                top = CreateChildWindow(self.tk_root, "Wave Info", "300x300")
                art_names = ('wave_peaks', 'wave_valleys')
                frm1 = tk.Frame(top)
                frm1.pack(fill='x', expand=True, pady=(0, 5))
                tk.Label(frm1, text="Window").pack(side='left', padx=5)
                winvar = IntVar(top, 1)
                window = tk.Entry(frm1, textvariable=winvar)
                window.pack(side='left', padx=5)
                result_var = StringVar(top, "")
                frm2 = tk.Frame(top)
                frm2.pack(fill='x', expand=True, pady=5)
                tk.Button(frm2, text="Show Info", command=lambda sel1=x0, sel2=x1, win=winvar.get(): extract_wave_info(sel1, sel2, win)).pack(side='left', padx=(10, 5))
                tk.Button(frm2, text="Close", command=lambda: close_info()).pack(side='left', padx=(5, 10))
                tk.Label(top, textvariable=result_var, justify='left').pack(fill='x', expand=True, pady=5)

                def close_info():
                    self.remove_highlighted_peaks_valleys(StockVisualData.AX_PRICE, artist_names=art_names)
                    self.first_wave_point = None
                    CloseChildWindow(top)

                def extract_wave_info(sel1, sel2, win):
                    self.remove_highlighted_peaks_valleys(StockVisualData.AX_PRICE, artist_names=art_names)
                    peaks, valleys = self.highlight_peaks_valleys(StockVisualData.AX_PRICE,
                                                                    self.feature, 
                                                                    window=winvar.get(),
                                                                    peak_color=pk_color,
                                                                    peak_marker=pk_marker,
                                                                    valley_color=vl_color,
                                                                    valley_marker=vl_marker,
                                                                    artist_names=art_names,
                                                                    start_date=sel1.Date,
                                                                    end_date=sel2.Date)
                    points = peaks + valleys
                    points.sort()
                    values = self.stock_data[self.feature].iloc[points]
                    # dates = self.stock_data['Date'].iloc[points]
                    last_value = None
                    last_diff = None
                    value_percent = []
                    # calculate percentage of value changes
                    for cur_value in values:
                        if last_value is None:
                            last_value = cur_value
                            continue
                        else:
                            cur_diff = cur_value - last_value
                            last_value = cur_value
                            if last_diff is None:
                                last_diff = cur_diff
                            else:
                                value_percent.append((cur_diff / last_diff)*(-100 if cur_diff > 0 else 100))
                                last_diff = cur_diff
                    # create string
                    result = ""
                    if len(value_percent) > 0:
                        for i, v in enumerate(value_percent):
                            result += f'{i+1}.[{v:.1f}%]\n'
                    result_var.set(result)
                    top.update()
            create_wave_dialog()

    def on_draw_horizontal_line(self, *args):
        self.remove_artist(StockVisualData.AX_PRICE, 'horizontal_line')
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        x, y = self.current_point.Coordinate
        artist1 = ax.axhline(y=y, color='purple', linestyle=self.plot_styles.get_line_style('solid_line'), alpha=0.5, linewidth=1)
        artist2 = ax.text(0.5, y, f'y = {y:.2f}', 
                    transform=ax.get_yaxis_transform(),
                    color='purple', ha='center', va='bottom')
        self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, (artist1, artist2), 'horizontal_line', StockVisualData.AX_PRICE)
        self.fig_canvas.draw_idle()

    def on_remove_horizontal_line(self, *args):
        self.remove_artist(StockVisualData.AX_PRICE, 'horizontal_line')

    def on_first_trend_point(self, *args):
        self.first_trend_point = self.current_point

    def on_draw_trend_line(self, *args):
        """
        Docstring for draw_line
        Visual angle = arctan(Data slope × (y_scale / x_scale))
             = arctan(Data slope / Aspect_ratio)
        """
        from plot_style import STYLE, PLOT_ELEMENT
        if self.first_trend_point is None:
            messagebox.showwarning("Trend Line", "Please, add starting trend point at first")
            return
        coord1 = self.first_trend_point.Coordinate
        coord2 = self.current_point.Coordinate
        if coord1 == coord2:
            messagebox.showwarning("Warning", "Start and End points must be different.")
        else:
            self.remove_artist(StockVisualData.AX_PRICE, 'trend_line')
            ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
            if ax is None:
                messagebox.showerror("Plot Trend Line", "Invalid ax")
                return
            line = ax.axline(coord1, coord2, 
                             linestyle=self.plot_styles.get_line_style('solid_line'),
                             alpha = self.plot_styles.get_setting(STYLE.alphas, PLOT_ELEMENT.trend_line),
                             linewidth=self.plot_styles.get_setting(STYLE.line_widths, PLOT_ELEMENT.trend_line),
                             color=self.plot_styles.get_setting(STYLE.colors, PLOT_ELEMENT.price_up if coord2[1] > coord1[1] else PLOT_ELEMENT.price_down))
            
            self.fig_canvas.draw_idle()
            self.visual_data.add_stock_visual_data(StockVisualData.TYPE.artists, line, 'trend_line', StockVisualData.AX_PRICE)

    def on_remove_trend_line(self, *args):
        self.remove_artist(StockVisualData.AX_PRICE, 'trend_line')

    def draw_vertical_line(self, coords):
        if coords:
            ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
            x, y = coords
            # Draw vertical line at x position
            artist1 = ax.axvline(x=x, color='g', linestyle=self.line_style, alpha=0.5)
            # Add text annotation
            artist2 = ax.text(x, ax.get_ylim()[1]*0.9, f'x={x:.2f}', 
                        rotation=90, verticalalignment='top')
            self.fig_canvas.draw()
            # self.layers[ElementLayer.GUIDELINE].append((artist1, artist2))
    
    def on_highlight_peaks_and_valleys(self):
        from StockModel import StockModel
        from plot_style import STYLE, PLOT_ELEMENT
        textvar = []
        # Create a highlight peaks and valleys dialog
        dialog = CreateChildWindow(self.tk_root, 'highlight peaks and valleys',
                                   modal=False,
                                   geometry="300x200",
                                   XClose=True)    
        frm1 = tk.Frame(dialog)
        frm1.pack(fill='x', expand=True, pady=5)    
        tk.Label(frm1, text="ax name:", font=('Arial', 12, 'bold')).pack(side='left', padx=5)
        ax_names = list(self.visual_data.plot_data.keys())
        textvar.append(StringVar(dialog, ax_names[0]))
        ax_name_combo = ttk.Combobox(frm1, state='onlyread', textvariable=textvar[0], values=ax_names)
        ax_name_combo.pack(side='left', padx=5)

        frm2 = tk.Frame(dialog)
        frm2.pack(fill='x', expand=True, pady=(0, 5))
        tk.Label(frm2, text="Feature").pack(side='left', padx=5)
        tk.Label(frm2, text=self.feature).pack(side='left', padx=5)
        
        frm3 = tk.Frame(dialog)
        frm3.pack(fill='x', expand=True, pady=(0, 5))
        tk.Label(frm3, text="Window").pack(side='left', padx=5)
        textvar.append(IntVar(dialog, 5))
        window = tk.Entry(frm3, textvariable=textvar[1])
        window.pack(side='left', padx=5)

        frm4 = tk.Frame(dialog)
        frm4.pack(fill='x', expand=True, pady=(0, 5))
        tk.Label(frm4, text="peak color").pack(side='left', padx=5)
        textvar.append(tk.StringVar(dialog, 'green'))
        peak_color = tk.Entry(frm4, textvariable=textvar[2])
        peak_color.pack(side='left', padx=5)
        tk.Label(frm4, text="peak marker").pack(side='left', padx=5)
        marker_list = self.plot_styles.get_marker_names()
        textvar.append(tk.StringVar(frm4, marker_list[3]))
        peak_marker = ttk.Combobox(frm4, textvariable=textvar[3], values=marker_list)
        peak_marker.pack(side='left', padx=5)

        frm5 = tk.Frame(dialog)
        frm5.pack(fill='x', expand=True, pady=(0, 5))
        tk.Label(frm5, text="valley color").pack(side='left', pady=(0, 5))
        textvar.append(tk.StringVar(dialog, 'red'))
        valley_color = tk.Entry(frm5, textvariable=textvar[4])
        valley_color.pack(side='left', padx=5)
        tk.Label(frm5, text="valley marker").pack(side='left', padx=5)
        textvar.append(tk.StringVar(frm5, marker_list[4]))
        valley_marker = ttk.Combobox(frm4, textvariable=textvar[5], values=marker_list)
        valley_marker.pack(side='left', padx=5)
        
        frm6 = tk.Frame(dialog)
        frm6.pack(fill='x', expand=True, pady=(0, 5))
        tk.Button(frm6, text="OK", command=lambda x=textvar: self.show_peaks_valleys(x)).pack(anchor='center', pady=10)

    def show_peaks_valleys(self, params):
        self.remove_highlighted_peaks_valleys()
        self.highlight_peaks_valleys(params[0].get(),
                                    self.feature, 
                                    window=params[1].get(),
                                    peak_color=params[2].get(),
                                    peak_marker=self.plot_styles.get_marker(params[3].get()),
                                    valley_color=params[4].get(),
                                    valley_marker=self.plot_styles.get_marker(params[5].get()))


    def on_remove_peaks_and_valleys(self):
        self.remove_highlighted_peaks_valleys()

    def on_open_file(self):
        pass

    def on_save_image(self):
        if self._plot_file_name:
            self.fig.savefig(self._plot_file_name, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Save", f"Plot saved to:\n{self._plot_file_name}")
        else:
            self.save_image_as()

    def on_save_image_as(self):
        self._plot_file_name = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        if self._plot_file_name:
            self.fig.savefig(self._plot_file_name, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Save as", f"Plot saved to:\n{self._plot_file_name}")

    def on_export_pdf(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if filename:
            self.fig.savefig(filename, format='pdf', bbox_inches='tight')
            messagebox.showinfo("Export PDF", f"Plot saved to:\n{filename}")
    
    def on_exit(self):
        sys.exit(0)

    def on_references(self):
        messagebox.showinfo("References", "Plot Analyser\nVersion 1.0")

    # def on_closing(self):
    #     if messagebox.askokcancel("Quit", "Do you want to quit?"):
    #         plt.close(self.fig)
    #         self.root.quit()
    #         self.root.destroy()
#endregion event

import numpy as np
def main(type):
    if type == 0: ## without fig
        # 2- create plot in PlotAnalyser class
        data_x = np.linspace(0, 10, 100)
        data_y = np.cos(data_x) * np.exp(-0.05*data_x)
        dynamic_menu = VisualAnalyser('test', data=(data_x, data_y), figsize=(10,6))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(0, 4*np.pi, 200)
        y = np.sin(x) * np.exp(-0.1*x)
        # ax.legend()
        # ax.grid(True, alpha=0.3)

        if type == 1: ## with fig
            # Create dynamic context menu
            dynamic_menu = VisualAnalyser('test', data=(x, y), fig=fig)
        else: ## with ax
            dynamic_menu = VisualAnalyser('test', data=(x, y), ax=ax)
    dynamic_menu.fill_between(alpha=0.2)
    dynamic_menu.set_labels(xlabel='X-axis', ylabel='Y-axis')
    dynamic_menu.set_legend()
    dynamic_menu.show_plot()

if __name__ == '__main__':
    main(2)