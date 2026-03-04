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
      zoom_in = ()
      zoom_out = ()
      seperator_1 = ()
      set_first_point = ()
      compare_point = ()
      seperator_2 = ()
      wave_begin = ()
      wave_middle = ()
      wave_end_compare = ()


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
        self._marker_style = MARKER_STYLE.red_circle
        self._line_style = LINE_STYLE.dashed_line
        self.axis_ratio_calculator = AxisRatioCalculator(self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)  )
        # if activate real-time search
        self._real_time_search = True # default is True
        self.current_point = None
        self.first_point = None

    def set_backend_window(self, parent):
        super().set_backend_window(parent)
        self._init_plot_window_()
        return (self.visual_data.fig, self.tk_root) # root is canvas of parent figure

    def _init_plot_window_(self):
        self._create_menu_bar()
        self._create_context_menu_commands()
                
#region # properties
    @property
    def real_time_search(self):
        return self._real_time_search
    @real_time_search.setter
    def real_time_search(self, value):
        if type(value) == bool:
            self._real_time_search = value

    @property
    def marker_style(self):
        if self._marker_style == MARKER_STYLE.green_cross:
            return 'g+'
        elif self._marker_style == MARKER_STYLE.blue_triangle:
            return 'b^'
        else:
            return 'ro' # default

    @marker_style.setter
    def marker_style(self, marker):
        if type(marker) == MARKER_STYLE:
            self._marker_style = marker

    @property
    def line_style(self):
        if self._line_style == LINE_STYLE.solid_line:
            return '-'        #  solid line style
        elif self._line_style == LINE_STYLE.dash_dot_line:
            return '-.'       #  dash-dot line style
        elif self._line_style == LINE_STYLE.dotted_line:
            return ':'        #  dotted line style
        else:
            return '--'       #  default: dashed line style
    
    @line_style.setter
    def line_style(self, line):
        if type(line) == LINE_STYLE:
            self._line_style = line
#endregion # properties

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
        self.plot_price_chart(ax)
        self.price_line_cursor = mplcursors.cursor(self.visual_data.get_stock_visual_data(StockVisualData.TYPE.artists, 
                                                                                          StockVisualData.AX_PRICE, 
                                                                                          data_name='price_line'),
                                                   hover=2) # Transient hover mode: the annotation appears when the mouse is near a point and disappears when the mouse moves away, with a delay of 2 seconds before disappearing. This mode is useful for providing information about points without requiring a click, while also ensuring that the annotation does not linger on the screen for too long.
        self.price_line_cursor.connect("add", self.on_add)
        # disable remove event to prevent right-click conflict with hover event, as they may trigger at the same time when user right-clicks on a point, which can cause the context menu to not show up
        self.price_line_cursor.connect("remove", None)
        # 配置图表格式
        self.format_chart_price(ax)
        
        # 添加交互功能, not activate interactive features for volume subplot for now, as it may cause some performance issue and the interaction on price plot is more intuitive and useful
        # self.add_mouse_hover_event(ax)
        
        # 调整布局
        plt.tight_layout()

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
                self.context_menu.entryconfig(index, command=lambda item=key: getattr(self, item)(self.last_click_coords))
            try:
                x_tk = self.tk_root.winfo_pointerx()
                y_tk = self.tk_root.winfo_pointery()
                self.context_menu.tk_popup(x_tk, y_tk)
            except:
                # Fallback
                self.context_menu.post(self.last_click_coords)
            
    def dummy_command(self):
        """Placeholder - will be replaced"""
        print("This dummy_command")

    def on_add(self, sel):
        from StockModel import StockModel
        self.current_point = PointData(sel)
        idx = int(sel.index)
        # Volume = self.stock_data['Volume'].fillna(0)  # Ensure Volume column has no NaN values
        Volume = self.stock_model.get_feature_value(StockModel.FEATURE.Volume, idx)
        max_volume = self.stock_model.get_ext_feature(StockModel.ExtendFeature.max_volume)
        vol_perc = Volume / max_volume * 100 if max_volume is None or max_volume > 0 else 0
        range = self.stock_model.get_ext_feature(StockModel.ExtendFeature.high_low_range, idx)
        # print(f"Selected point: x={sel.target[0]:.2f}, y={sel.target[1]:.2f}, Volume={Volume.iloc[int(sel.target.index)]:.0f}")
        # sel.annotation.set(ha='left', text=f"Date: {mdates.num2date(sel.target[0]).strftime('%Y-%m-%d')}\nPrice: {sel.target[1]:.2f}\nVolume: {Volume.iloc[int(sel.index)]:.0f}")
        sel.annotation.set(ha='left', text=f"{str(self.current_point)}\nVolume: {self.format_large_numbers(Volume, 0)}({vol_perc:.1f}%)\nHL Range: {range:.0f}")
        sel.annotation.get_bbox_patch().set_alpha(0.9)
        # draw selected point marker
        x, y = sel.target
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        highlight = ax.plot(x, y, color='red', marker='o', markersize=5)
        # add it to extras so it is removed on deselection
        sel.extras.append(highlight[0])

    # set selected first point for comparison
    def set_first_point(self, *args):
        self.first_point = self.current_point

    # compare current point with the first point and show the result in a dialog
    def compare_point(self, *args):
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

    def draw_horizontal_line(self, coords):
        if coords:
            ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
            x, y = coords
            artist1 = ax.axhline(y=y, color='purple', linestyle=self.line_style, alpha=0.7, linewidth=2)
            artist2 = ax.text(0.5, y, f'y = {y:.2f}', 
                        transform=ax.get_yaxis_transform(),
                        color='purple', ha='center', va='bottom')
            self.fig_canvas.draw()
            self.layers[ElementLayer.GUIDELINE].append((artist1, artist2))
        
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
    
    def zoom_in(self, coords):
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0]*0.8, xlim[1]*0.8)
        ax.set_ylim(ylim[0]*0.8, ylim[1]*0.8)
        self.fig_canvas.draw()
    
    def zoom_out(self, coords):
        ax = self.visual_data.get_stock_visual_data(StockVisualData.TYPE.ax, StockVisualData.AX_PRICE)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0]*1.2, xlim[1]*1.2)
        ax.set_ylim(ylim[0]*1.2, ylim[1]*1.2)
        self.fig_canvas.draw()
    
    def draw_line(self):
        """
        Docstring for draw_line
        Visual angle = arctan(Data slope × (y_scale / x_scale))
             = arctan(Data slope / Aspect_ratio)
        """
        if len(self.layers[ElementLayer.MARKER]) >= 2:
            point_num = len(self.layers[ElementLayer.MARKER])
            first_points_idx = [i for i in range(1, point_num+1)]
            second_points_idx = [i for i in range(1, point_num+1)]
            def create_select_start_end_point_dialog():
                dialog = tk.Toplevel(self.tk_root)
                dialog.title("Select Start and End Points")
                dialog.geometry("300x200")
                
                tk.Label(dialog, text="Select Start Point:").pack()
                start_var = StringVar(value=str(first_points_idx[0]))
                start_combo = ttk.Combobox(dialog, values=first_points_idx, state='readonly', textvariable=start_var)
                start_combo.pack(pady=5)
                
                tk.Label(dialog, text="Select End Point:").pack()
                end_var = StringVar(value=str(second_points_idx[0]))
                end_combo = ttk.Combobox(dialog, values=second_points_idx, state='readonly', textvariable=end_var)
                end_combo.pack(pady=5)
                
                def on_confirm():
                    start_idx = int(start_var.get()) - 1
                    end_idx = int(end_var.get()) - 1
                    if start_idx != end_idx:
                        self._draw_line_between_points(start_idx, end_idx)
                        dialog.destroy()
                    else:
                        messagebox.showwarning("Warning", "Start and End points must be different.")
                
                tk.Button(dialog, text="Confirm", command=on_confirm).pack(pady=10)
            create_select_start_end_point_dialog()

    def _draw_line_between_points(self, start_idx, end_idx):
            if start_idx == end_idx:
                messagebox.showwarning("Warning", "Start and End points must be different.")
                return
            x1 = float(self.layers[ElementLayer.MARKER][start_idx][0].get_xdata()[0])
            y1 = float(self.layers[ElementLayer.MARKER][start_idx][0].get_ydata()[0])
            x2 = float(self.layers[ElementLayer.MARKER][end_idx][0].get_xdata()[0])
            y2 = float(self.layers[ElementLayer.MARKER][end_idx][0].get_ydata()[0])
            artist1 = self.ax.plot([x1, x2], [y1, y2], self.line_style, color='b')
            visual_angle = self.axis_ratio_calculator.get_visual_angle(x1, y1, x2, y2)
            text_number = len(self.layers[ElementLayer.GUIDELINE]) + 1
            artist2 = self.ax.text((x1+x2)/2, (y1+y2)/2, f'L{text_number}', 
                        rotation=visual_angle, verticalalignment='top')
            self.fig_canvas.draw()
            self.points_to_line[(start_idx, end_idx)] = len(self.layers[ElementLayer.GUIDELINE])
            self.layers[ElementLayer.GUIDELINE].append((artist1, artist2, text_number))

    def update_lines(self):
        for line, text, text_number in self.layers[ElementLayer.GUIDELINE]:
            xdata = line[0].get_xdata()
            ydata = line[0].get_ydata()
            x1, x2 = xdata[0], xdata[1]
            y1, y2 = ydata[0], ydata[1]
            visual_angle = self.axis_ratio_calculator.get_visual_angle(x1, y1, x2, y2)
            text.set_position(((x1+x2)/2, (y1+y2)/2))
            text.set_rotation(visual_angle)
            text.set_text(f'L{text_number}')
        self.fig_canvas.draw()

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
        ax_names = list(self.visual_data.data.keys())
        textvar.append(StringVar(dialog, ax_names[0]))
        ax_name_combo = ttk.Combobox(frm1, state='onlyread', textvariable=textvar[0], values=ax_names)
        ax_name_combo.pack(side='left', padx=5)

        frm2 = tk.Frame(dialog)
        frm2.pack(fill='x', expand=True, pady=(0, 5))
        tk.Label(frm2, text="Feature").pack(side='left', padx=5)
        features = [f.name for f in StockModel.FEATURE]
        textvar.append(StringVar(dialog, features[0]))
        feature_combo = ttk.Combobox(frm2, textvariable=textvar[1], values=features)
        feature_combo.pack(side='left', padx=5)
        
        frm3 = tk.Frame(dialog)
        frm3.pack(fill='x', expand=True, pady=(0, 5))
        tk.Label(frm3, text="Window").pack(side='left', padx=5)
        textvar.append(IntVar(dialog, 5))
        window = tk.Entry(frm3, textvariable=textvar[2])
        window.pack(side='left', padx=5)

        frm4 = tk.Frame(dialog)
        frm4.pack(fill='x', expand=True, pady=(0, 5))
        tk.Label(frm4, text="peak color").pack(side='left', padx=5)
        textvar.append(tk.StringVar(dialog, 'green'))
        peak_color = tk.Entry(frm4, textvariable=textvar[3])
        peak_color.pack(side='left', padx=5)
        tk.Label(frm4, text="peak marker").pack(side='left', padx=5)
        marker_list = self.plot_styles.get_marker_names()
        textvar.append(tk.StringVar(frm4, marker_list[3]))
        peak_marker = ttk.Combobox(frm4, textvariable=textvar[4], values=marker_list)
        peak_marker.pack(side='left', padx=5)

        frm5 = tk.Frame(dialog)
        frm5.pack(fill='x', expand=True, pady=(0, 5))
        tk.Label(frm5, text="valley color").pack(side='left', pady=(0, 5))
        textvar.append(tk.StringVar(dialog, 'red'))
        valley_color = tk.Entry(frm5, textvariable=textvar[5])
        valley_color.pack(side='left', padx=5)
        tk.Label(frm5, text="valley marker").pack(side='left', padx=5)
        textvar.append(tk.StringVar(frm5, marker_list[4]))
        valley_marker = ttk.Combobox(frm4, textvariable=textvar[6], values=marker_list)
        valley_marker.pack(side='left', padx=5)
        
        frm6 = tk.Frame(dialog)
        frm6.pack(fill='x', expand=True, pady=(0, 5))
        tk.Button(frm6, text="OK", command=lambda x=textvar: self.highlight_peaks_valleys(x[0].get(),
                                                                                          x[1].get(), 
                                                                                          window=x[2].get(),
                                                                                          peak_color=x[3].get(),
                                                                                          peak_marker=self.plot_styles.get_marker(x[4].get()),
                                                                                          valley_color=x[5].get(),
                                                                                          valley_marker=self.plot_styles.get_marker(x[6].get()))).pack(anchor='center', pady=10)

    def highlight_peaks_valleys(self, ax_name:str, feature:str,
                                window=5, 
                                peak_color='green', 
                                valley_color='red',
                                peak_marker='v',
                                valley_marker='^',
                                peak_label='Local Highest Point',
                                valley_label='Local Lowest Point'
                                ):
        self.remove_highlighted_peaks_valleys()
        super().highlight_peaks_valleys(ax_name, feature, window=window, peak_color=peak_color, valley_color=valley_color,
                                        peak_marker=peak_marker,
                                        valley_marker=valley_marker,
                                        peak_label=peak_label,
                                        valley_label=valley_label)

    def on_remove_peaks_and_valleys(self):
        self.remove_highlighted_peaks_valleys()

    def remove_highlighted_peaks_valleys(self):
        super().remove_highlighted_peaks_valleys()

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

    def remove_point(self):
        def create_remove_point_dialog():
            dialog = tk.Toplevel(self.tk_root)
            dialog.title("Remove Point Marker")
            dialog.geometry("300x150")
            
            tk.Label(dialog, text="Select Point to Remove:").pack()
            point_indices = [item[2] for item in self.layers[ElementLayer.MARKER]]
            point_var = StringVar(value=str(point_indices[0]))
            point_combo = ttk.Combobox(dialog, values=point_indices, state='readonly', textvariable=point_var)
            point_combo.pack(pady=5)
            def on_delete_point():
                point_idx = point_combo.current()
                print(f"Deleting point index: {point_idx}")
                to_deleted_line_idx = []
                to_deleted_points =None
                to_deleted_point_idx = -1
                if 0 <= point_idx < len(self.layers[ElementLayer.MARKER]):
                    ## remove entry in the points_to_line dict and update accordingly
                    for points, line_idx in list(self.points_to_line.items()):
                        if point_idx in points:
                            # remember to delete this line
                            to_deleted_line_idx.append(line_idx)
                            to_deleted_points = points
                            print(f"Will delete line index: {line_idx} associated with points: {points}")
                        else:
                            # update point indices in the keys
                            new_pnts = tuple(p-1 if p > point_idx else p for p in points)
                            if new_pnts != points:
                                self.points_to_line[new_pnts] = self.points_to_line.pop(points)
                    # remove the point
                    for i, (artist, artist_idx, idx) in enumerate(self.layers[ElementLayer.MARKER]):
                        if i == point_idx:
                            if artist:
                                artist.remove()
                            if artist_idx:
                                artist_idx.remove()
                            self.layers[ElementLayer.MARKER][i] = None
                            to_deleted_point_idx = i
                            self.fig_canvas.draw()
                        else:
                            if idx > point_idx:
                                self.layers[ElementLayer.MARKER][i] = (artist, artist_idx, idx-1)
                    self.layers[ElementLayer.MARKER].pop(to_deleted_point_idx)
                    # remove the associated lines
                    if len(to_deleted_line_idx) > 0:
                        sorted_line_idx = sorted(to_deleted_line_idx)
                        # update line indices in points_to_line dict
                        for points, l_idx in list(self.points_to_line.items()):
                            decrement = sum(1 for dl_idx in sorted_line_idx if dl_idx < l_idx)
                            if decrement > 0:
                                self.points_to_line[points] = l_idx - decrement
                        # update indices of lines
                        for i, (line, text, idx) in enumerate(self.layers[ElementLayer.GUIDELINE]):
                            # update only those lines after the removed line
                            decrement = sum(1 for dl_idx in sorted_line_idx if dl_idx < idx)
                            if decrement > 0:
                                self.layers[ElementLayer.GUIDELINE][i] = (line, text, idx - decrement)

                        # delete item from points_to_line
                        del self.points_to_line[to_deleted_points]
                        # remove associated line
                        for line_idx in to_deleted_line_idx:
                            line, text, idx = self.layers[ElementLayer.GUIDELINE][line_idx]
                            if line:
                                for artist in line:
                                    if artist:
                                        artist.remove()
                            if text:
                                text.remove()
                            self.layers[ElementLayer.GUIDELINE][line_idx] = None
                        # remove None entries
                        self.layers[ElementLayer.GUIDELINE] = [item for item in self.layers[ElementLayer.GUIDELINE] if item is not None]
                    self.update_points_marker()
                    self.update_lines()
                    self.fig_canvas.draw()
                    if dialog:
                        dialog.destroy()
                else:
                    messagebox.showwarning("Warning", "Invalid point index.")
            tk.Button(dialog, text="Confirm", command=on_delete_point).pack(pady=10)
        create_remove_point_dialog()

    def clear_all_markers(self):
        for marker in self.layers[ElementLayer.MARKER]:
            if marker:
                marker.remove()
        self.layers[ElementLayer.MARKER].clear()
        self.fig_canvas.draw()

    def remove_line(self):
        def create_remove_line_dialog():
            dialog = tk.Toplevel(self.tk_root)
            dialog.title("Remove Guideline")
            dialog.geometry("300x150")
            
            tk.Label(dialog, text="Select Line to Remove:").pack()
            line_indices = [i+1 for i in range(len(self.layers[ElementLayer.GUIDELINE]))]
            line_var = StringVar(value=str(line_indices[0]))
            line_combo = ttk.Combobox(dialog, values=line_indices, state='readonly', textvariable=line_var)
            line_combo.pack(pady=5)
            remove_points_var = tk.BooleanVar(value=False)
            tk.Checkbutton(dialog, text="Also remove associated points", variable=remove_points_var).pack()
            
            def on_confirm():
                line_idx = int(line_var.get()) - 1
                delete_points = []
                if 0 <= line_idx < len(self.layers[ElementLayer.GUIDELINE]):
                    # update points_to_line dict
                    for points, l_idx in list(self.points_to_line.items()):
                        if l_idx == line_idx:
                            if remove_points_var.get():
                                delete_points.extend(points)
                            # del self.points_to_line[points]
                        elif l_idx > line_idx:
                            self.points_to_line[points] = l_idx - 1
                    line, text = self.layers[ElementLayer.GUIDELINE].pop(line_idx)
                    if line:
                        for artist in line:
                            if artist:
                                artist.remove()
                    if text:
                        text.remove()
                    if remove_points_var.get():
                        for points, l_idx in list(self.points_to_line.items()):
                            if l_idx == line_idx:
                                for p_idx in points:
                                    if 0 <= p_idx < len(self.layers[ElementLayer.MARKER]):
                                        marker, idx = self.layers[ElementLayer.MARKER][p_idx]
                                        if marker:
                                            marker.remove()
                                        if idx:
                                            idx.remove()
                                del self.points_to_line[points]
                    self.update_points_marker()
                    self.update_lines()
                    self.fig_canvas.draw()
                    dialog.destroy()
                else:
                    messagebox.showwarning("Warning", "Invalid line index.")
            
            tk.Button(dialog, text="Confirm", command=on_confirm).pack(pady=10)
        create_remove_line_dialog()

    def clear_all_lines(self):
        for line in self.layers[ElementLayer.GUIDELINE]:
            if line:
                for artist in line:
                    if artist:
                        artist.remove()
        self.layers[ElementLayer.GUIDELINE] = []
        self.fig_canvas.draw()

    def reset_view(self):
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.clear_all_markers()
        self.clear_all_lines()
    
    def on_references(self):
        messagebox.showinfo("References", "Plot Analyser\nVersion 1.0")

    # def on_closing(self):
    #     if messagebox.askokcancel("Quit", "Do you want to quit?"):
    #         plt.close(self.fig)
    #         self.root.quit()
    #         self.root.destroy()
#endregion

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