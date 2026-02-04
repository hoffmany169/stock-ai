import math, sys
from tkinter import StringVar, filedialog, messagebox
import matplotlib

from AxisRatioCalculator import AxisRatioCalculator
matplotlib.use('TkAgg')  # Use Tk backend
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from Common.AutoNumber import AutoIndex

class ElementLayer(AutoIndex):
    MARKER = ()
    ANNOTATION = ()
    GUIDELINE = ()
    ORIGINAL = ()

class COMMAND(AutoIndex):
      zoom_in = ()
      zoom_out = ()
      seperator_1 = ()
      draw_horizontal_line = ()
      draw_vertical_line = ()
      seperator_2 = ()
      add_point = ()
    #   add_second_point = ()


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
      show_point_properties = ()
      show_comparison_2_points = ()
      draw_line = ()
      seperator_1 = ()
      remove_point = ()
      remove_line = ()
      clear_all_markers = ()
      clear_all_lines = ()
      reset_view = ()

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

class VisualAnalyser:
    CONTEXT_MENU_TEXT = ['label', 'command']
    def __init__(self, fig=None, ax=None, data=None, plot_label='Data', figsize=(10,6)):
        """
        Docstring for __init__
        two scenarios:
        1. pass in fig and ax from outside
        :param fig: figure object in plotting
        :param ax: axis object in plotting
        2. create fig and ax inside
        :param data: data to plot
        :param figsize: figure size
        """
        if data is None:
            raise ValueError("Data must be provided when fig and ax are not given.")
        self._plot_file_name = None
        self.last_click_coords = None
        self.menu_items = {} # map function name (string) -> command index
        if fig:
            self.fig = fig
            self.ax = plt.gca()
        elif ax:
            self.ax = ax
            self.fig = plt.gcf()
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)

        self.x, self.y = data
        line, = self.ax.plot(self.x, self.y, 'b-', label=plot_label, linewidth=2)
        self.ax.grid(True, alpha=0.3)
        self.canvas = self.fig.canvas
        self.axis_ratio_calculator = AxisRatioCalculator(self.ax)
        self.curve = {
            'x': np.array(self.x),
            'y': np.array(self.y),
            'points': np.column_stack((self.x, self.y)),
            'artist': line,
            'color': line.get_color()
        }
        self.__comm_init__()
        # plt.show()
        
    def __comm_init__(self):
        # Get the Tk root window
        self.root = self.canvas.manager.window
        self._marker_style = MARKER_STYLE.red_circle
        self._line_style = LINE_STYLE.dashed_line
        # if activate real-time search
        self._real_time_search = True # default is True
        # Set window title
        self.root.title("Matplotlib Plot with Menu Bar")
        
        # Set window size
        self.root.geometry("800x600")
        
        # Create the menu bar
        self._create_menu_bar()
        
        # Create toolbar (matplotlib's default)
        self.fig.canvas.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._create_context_menu_commands()
        # store elements by layer
        self.layers = {
            ElementLayer.MARKER: [], # keep only two markers for point selection
            ElementLayer.ANNOTATION: [],
            ElementLayer.GUIDELINE: []
        }
        self.points_to_line = {}
        # We'll create the menu dynamically each time        
        self.canvas.mpl_connect('button_press_event', self.on_right_click)
        # self.canvas.mpl_connect('motion_notify_event', self.on_hover)

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

    def show_plot(self):
        plt.show()

    def _create_menu_bar(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
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
                cmd = getattr(self, e.name)
                menu.add_command(label=label_text, 
                                command=cmd)

    def _create_context_menu_commands(self):
        self.context_menu = tk.Menu(self.root, tearoff=0)
        for e in COMMAND:
            if e.name.startswith('seperator'):
                self.context_menu.add_separator()
            else:
                self.context_menu.add_command(label=' '.join(e.name.split('_')), command=self.dummy_command)
                self.menu_items[e.name] = self.context_menu.index("end")

    def fill_between(self, alpha=0.2):
        """Fill area under the curve with specified alpha transparency."""
        self.ax.fill_between(self.x, self.y, alpha=alpha)
        self.canvas.draw()

    def set_labels(self, xlabel='', ylabel=''):
        """Set x and y axis labels."""
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.canvas.draw()

    def set_legend(self, location='upper right', shadow=True, fsize='x-small'):
        """Set legend on the plot."""
        self.ax.legend(loc=location, shadow=shadow, fontsize=fsize)
        self.canvas.draw()

    def on_right_click(self, event):
        if event.button == 3 and event.inaxes == self.ax:  # Right-click in axes
            self.last_click_coords = (event.xdata, event.ydata)
            self.fig._last_right_click = (event.xdata, event.ydata, event.inaxes)
            for key, index in self.menu_items.items():
                # trick here: set key as a default argument to prevent of late binding of variables in lambdas inside a loop!!!
                self.context_menu.entryconfig(index, command=lambda item=key: getattr(self, item)(self.last_click_coords))
            try:
                x_tk = self.root.winfo_pointerx()
                y_tk = self.root.winfo_pointery()
                self.context_menu.tk_popup(x_tk, y_tk)
            except:
                # Fallback
                self.context_menu.post(self.last_click_coords)
            
#region ### event callback functions    
    def dummy_command(self):
        """Placeholder - will be replaced"""
        print("This dummy_command")

    # def add_point_marker(self, coords):
    
    def draw_horizontal_line(self, coords):
        if coords:
            x, y = coords
            artist1 = self.ax.axhline(y=y, color='purple', linestyle=self.line_style, alpha=0.7, linewidth=2)
            artist2 = self.ax.text(0.5, y, f'y = {y:.2f}', 
                        transform=self.ax.get_yaxis_transform(),
                        color='purple', ha='center', va='bottom')
            self.canvas.draw()
            self.layers[ElementLayer.GUIDELINE].append((artist1, artist2))
        
    def draw_vertical_line(self, coords):
        if coords:
            x, y = coords
            # Draw vertical line at x position
            artist1 = self.ax.axvline(x=x, color='g', linestyle=self.line_style, alpha=0.5)
            # Add text annotation
            artist2 = self.ax.text(x, self.ax.get_ylim()[1]*0.9, f'x={x:.2f}', 
                        rotation=90, verticalalignment='top')
            self.canvas.draw()
            self.layers[ElementLayer.GUIDELINE].append((artist1, artist2))
    
    def zoom_in(self, coords):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim(xlim[0]*0.8, xlim[1]*0.8)
        self.ax.set_ylim(ylim[0]*0.8, ylim[1]*0.8)
        self.fig.canvas.draw()
    
    def zoom_out(self, coords):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim(xlim[0]*1.2, xlim[1]*1.2)
        self.ax.set_ylim(ylim[0]*1.2, ylim[1]*1.2)
        self.fig.canvas.draw()
    
    def add_point(self, coords):
        if coords:
            result = self.find_closest_point(coords[0], coords[1])
            x, y = result['point']
            # print(f"Clicked at data coords: ({x}, {y})")
            # x_tk = self.root.winfo_pointerx() - self.root.winfo_rootx()
            # y_tk = self.root.winfo_pointery() - self.root.winfo_rooty()
            # print(f"Clicked at Tk coords: ({x_tk}, {y_tk})")
            artist = self.ax.plot(x, y, self.marker_style, markersize=4, alpha=0.7, 
                        markeredgecolor='black', markeredgewidth=2)[0]
            idx = len(self.layers[ElementLayer.MARKER]) + 1
            artist_idx = self.ax.text(x, y, f'P{idx}', weight='bold', 
                        verticalalignment='bottom', horizontalalignment='left', color='black')
            self.layers[ElementLayer.MARKER].append((artist, artist_idx, idx))
            self.canvas.draw()
            print(f"Added point marker at ({x:.3f}, {y:.3f})")

    def find_closest_point(self, x_click, y_click):
        """
        暴力搜索：计算所有距离，找到最小值
        时间复杂度：O(n)，适合小数据集
        """
        x_curve = self.curve['x']
        y_curve = self.curve['y']
        # 计算所有距离
        distances = np.sqrt((x_curve - x_click)**2 + (y_curve - y_click)**2)
        
        # 找到最小距离的索引
        min_index = np.argmin(distances)
        
        return {
            'index': min_index,
            'point': (x_curve[min_index], y_curve[min_index]),
            'distance': distances[min_index]
        }

    def _find_closest_marker_index(self, x, y):
        min_dist = float('inf')
        closest_idx = -1
        for i, (artist, artist_idx, idx) in enumerate(self.layers[ElementLayer.MARKER]):
            px = float(artist.get_xdata()[0])
            py = float(artist.get_ydata()[0])
            dist = math.hypot(px - x, py - y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx

    def update_points_marker(self):
        for artist, artist_idx, idx in self.layers[ElementLayer.MARKER]:
            x = float(artist.get_xdata()[0])
            y = float(artist.get_ydata()[0])
            artist_idx.set_position((x, y))
            artist_idx.set_text(f'P{idx}')
        self.canvas.draw()

    def show_comparison_2_points(self):
        x1, y1 = self.layers[ElementLayer.MARKER][0].get_xdata()[0], self.layers[ElementLayer.MARKER][0].get_ydata()[0]
        x2, y2 = self.layers[ElementLayer.MARKER][1].get_xdata()[0], self.layers[ElementLayer.MARKER][1].get_ydata()[0]
        x_diff = x2 - x1
        y_diff = y2 - y1
        y_perc = (y_diff / y1 * 100) if y1 != 0 else float('inf')
        tangent = (y_diff / x_diff) if x_diff !=0 else float('inf')
        selections = [' '.join(e.name.split('_')) for e in PROPERTY_2_POINTS]
        data = [0]*len(selections)
        data[0] = x_diff
        data[1] = y_diff
        data[2] = y_perc
        data[3] = tangent
        sel_data = dict(zip(selections, data))
        selected_text = StringVar(value=selections[0])
        # create pup-up dialog
        def create_properties_dialog():
            dialog = tk.Toplevel(self.root)
            dialog.title("comparison of 2 points")
            dialog.geometry("300x300")
            
            tk.Label(dialog, text=f"Coordinates:", font=('Arial', 12, 'bold')).pack()
            tk.Label(dialog, text=f"x1, y1 = ({x1:.6f}, {y1:.6f}) ").pack()
            tk.Label(dialog, text=f"x2, y2 = ({x2:.6f}, {y2:.6f})").pack()
            tk.Label(dialog, text=f"{selections[0]} = {x_diff:.6f}").pack()
            tk.Label(dialog, text=f"{selections[1]} = {y_diff:.6f}").pack()
            tk.Label(dialog, text=f"{selections[2]} = {y_perc:.2f}%").pack()
            tk.Label(dialog, text=f"{selections[3]} = {tangent:.2f}").pack()
            tk.Label(dialog, text=f"Degree of Line = {math.atan2(y_diff, x_diff)*180/math.pi:.2f}").pack()
            combo = ttk.Combobox(dialog, values=selections, state='readonly', textvariable=selected_text)
            combo.pack(pady=10)
            tk.Button(dialog, text="Record", command=lambda text=selected_text.get(): self.record_data(sel_data[text])).pack(pady=10)
        create_properties_dialog()

    def record_data(self, value):
        print(f"Recorded value: {value}")

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
                dialog = tk.Toplevel(self.root)
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
            self.canvas.draw()
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
        self.canvas.draw()

    def show_point_properties(self):
        # Create a properties dialog
        coords = self.last_click_coords
        if coords:
            x, y = coords
            dialog = tk.Toplevel(self.root)
            dialog.title("Point Properties")
            dialog.geometry("300x200")
            
            tk.Label(dialog, text=f"Coordinates:", font=('Arial', 12, 'bold')).pack()
            tk.Label(dialog, text=f"x = {x:.6f}").pack()
            tk.Label(dialog, text=f"y = {y:.6f}").pack()
            
            # Add entry to modify values
            tk.Label(dialog, text="\nNew y value:").pack()
            new_y_var = tk.StringVar(value=str(y))
            entry = tk.Entry(dialog, textvariable=new_y_var)
            entry.pack()
            
            def update_point():
                try:
                    new_y = float(new_y_var.get())
                    # Find and update the point if it exists
                    for artist in self.ax.get_children():
                        if hasattr(artist, 'get_label') and artist.get_label() == 'Clicked point':
                            # This is a simplified approach - in reality, you'd need to track the point
                            pass
                    print(f"Would update point to y={new_y}")
                except ValueError:
                    pass
            tk.Button(dialog, text="Update", command=update_point).pack(pady=10)

    def open_file(self):
        pass

    def save_image(self):
        if self._plot_file_name:
            self.fig.savefig(self._plot_file_name, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Save", f"Plot saved to:\n{self._plot_file_name}")
        else:
            self.save_image_as()

    def save_image_as(self):
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

    def export_pdf(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if filename:
            self.fig.savefig(filename, format='pdf', bbox_inches='tight')
            messagebox.showinfo("Export PDF", f"Plot saved to:\n{filename}")
    
    def exit(self):
        sys.exit(0)

    def remove_point(self):
        def create_remove_point_dialog():
            dialog = tk.Toplevel(self.root)
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
                            self.canvas.draw()
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
                    self.canvas.draw()
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
        self.canvas.draw()

    def remove_line(self):
        def create_remove_line_dialog():
            dialog = tk.Toplevel(self.root)
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
                    self.canvas.draw()
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
        self.canvas.draw()

    def reset_view(self):
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.clear_all_markers()
        self.clear_all_lines()
    
    def references(self):
        messagebox.showinfo("References", "Plot Analyser\nVersion 1.0")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            plt.close(self.fig)
            self.root.quit()
            self.root.destroy()
#endregion

import numpy as np
def main(type):
    if type == 0: ## without fig
        # 2- create plot in PlotAnalyser class
        data_x = np.linspace(0, 10, 100)
        data_y = np.cos(data_x) * np.exp(-0.05*data_x)
        dynamic_menu = PlotAnalyser(data=(data_x, data_y), figsize=(10,6))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(0, 4*np.pi, 200)
        y = np.sin(x) * np.exp(-0.1*x)
        # ax.legend()
        # ax.grid(True, alpha=0.3)

        if type == 1: ## with fig
            # Create dynamic context menu
            dynamic_menu = PlotAnalyser(data=(x, y), fig=fig)
        else: ## with ax
            dynamic_menu = PlotAnalyser(data=(x, y), ax=ax)
    dynamic_menu.fill_between(alpha=0.2)
    dynamic_menu.set_labels(xlabel='X-axis', ylabel='Y-axis')
    dynamic_menu.set_legend()
    dynamic_menu.show_plot()

if __name__ == '__main__':
    main(2)