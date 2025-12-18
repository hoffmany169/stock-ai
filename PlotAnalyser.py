import math
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
      add_first_point = ()
      add_second_point = ()
      seperator_3 = ()


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
      remove_last_point = ()
      remove_last_line = ()
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

class PlotAnalyser:
    CONTEXT_MENU_TEXT = ['label', 'command']
    def __init__(self, fig=None, ax=None, data=None, figsize=(10,6)):
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
        self._plot_file_name = None
        self.last_click_coords = None
        self.menu_items = {} # map function name (string) -> command index
        if data is not None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
            x, y = data
            self.ax.plot(x, y, 'b-', linewidth=2)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.grid(True, alpha=0.3)
        else:
            self.fig = fig
            self.ax = ax
        self.canvas = self.fig.canvas
        self.axis_ratio_calculator = AxisRatioCalculator(self.ax)
        self.__comm_init__()
        plt.show()
        
    # def __init__(self, data):
    #     self.fig, self.ax = plt.subplots(figsize=(10, 6))
    #     self.canvas = self.fig.canvas
    #     self.__comm_init__()

    def __comm_init__(self):
        # Get the Tk root window
        self.root = self.canvas.manager.window
        self._marker_style = MARKER_STYLE.red_circle
        self._line_style = LINE_STYLE.dashed_line
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
        # We'll create the menu dynamically each time        
        self.canvas.mpl_connect('button_press_event', self.on_right_click)

#region # properties    
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
                if e.name == 'exit':
                    menu.add_command(label=label_text, 
                                    command=cmd)
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
    
    def add_first_point(self, coords):
        if coords:
            x, y = coords
            # print(f"Clicked at data coords: ({x}, {y})")
            # x_tk = self.root.winfo_pointerx() - self.root.winfo_rootx()
            # y_tk = self.root.winfo_pointery() - self.root.winfo_rooty()
            # print(f"Clicked at Tk coords: ({x_tk}, {y_tk})")
            artist = self.ax.plot(x, y, self.marker_style, markersize=10, alpha=0.7, 
                        markeredgecolor='black', markeredgewidth=2)[0]
            if len(self.layers[ElementLayer.MARKER]) > 0:
                self.layers[ElementLayer.MARKER].clear()
            self.layers[ElementLayer.MARKER].append(artist)
            self.canvas.draw()
            print(f"Added point marker at ({x:.3f}, {y:.3f})")

    def add_second_point(self, coords):
        if coords:
            x, y = coords
            artist = self.ax.plot(x, y, self.marker_style, markersize=10, alpha=0.7, 
                        markeredgecolor='black', markeredgewidth=2)[0]
            if len(self.layers[ElementLayer.MARKER]) == 0:
                messagebox.showwarning("Warning", "Please add the first point before adding the second point.")
                return
            elif len(self.layers[ElementLayer.MARKER]) > 1:
                self.layers[ElementLayer.MARKER].slice(1, None)  # keep only the first point
            self.layers[ElementLayer.MARKER].append(artist)  # placeholder for first point
            self.canvas.draw()
            print(f"Added point marker at ({x:.3f}, {y:.3f})")

    def show_comparison_2_points(self):
        x1, y1 = self.layers[ElementLayer.MARKER][0].get_xdata()[0], self.layers[ElementLayer.MARKER][0].get_ydata()[0]
        x2, y2 = self.layers[ElementLayer.MARKER][1].get_xdata()[0], self.layers[ElementLayer.MARKER][1].get_ydata()[0]
        x_diff = x2 - x1
        y_diff = y2 - y1
        y_perc = (y_diff / y1 * 100) if y1 != 0 else float('inf')
        tangent = (y_diff / x_diff) if x_diff !=0 else float('inf')
        # create pup-up dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("comparison of 2 points")
        dialog.geometry("300x300")
        
        selections = [' '.join(e.name.split('_')) for e in PROPERTY_2_POINTS]
        data = [0]*len(selections)
        data[0] = x_diff
        data[1] = y_diff
        data[2] = y_perc
        data[3] = tangent
        sel_data = dict(zip(selections, data))
        tk.Label(dialog, text=f"Coordinates:", font=('Arial', 12, 'bold')).pack()
        tk.Label(dialog, text=f"x1, y1 = ({x1:.6f}, {y1:.6f}) ").pack()
        tk.Label(dialog, text=f"x2, y2 = ({x2:.6f}, {y2:.6f})").pack()
        tk.Label(dialog, text=f"{selections[0]} = {x_diff:.6f}").pack()
        tk.Label(dialog, text=f"{selections[1]} = {y_diff:.6f}").pack()
        tk.Label(dialog, text=f"{selections[2]} = {y_perc:.2f}%").pack()
        tk.Label(dialog, text=f"{selections[3]} = {tangent:.2f}").pack()
        tk.Label(dialog, text=f"Degree of Line = {math.atan2(y_diff, x_diff)*180/math.pi:.2f}").pack()
        selected_text = StringVar(value=selections[0])
        combo = ttk.Combobox(dialog, values=selections, state='readonly', textvariable=selected_text)
        combo.pack(pady=10)
        tk.Button(dialog, text="Record", command=lambda text=selected_text.get(): self.record_data(sel_data[text])).pack(pady=10)

    def record_data(self, value):
        print(f"Recorded value: {value}")

    def draw_line(self):
        """
        Docstring for draw_line
        Visual angle = arctan(Data slope × (y_scale / x_scale))
             = arctan(Data slope / Aspect_ratio)
        """
        if len(self.layers[ElementLayer.MARKER]) == 2:
            x1 = float(self.layers[ElementLayer.MARKER][0].get_xdata()[0])
            y1 = float(self.layers[ElementLayer.MARKER][0].get_ydata()[0])
            x2 = float(self.layers[ElementLayer.MARKER][1].get_xdata()[0])
            y2 = float(self.layers[ElementLayer.MARKER][1].get_ydata()[0])
            artist1 = self.ax.plot([x1, x2], [y1, y2], self.line_style, color='b')
            visual_angle = self.axis_ratio_calculator.get_visual_angle(x1, y1, x2, y2)
            artist2 = self.ax.text((x1+x2)/2, (y1+y2)/2, f'tangent={visual_angle:.2f}', 
                        rotation=visual_angle, verticalalignment='top')
            self.canvas.draw()
            self.layers[ElementLayer.GUIDELINE].append((artist1, artist2))

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
        pass

    def remove_last_point(self):
        if len(self.layers[ElementLayer.MARKER]) > 0:
            last_marker = self.layers[ElementLayer.MARKER].pop()
            if last_marker:
                last_marker.remove()
                self.canvas.draw()

    def clear_all_markers(self):
        for marker in self.layers[ElementLayer.MARKER]:
            if marker:
                marker.remove()
        self.layers[ElementLayer.MARKER].clear()
        self.canvas.draw()

    def remove_last_line(self):
        if len(self.layers[ElementLayer.GUIDELINE]) > 0:
            last_line, text = self.layers[ElementLayer.GUIDELINE].pop()
            if last_line:
                for artist in last_line:
                    if artist:
                        artist.remove()
            if text:
                text.remove()
            self.canvas.draw()

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
    if type == 1:
        # 1- Create sample plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(0, 4*np.pi, 200)
        y = np.sin(x) * np.exp(-0.1*x)
        ax.plot(x, y, 'b-', linewidth=2, label='sin(x) * exp(-0.1x)')
        ax.fill_between(x, y, alpha=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Create dynamic context menu
        dynamic_menu = PlotAnalyser(fig, ax)
    else:
        # 2- create plot in PlotAnalyser class
        data_x = np.linspace(0, 10, 100)
        data_y = np.cos(data_x) * np.exp(-0.05*data_x)
        dynamic_menu = PlotAnalyser(data=(data_x, data_y), figsize=(10,6))

if __name__ == '__main__':
    main(0)