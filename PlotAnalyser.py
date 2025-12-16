import math
from tkinter import messagebox
import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend
import matplotlib.pyplot as plt
import tkinter as tk
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
      show_point_properties = ()

class MENU_COMMAND(AutoIndex):
      calculate_period_distance = ()
      calculate_value_percentage = ()
      draw_line = ()
      seperator_1 = ()
      clear_markers = ()
      reset_view = ()
      seperator_2 = ()
      save_image = ()

class MARKER_STYLE(AutoIndex):
    red_circle = ()
    green_cross = ()
    blue_triangle = ()

class LINE_STYLE(AutoIndex):
    solid_line = ()
    dashed_line = ()
    dash_dot_line = ()
    dotted_line = ()

class PlotAnalyser:
    CONTEXT_MENU_TEXT = ['label', 'command']
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.canvas = fig.canvas
        self.last_click_coords = None
        self.menu_items = {} # map function name (string) -> command index
        self.__comm_init__()
        
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
        self.create_menu_bar()
        
        # Create toolbar (matplotlib's default)
        self.fig.canvas.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
                
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._create_context_menu_commands()
        # store elements by layer
        self.layers = {
            ElementLayer.MARKER: [None]*2,
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

    def create_menu_bar(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.action_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='Action', menu=self.action_menu)
        for e in MENU_COMMAND:
            label_text = ' '.join(e.name.split('_'))
            if e.name.startswith('seperator'):
                self.action_menu.add_separator()
            else:
                cmd = getattr(self, e.name)
                self.action_menu.add_command(label=label_text, 
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
            # Convert matplotlib coordinates to tkinter coordinates
            # x_tk = self.root.winfo_pointerx() - self.root.winfo_rootx()
            # y_tk = self.root.winfo_pointery() - self.root.winfo_rooty()
            # menu = self._update_context_menu_event(event)
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

    def clear_markers(self):
        for line in self.fig._last_right_click[2].lines[1:]:
            line.remove()

    def save_image(self):
        self.fig.savefig('output.png')
        print("Image saved")
    
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

    def reset_view(self):
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
    
    def add_first_point(self, coords):
        if coords:
            x, y = coords
            artist = self.ax.plot(x, y, self.marker_style, markersize=10, alpha=0.7, 
                        markeredgecolor='black', markeredgewidth=2)[0]
            self.layers[ElementLayer.MARKER][0] = artist
            self.canvas.draw()
            print(f"Added point marker at ({x:.3f}, {y:.3f})")

    def add_second_point(self, coords):
        if coords:
            x, y = coords
            artist = self.ax.plot(x, y, self.marker_style, markersize=10, alpha=0.7, 
                        markeredgecolor='black', markeredgewidth=2)[0]
            self.layers[ElementLayer.MARKER][1] = artist
            self.canvas.draw()
            print(f"Added point marker at ({x:.3f}, {y:.3f})")

    def calculate_value_percentage(self):
        # x_diff = coords[0] - self.first_point[0]
        # x_perc = x_diff / self.add_first_point[0] * 100
        pass

    def calculate_period_distance(self):
        pass

    def draw_line(self):
        if len(self.layers[ElementLayer.MARKER]) == 2:
            x1 = float(self.layers[ElementLayer.MARKER][0].get_xdata()[0])
            y1 = float(self.layers[ElementLayer.MARKER][0].get_ydata()[0])
            x2 = float(self.layers[ElementLayer.MARKER][1].get_xdata()[0])
            y2 = float(self.layers[ElementLayer.MARKER][1].get_ydata()[0])
            artist1 = self.ax.plot([x1, x2], [y1, y2], self.line_style, color='b')
            artist2 = self.ax.text(x1, y1, f'tangent={(y2-y1)/(x2-x1):.2f}', 
                        rotation=(math.atan2((y2-y1), (x2-x1)))*60, verticalalignment='top')
            self.canvas.draw()
            self.layers[ElementLayer.GUIDELINE].append((artist1, artist2))

    def show_point_properties(self, coords):
        # Create a properties dialog
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

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            plt.close(self.fig)
            self.root.quit()
            self.root.destroy()
#endregion


    # def _update_context_menu_event(self, event):
    #     self.last_click_coords = (event.xdata, event.ydata)
    #     self.fig._last_right_click = (event.xdata, event.ydata, event.inaxes)
    #     menu = tk.Menu(self.root, tearoff=0)
    #     for item in self.commands:
    #         keys = list(item.keys())
    #         if item.get(keys[0]).startswith('seperator'):
    #             cmd = getattr(menu, item.get(keys[1]))
    #             cmd()
    #         else:
    #             cmd = item.get(keys[1])
    #             menu.add_command(label=item.get(keys[0]), command=lambda: cmd(self.last_click_coords))
    #     return menu

import numpy as np
def main():
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

    plt.show()

if __name__ == '__main__':
    main()