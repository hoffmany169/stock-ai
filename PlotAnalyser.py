import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend
import matplotlib.pyplot as plt
import tkinter as tk
from Common.AutoNumber import AutoIndex

class COMMAND(AutoIndex):
      zoom_in = ()
      zoom_out = ()
      seperator_1 = ()
      add_point_marker = ()
      draw_horizontal_line = ()
      draw_vertical_line = ()
      seperator_2 = ()
      add_first_point = ()
      add_second_point = ()
      reset_view = ()
      seperator_3 = ()
      show_properties = ()
      save_image = ()

class PlotAnalyser:
    CONTEXT_MENU_TEXT = ['label', 'command']
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.canvas = fig.canvas
        self.last_click_coords = None
        self.commands = [] 
        self.__comm_init__()
        
    # def __init__(self, data, *args, **kwargs):
    #     self.fig, self.ax = plt.subplots()
    #     self.ax.plot(data, args, kwargs)
    #     self.canvas = self.fig.canvas
    #     self.__comm_init__()

    def __comm_init__(self):
        # Get the Tk root window
        self.root = self.canvas.manager.window
        self._create_context_menu_commands()
        # We'll create the menu dynamically each time        
        self.canvas.mpl_connect('button_press_event', self.on_right_click)
    
    def _create_context_menu_commands(self):
        self.menu = tk.Menu(self.root, tearoff=0)
        for e in COMMAND:
            if e.name.startswith('seperator'):
                self.menu.add_separator()
            else:
                self.menu.add_command(label=' '.join(e.name.split('_')))

    def on_right_click(self, event):
        if event.button == 3 and event.inaxes == self.ax:  # Right-click in axes
            # Convert matplotlib coordinates to tkinter coordinates
            # x_tk = self.root.winfo_pointerx() - self.root.winfo_rootx()
            # y_tk = self.root.winfo_pointery() - self.root.winfo_rooty()
            # menu = self._update_context_menu_event(event)
            self.last_click_coords = (event.xdata, event.ydata)
            self.fig._last_right_click = (event.xdata, event.ydata, event.inaxes)
            for e in COMMAND:
                if e.name.startswith('seperator'):
                    continue
                text = ' '.join(e.name.split('_'))
                self.menu.entryconfig(text, command=eval('self.{}(self.last_click_coords)'.format(e.name)))
            try:
                x_tk = self.root.winfo_pointerx()
                y_tk = self.root.winfo_pointery()
                self.menu.tk_popup(x_tk, y_tk)
            except:
                # Fallback
                self.menu.post(self.last_click_coords)
#region ### event callback functions    

    def add_point_marker(self, coords):
        if coords:
            x, y = coords
            self.ax.plot(x, y, 'ro', markersize=12, alpha=0.7, 
                        markeredgecolor='black', markeredgewidth=2)
            self.canvas.draw()
            print(f"Added point marker at ({x:.3f}, {y:.3f})")
    
    def draw_horizontal_line(self, coords):
        if coords:
            x, y = coords
            self.ax.axhline(y=y, color='purple', linestyle=':', alpha=0.7, linewidth=2)
            self.ax.text(0.5, y, f'y = {y:.2f}', 
                        transform=self.ax.get_yaxis_transform(),
                        color='purple', ha='center', va='bottom')
            self.canvas.draw()
        
    def draw_vertical_line(self, coords):
        if coords:
            x, y = coords
            # Draw vertical line at x position
            self.ax.axvline(x=x, color='g', linestyle='--', alpha=0.5)
            # Add text annotation
            self.ax.text(x, self.ax.get_ylim()[1]*0.9, f'x={x:.2f}', 
                        rotation=90, verticalalignment='top')
            self.canvas.draw()
            
    def save_image(self, coords):
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

    def reset_view(self, coords):
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
    
    def add_first_point(self, coords):
        if coords:
            x, y = coords
            self.ax.plot(x, y, 'go', markersize=10)
            self.fig.canvas.draw()

    def add_second_point(self, coords):
        if coords:
            x, y = coords
            self.ax.plot(x, y, 'go', markersize=10)
            self.fig.canvas.draw()

    def show_properties(self, coords):
        if coords:
            x, y = coords
            # Create a properties dialog
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