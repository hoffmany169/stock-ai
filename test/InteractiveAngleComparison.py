import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Arc

class InteractiveAngleComparison:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Line Angle Comparison")
        self.root.geometry("1200x700")
        
        # Create matplotlib figure with two subplots
        self.fig, (self.ax_data, self.ax_visual) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Control panel
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.setup_controls()
        self.setup_plots()
        
        # Initialize line
        self.line_start = (2, 2)
        self.line_end = (5, 4)
        self.update_line()
        
        # Bind mouse events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.dragging = False
        self.drag_point = None  # 'start' or 'end'
    
    def setup_controls(self):
        """Setup control widgets"""
        # Aspect ratio control
        tk.Label(self.control_frame, text="Aspect Ratio:").pack(side=tk.LEFT, padx=5)
        
        self.aspect_var = tk.StringVar(value='auto')
        aspect_menu = tk.OptionMenu(self.control_frame, self.aspect_var, 
                                   'auto', 'equal', '1.0', '2.0', '0.5',
                                   command=self.on_aspect_change)
        aspect_menu.pack(side=tk.LEFT, padx=5)
        
        # Line coordinates
        coord_frame = tk.Frame(self.control_frame)
        coord_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(coord_frame, text="Start (x,y):").grid(row=0, column=0)
        self.start_x_var = tk.StringVar(value='2.0')
        self.start_y_var = tk.StringVar(value='2.0')
        tk.Entry(coord_frame, textvariable=self.start_x_var, width=6).grid(row=0, column=1)
        tk.Entry(coord_frame, textvariable=self.start_y_var, width=6).grid(row=0, column=2)
        
        tk.Label(coord_frame, text="End (x,y):").grid(row=1, column=0)
        self.end_x_var = tk.StringVar(value='5.0')
        self.end_y_var = tk.StringVar(value='4.0')
        tk.Entry(coord_frame, textvariable=self.end_x_var, width=6).grid(row=1, column=1)
        tk.Entry(coord_frame, textvariable=self.end_y_var, width=6).grid(row=1, column=2)
        
        tk.Button(coord_frame, text="Update", 
                 command=self.update_from_entries).grid(row=0, column=3, rowspan=2)
        
        # Info display
        info_frame = tk.Frame(self.control_frame)
        info_frame.pack(side=tk.RIGHT, padx=10)
        
        self.angle_label = tk.Label(info_frame, text="Data Angle: --°", font=('Arial', 10))
        self.angle_label.pack()
        self.visual_label = tk.Label(info_frame, text="Visual Angle: --°", font=('Arial', 10))
        self.visual_label.pack()
        self.ratio_label = tk.Label(info_frame, text="Axis Ratio: --", font=('Arial', 10))
        self.ratio_label.pack()
    
    def setup_plots(self):
        """Setup the two plot areas"""
        # Left: Data space (with grid showing data units)
        self.ax_data.clear()
        self.ax_data.set_xlim(0, 10)
        self.ax_data.set_ylim(0, 10)
        self.ax_data.grid(True, alpha=0.3)
        self.ax_data.set_aspect('equal')  # Always equal for data view
        self.ax_data.set_title("Data Space (Mathematical)")
        self.ax_data.set_xlabel("X (data units)")
        self.ax_data.set_ylabel("Y (data units)")
        
        # Add reference lines and circle for 45°
        self.ax_data.axhline(0, color='gray', alpha=0.5, linewidth=0.5)
        self.ax_data.axvline(0, color='gray', alpha=0.5, linewidth=0.5)
        
        # Right: Visual space (with current aspect ratio)
        self.ax_visual.clear()
        self.ax_visual.set_xlim(0, 10)
        self.ax_visual.set_ylim(0, 10)
        self.ax_visual.grid(True, alpha=0.3)
        self.ax_visual.set_title("Visual Space (Display)")
        self.ax_visual.set_xlabel("X (data units)")
        self.ax_visual.set_ylabel("Y (data units)")
        
        self.ax_visual.axhline(0, color='gray', alpha=0.5, linewidth=0.5)
        self.ax_visual.axvline(0, color='gray', alpha=0.5, linewidth=0.5)
        
        # Draw reference circle that will appear as ellipse if aspect ≠ 1
        circle = plt.Circle((5, 5), 3, fill=False, color='blue', 
                           alpha=0.3, linewidth=1, linestyle='--')
        self.ax_visual.add_patch(circle)
    
    def on_aspect_change(self, value):
        """Handle aspect ratio change"""
        if value == 'auto':
            self.ax_visual.set_aspect('auto')
        elif value == 'equal':
            self.ax_visual.set_aspect('equal')
        else:
            self.ax_visual.set_aspect(float(value))
        
        self.update_line()
    
    def update_from_entries(self):
        """Update line from entry widgets"""
        try:
            self.line_start = (float(self.start_x_var.get()), 
                              float(self.start_y_var.get()))
            self.line_end = (float(self.end_x_var.get()), 
                            float(self.end_y_var.get()))
            self.update_line()
        except ValueError:
            pass
    
    def update_line(self):
        """Update the line in both plots"""
        # Clear previous lines and annotations
        self.ax_data.clear()
        self.ax_visual.clear()
        
        # Re-setup plots
        self.setup_plots()
        
        # Draw line in both plots
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        
        # Data plot (always equal aspect)
        self.line_data, = self.ax_data.plot([x1, x2], [y1, y2], 
                                           'r-', linewidth=3, label='Line')
        
        # Visual plot (with current aspect)
        self.line_visual, = self.ax_visual.plot([x1, x2], [x1, x2], 
                                               'r-', linewidth=3, label='Line')
        
        # Add points
        self.ax_data.plot(x1, y1, 'ro', markersize=8)
        self.ax_data.plot(x2, y2, 'ro', markersize=8)
        self.ax_visual.plot(x1, y1, 'ro', markersize=8)
        self.ax_visual.plot(x2, y2, 'ro', markersize=8)
        
        # Calculate and display angles
        self.display_angles(x1, y1, x2, y2)
        
        # Redraw
        self.canvas.draw()
    
    def display_angles(self, x1, y1, x2, y2):
        """Calculate and display angle information"""
        # Data angle (mathematical)
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0:
            data_angle = 90.0
        else:
            data_angle = math.degrees(math.atan(dy / dx))
        
        # Adjust for quadrant
        if dx < 0:
            data_angle += 180
        elif dy < 0:
            data_angle += 360
        
        data_angle = data_angle % 360
        
        # Display angle (what you see)
        trans = self.ax_visual.transData
        p1_display = trans.transform((x1, y1))
        p2_display = trans.transform((x2, y2))
        
        dx_display = p2_display[0] - p1_display[0]
        dy_display = p2_display[1] - p1_display[1]
        
        if dx_display == 0:
            display_angle = 90.0
        else:
            display_angle = math.degrees(math.atan(dy_display / dx_display))
        
        # Adjust for quadrant in display
        if dx_display < 0:
            display_angle += 180
        elif dy_display < 0:
            display_angle += 360
        
        display_angle = display_angle % 360
        
        # Axis ratio
        delta = 1e-6
        p0 = trans.transform((x1, y1))
        px = trans.transform((x1 + delta, y1))
        py = trans.transform((x1, y1 + delta))
        
        x_scale = abs(px[0] - p0[0]) / delta
        y_scale = abs(py[1] - p0[1]) / delta
        
        if y_scale == 0:
            ratio = float('inf')
        else:
            ratio = x_scale / y_scale
        
        # Update labels
        self.angle_label.config(text=f"Data Angle: {data_angle:.1f}°")
        self.visual_label.config(text=f"Visual Angle: {display_angle:.1f}°")
        self.ratio_label.config(text=f"Axis Ratio (x/y): {ratio:.3f}")
        
        # Add angle annotations on plots
        # Data plot
        self.ax_data.text(x1, y1 - 0.5, f"{data_angle:.1f}°", 
                         fontsize=10, color='red',
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor="yellow", alpha=0.7))
        
        # Visual plot
        self.ax_visual.text(x1, y1 - 0.5, f"{display_angle:.1f}°", 
                           fontsize=10, color='red',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                    facecolor="yellow", alpha=0.7))
        
        # Draw angle arcs
        self.draw_angle_arc(self.ax_data, x1, y1, data_angle, 'green')
        self.draw_angle_arc(self.ax_visual, x1, y1, display_angle, 'blue')
        
        # Add line equation
        if dx != 0:
            slope = dy / dx
            intercept = y1 - slope * x1
            eq_text = f"y = {slope:.2f}x + {intercept:.2f}"
        else:
            eq_text = f"x = {x1:.2f}"
        
        self.ax_data.text(0.5, 0.95, eq_text, 
                         transform=self.ax_data.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor="white", alpha=0.8))
    
    def draw_angle_arc(self, ax, x, y, angle_deg, color):
        """Draw an arc showing the angle"""
        # Convert angle to radians
        angle_rad = math.radians(angle_deg)
        
        # Determine arc parameters
        radius = 1.0
        theta1 = 0  # Start at positive x-axis
        theta2 = angle_deg  # End at line angle
        
        # Draw arc
        arc = Arc((x, y), radius*2, radius*2, 
                 angle=0, theta1=theta1, theta2=theta2,
                 color=color, linewidth=2, alpha=0.7)
        ax.add_patch(arc)
    
    def on_press(self, event):
        """Handle mouse press"""
        if event.inaxes not in [self.ax_data, self.ax_visual]:
            return
        
        # Check which point was clicked
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        
        # Tolerance in data units
        tol = 0.5
        
        if abs(event.xdata - x1) < tol and abs(event.ydata - y1) < tol:
            self.dragging = True
            self.drag_point = 'start'
        elif abs(event.xdata - x2) < tol and abs(event.ydata - y2) < tol:
            self.dragging = True
            self.drag_point = 'end'
    
    def on_drag(self, event):
        """Handle mouse drag"""
        if not self.dragging or event.inaxes is None:
            return
        
        if self.drag_point == 'start':
            self.line_start = (event.xdata, event.ydata)
        elif self.drag_point == 'end':
            self.line_end = (event.xdata, event.ydata)
        
        # Update entry widgets
        self.start_x_var.set(f"{self.line_start[0]:.2f}")
        self.start_y_var.set(f"{self.line_start[1]:.2f}")
        self.end_x_var.set(f"{self.line_end[0]:.2f}")
        self.end_y_var.set(f"{self.line_end[1]:.2f}")
        
        self.update_line()
    
    def on_release(self, event):
        """Handle mouse release"""
        self.dragging = False
        self.drag_point = None
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

# Run the interactive application
if __name__ == "__main__":
    app = InteractiveAngleComparison()
    app.run()