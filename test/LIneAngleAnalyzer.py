import matplotlib.pyplot as plt
import numpy as np
import math

class LineAngleAnalyzer:
    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        self.line = None
        
    def draw_line(self, x1, y1, x2, y2, color='red', linewidth=2):
        """Draw a line and store it"""
        self.line, = self.ax.plot([x1, x2], [y1, y2], 
                                 color=color, linewidth=linewidth)
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        
    def calculate_data_angle(self):
        """Calculate angle in data coordinates (mathematical slope)"""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        
        if dx == 0:  # Vertical line
            return 90.0 if dy > 0 else -90.0
        
        # Slope in data units
        slope_data = dy / dx
        
        # Angle in degrees (arctan gives angle in radians)
        angle_data = math.degrees(math.atan(slope_data))
        
        # Adjust for quadrant
        if dx < 0:
            angle_data += 180
        elif dy < 0:
            angle_data += 360
        
        # Normalize to 0-360
        angle_data = angle_data % 360
        
        return angle_data, slope_data
    
    def calculate_display_angle(self):
        """Calculate angle in display coordinates (what you see)"""
        # Get transformation from data to display
        trans = self.ax.transData
        
        # Transform endpoints to display coordinates (pixels)
        p1_display = trans.transform((self.x1, self.y1))
        p2_display = trans.transform((self.x2, self.y2))
        
        # Calculate differences in pixels
        dx_pixels = p2_display[0] - p1_display[0]
        dy_pixels = p2_display[1] - p1_display[1]
        
        if dx_pixels == 0:  # Vertical in display
            return 90.0 if dy_pixels > 0 else -90.0
        
        # Slope in pixels
        slope_display = dy_pixels / dx_pixels
        
        # Angle in degrees
        angle_display = math.degrees(math.atan(slope_display))
        
        # Adjust for quadrant
        if dx_pixels < 0:
            angle_display += 180
        elif dy_pixels < 0:
            angle_display += 360
        
        # Normalize to 0-360
        angle_display = angle_display % 360
        
        return angle_display, slope_display
    
    def calculate_axis_ratio(self):
        """Get the axis scaling ratio"""
        # Get scaling at the midpoint of the line
        x_mid = (self.x1 + self.x2) / 2
        y_mid = (self.y1 + self.y2) / 2
        
        trans = self.ax.transData
        delta = 1e-6
        
        p0 = trans.transform((x_mid, y_mid))
        px = trans.transform((x_mid + delta, y_mid))
        py = trans.transform((x_mid, y_mid + delta))
        
        x_scale = abs(px[0] - p0[0]) / delta  # pixels per x-unit
        y_scale = abs(py[1] - p0[1]) / delta  # pixels per y-unit
        
        if y_scale == 0:
            return float('inf')
        
        ratio = x_scale / y_scale
        return ratio, x_scale, y_scale
    
    def get_corrected_visual_angle(self):
        """
        Calculate what the visual angle SHOULD be based on data angle
        and axis ratio. This is the angle you would measure if you 
        printed the plot and used a protractor.
        """
        angle_data, slope_data = self.calculate_data_angle()
        ratio, x_scale, y_scale = self.calculate_axis_ratio()
        
        # The visual slope is affected by the axis ratio
        # visual_slope = data_slope / ratio
        visual_slope = slope_data / ratio
        
        # Calculate visual angle
        visual_angle = math.degrees(math.atan(visual_slope))
        
        # Adjust quadrant (simplified - assumes dx > 0)
        if visual_slope < 0:
            visual_angle += 360
        
        return visual_angle % 360, visual_slope
    
    def print_analysis(self):
        """Print comprehensive analysis"""
        angle_data, slope_data = self.calculate_data_angle()
        angle_display, slope_display = self.calculate_display_angle()
        ratio, x_scale, y_scale = self.calculate_axis_ratio()
        angle_corrected, visual_slope = self.get_corrected_visual_angle()
        
        print("\n" + "="*60)
        print("LINE ANGLE ANALYSIS")
        print("="*60)
        
        print(f"\nLine coordinates:")
        print(f"  From: ({self.x1:.2f}, {self.y1:.2f})")
        print(f"  To:   ({self.x2:.2f}, {self.y2:.2f})")
        print(f"  Δx (data): {self.x2 - self.x1:.2f}")
        print(f"  Δy (data): {self.y2 - self.y1:.2f}")
        
        print(f"\nData coordinates (mathematical):")
        print(f"  Slope: {slope_data:.4f}")
        print(f"  Angle: {angle_data:.2f}°")
        
        print(f"\nDisplay coordinates (pixels):")
        print(f"  X scaling: {x_scale:.2f} pixels per unit")
        print(f"  Y scaling: {y_scale:.2f} pixels per unit")
        print(f"  Axis ratio (x/y): {ratio:.4f}")
        print(f"  Display slope: {slope_display:.4f}")
        print(f"  Display angle: {angle_display:.2f}°")
        
        print(f"\nVisual appearance:")
        print(f"  Corrected visual slope: {visual_slope:.4f}")
        print(f"  Corrected visual angle: {angle_corrected:.2f}°")
        
        print(f"\nDifference:")
        print(f"  Data vs Display angle: {abs(angle_data - angle_display):.2f}°")
        print(f"  Data vs Corrected visual: {abs(angle_data - angle_corrected):.2f}°")
        
        # Check if they match (equal aspect)
        if abs(ratio - 1.0) < 0.01:
            print("\n✓ Aspect ratio is ~1: Data angle matches visual angle")
        else:
            print(f"\n⚠ Aspect ratio is {ratio:.2f}:1")
            print("  Data angle ≠ Visual angle")
        
        print("="*60)

# Example usage
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Example 1: Line with slope 1, different aspect ratios
print("EXAMPLE 1: Line with slope 1 (45° in data)")
print("-" * 40)

# Left: Auto aspect (stretched)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 5)  # Different ranges = different scaling
ax1.set_aspect('auto')
ax1.grid(True)
ax1.set_title("Aspect: 'auto'\nX: 0-10, Y: 0-5")

analyzer1 = LineAngleAnalyzer(ax1)
analyzer1.draw_line(1, 1, 6, 4)  # Slope = (4-1)/(6-1) = 3/5 = 0.6
analyzer1.print_analysis()

# Right: Equal aspect
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 5)
ax2.set_aspect('equal')  # Forces same scaling
ax2.grid(True)
ax2.set_title("Aspect: 'equal'\nX: 0-10, Y: 0-5")

analyzer2 = LineAngleAnalyzer(ax2)
analyzer2.draw_line(1, 1, 6, 4)
analyzer2.print_analysis()

plt.tight_layout()
plt.show()