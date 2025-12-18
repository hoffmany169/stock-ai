import matplotlib.pyplot as plt
import numpy as np

class AxisRatioCalculator:
    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        
    def get_pixels_per_data_unit(self, at_point=None):
        """
        Get pixels per data unit at a specific point or view center.
        
        Parameters:
        -----------
        at_point : tuple (x, y) or None
            Point in data coordinates. If None, uses view center.
        
        Returns:
        --------
        dict with keys:
            'x_scale': pixels per x-unit
            'y_scale': pixels per y-unit  
            'ratio': x_scale / y_scale
            'aspect': current aspect ratio setting
            'data_ratio': (ylim_range / xlim_range) * aspect
        """
        # Get current aspect ratio setting
        aspect = self.ax.get_aspect()
        
        # Get axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Use specified point or center of view
        if at_point is None:
            x_point = (xlim[0] + xlim[1]) / 2
            y_point = (ylim[0] + ylim[1]) / 2
        else:
            x_point, y_point = at_point
        
        # Get transformation
        trans = self.ax.transData
        
        # Calculate for very small delta to avoid precision issues
        delta = 1e-6
        
        # Transform points
        p_center = trans.transform((x_point, y_point))
        p_x = trans.transform((x_point + delta, y_point))
        p_y = trans.transform((x_point, y_point + delta))
        
        # Calculate scaling factors (pixels per data unit)
        x_scale = abs(p_x[0] - p_center[0]) / delta
        y_scale = abs(p_y[1] - p_center[1]) / delta
        
        # Calculate ratios
        if y_scale != 0:
            ratio_xy = x_scale / y_scale
            ratio_yx = y_scale / x_scale
        else:
            ratio_xy = float('inf')
            ratio_yx = 0
        
        # Data ratio (height/width in data coordinates)
        if x_range != 0:
            data_ratio = y_range / x_range
        else:
            data_ratio = float('inf')
        
        return {
            'x_scale': x_scale,
            'y_scale': y_scale,
            'ratio_xy': ratio_xy,  # x units per y unit in display
            'ratio_yx': ratio_yx,  # y units per x unit in display
            'aspect': aspect,
            'data_ratio': data_ratio,
            'xlim': xlim,
            'ylim': ylim,
            'x_range': x_range,
            'y_range': y_range
        }
    
    def get_display_size(self):
        """Get display size in pixels"""
        bbox = self.ax.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted()
        )
        width_pixels = bbox.width * self.fig.dpi
        height_pixels = bbox.height * self.fig.dpi
        return width_pixels, height_pixels
    
    def convert_data_to_pixels(self, x_data, y_data):
        """Convert data coordinates to pixel coordinates"""
        trans = self.ax.transData
        return trans.transform((x_data, y_data))
    
    def convert_pixels_to_data(self, x_pixel, y_pixel):
        """Convert pixel coordinates to data coordinates"""
        trans = self.ax.transData
        return trans.inverted().transform((x_pixel, y_pixel))
    
    def is_square_pixels(self, tolerance=0.01):
        """Check if x and y scales are equal (square pixels)"""
        info = self.get_pixels_per_data_unit()
        return abs(1.0 - info['ratio_xy']) < tolerance

    def get_visual_angle(self, data_x1, data_y1, data_x2, data_y2):
        """Calculate visual angle of line between two data points"""
        info = self.get_pixels_per_data_unit()
        if data_x2 - data_x1 != 0:
            data_slope = (data_y2 - data_y1) / (data_x2 - data_x1)
            aspect_ratio = info['x_scale'] / info['y_scale']
            visual_angle = np.arctan(data_slope / aspect_ratio) * 180 / np.pi
        else:
            visual_angle = 90.0
        return visual_angle

if __name__ == "__main__":
    # Usage example
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_xlabel('X (units)')
    ax.set_ylabel('Y (units)')
    ax.grid(True)

    calculator = AxisRatioCalculator(ax)

    # Get ratio info
    info = calculator.get_pixels_per_data_unit()
    print("Axis Ratio Information:")
    print(f"X scale: {info['x_scale']:.2f} pixels/unit")
    print(f"Y scale: {info['y_scale']:.2f} pixels/unit")
    print(f"Ratio X/Y: {info['ratio_xy']:.3f}")
    print(f"Ratio Y/X: {info['ratio_yx']:.3f}")
    print(f"Aspect setting: {info['aspect']}")
    print(f"Data ratio (height/width): {info['data_ratio']:.3f}")
    print(f"X range: {info['x_range']:.2f} units")
    print(f"Y range: {info['y_range']:.2f} units")

    # Get display size
    width_px, height_px = calculator.get_display_size()
    print(f"\nDisplay size: {width_px:.0f} × {height_px:.0f} pixels")