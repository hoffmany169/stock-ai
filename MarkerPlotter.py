from StockChartPlotter import StockChartPlotter


class MarkerPlotter(StockChartPlotter):
    """
    This class is responsible for plotting markers on the stock chart. It inherits from StockChartPlotter and provides methods to plot markers and extract curve vertices.
    """
    def __init__(self, ax, stock_model, window_width=10):
        """
        Initializes the MarkerPlotter with the given parameters.
        :param ax: The matplotlib axes to plot on.
        :param symbol: The stock symbol to plot.
        :param stock_model: The stock model containing the data to plot.
        :param window_width: The width of the window (x-axis) for calculating vertices (default is 10).
        """
        super().__init__(stock_model.ticker_symbol, stock_model)
        self._ax = ax
        self._stock_model = stock_model
        self._window_width = window_width
        self._markers = []

    def plot_markers(self, markers):
        self.plot(self._ax)

    def extract_curve_vertices(self, feature='Close'):
        # Implement the logic to extract extrem points from the stock chart curve
        feature_data = self._stock_model.loaded_data[feature]
        size = len(feature_data)
        for idx in range(0, size - self._window_width, self._window_width):
            value = feature_data[idx]
            window_values = feature_data[idx - self._window_width:idx + self._window_width + 1]
            if value == window_values.max():
                self._markers.append((idx, value, 1))
            if value == window_values.min():
                self._markers.append((idx, value, 0))
    
    def plot(self, ax):
        if len(self._markers) == 0:
            self.extract_curve_vertices()
        for marker in self._markers:
            idx, value, marker_type = marker
            if marker_type == 1:
                ax.plot(value, 'ro')  # Plot maxima as red circles
            else:
                ax.plot(value, 'go')  # Plot minima as green circles

    def create_plot(self):
        pass
    def on_hover_info(self, *args):
        pass
    def on_leave_info(self, *args):
        pass
