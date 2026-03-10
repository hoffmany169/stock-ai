from Common.AutoNumber import AutoIndex

class STYLE(AutoIndex):
    colors = ()
    line_widths = ()
    alphas = ()
    font_sizes = ()
    markers = ()

class PLOT_ELEMENT(AutoIndex):
    title = ()
    price_line = ()
    hover_line = ()
    trend_line = ()
    price_up = ()
    price_down = ()
    volume = ()
    volume_up = ()
    volume_down = ()
    annotation_bg = ()
    grid_color = ()
    axis_label = ()
    annotation = ()
    tick_label = ()
    peaks = ()
    valleys = ()


class PlotStyle:
    def __init__(self):
        self._init_default_styles()

    def _init_default_styles(self):
        """设置图表样式"""
        self.styles = {
            # 颜色配置
            'colors': {
                'price_line': '#1f77b4',        # 股价线颜色
                'price_up': '#2ecc71',          # 上涨颜色
                'price_down': '#e74c3c',        # 下跌颜色
                'volume_up': 'lightgreen',      # 上涨交易量颜色
                'volume_down': 'lightcoral',    # 下跌交易量颜色
                'hover_line': '#e74c3c',        # 悬停线颜色
                'annotation_bg': 'white',       # 注释框背景
                'grid_color': '#ecf0f1',        # 网格颜色
                'horizontal_line' : 'purple',
            },
            # 线型配置
            'line_widths': {
                'price_line': 2,
                'hover_line': 1,
                'trend_line': 2,
            },
            # 透明度
            'alphas': {
                'volume': 0.7,
                'hover_line': 0.5,
                'trend_line': 0.5,
            },
            'line_styles': {
                'solid_line': '-',
                'dash_dot_line': '-.',
                'dotted_line': ':',
                'dashed_line': '--',
            },
            # 字体大小
            'font_sizes': {
                'title': 16,
                'axis_label': 12,
                'annotation': 10,
                'tick_label': 10,
            },
            'markers' : {
                'point' : '.',
                'pixel' : ',',
                'circle' : 'o',
                'down_triangle' : 'v',
                'up triangle' : '^',
                'left triangle' : '<',
                'right triangle' : '>',
                'square' : 's',
                'pentagon' : 'p',
                'star' : '*',
                'hexagon1' : 'h',
                'hexagon2' : 'H',
                'plus' : '+',
                'cross' : 'x',
                'diamond' : 'D',
                'thin diamond' : 'd',
                'vertical bar' : '|',
                'hyphen' : '_'
            }
        }

    def get_setting(self, style:STYLE, element:PLOT_ELEMENT):
        for elem, value in self.styles[style.name].items():
            if element.name == elem:
                return value
            
    def set_setting(self, style:STYLE, element:PLOT_ELEMENT, value):
        for elem in self.styles[style.name].keys():
            if element.name == elem:
                self.styles[style.name][elem] = value
                return
        raise ValueError("No style or plot element is found!")
    
    def get_marker_names(self)->list:
        return list(self.styles['markers'].keys())
    
    def get_marker(self, name):
        for n, v in self.styles['markers'].items():
            if name == n:
                return v
            
    def get_line_style(self, name):
        for n, l in self.styles['line_styles'].items():
            if name == n:
                return l