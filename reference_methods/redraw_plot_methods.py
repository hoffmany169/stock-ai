
## 方法1: 使用 clear() 或 cla() 清空整个坐标轴
def update_chart(self, new_data):
    """
    清空当前图形并绘制新数据
    """
    # 清空整个坐标轴
    self.ax_price.clear()
    
    # 更新数据
    self.stock_data = new_data
    self.dates_mpl = mdates.date2num(self.stock_data['Date'])
    
    # 重新绘制
    self.plot_price_chart()
    
    # 重新格式化
    self.format_chart()
    
    # 重绘
    self.fig.canvas.draw_idle()

## 方法2: 只移除特定图形对象
def update_price_line(self, new_data):
    """
    只更新价格线，保留其他图形元素
    """
    # 移除原有的价格线
    if hasattr(self, 'price_line'):
        self.price_line.remove()
    
    # 更新数据
    self.stock_data = new_data
    self.dates_mpl = mdates.date2num(self.stock_data['Date'])
    
    # 绘制新的价格线
    self.price_line, = self.ax_price.plot(
        self.dates_mpl,
        self.stock_data['Close'],
        color=self.styles['colors']['price_line'],
        linewidth=self.styles['linewidths']['price_line'],
        label='收盘价',
        zorder=5
    )
    
    # 重绘
    self.fig.canvas.draw_idle()    


## 方法3: 使用列表跟踪所有图形对象
class StockChartPlotter:
    def __init__(self, stock_data, figsize=(14, 10)):
        # ... 其他初始化代码 ...
        self.plot_objects = {
            'price_line': None,
            'volume_bars': None,
            'ma_lines': [],
            'markers': [],
            'annotations': []
        }
    
    def clear_all_plots(self):
        """清空所有图形对象"""
        # 移除价格线
        if self.plot_objects['price_line']:
            self.plot_objects['price_line'].remove()
            self.plot_objects['price_line'] = None
        
        # 移除所有移动平均线
        for line in self.plot_objects['ma_lines']:
            line.remove()
        self.plot_objects['ma_lines'] = []
        
        # 移除所有标记
        for marker in self.plot_objects['markers']:
            marker.remove()
        self.plot_objects['markers'] = []
        
        # 移除所有注释
        for ann in self.plot_objects['annotations']:
            ann.remove()
        self.plot_objects['annotations'] = []
        
        # 重绘
        if self.fig:
            self.fig.canvas.draw_idle()
    

    def update_with_new_data(self, new_data):
        """
        用新数据完全更新图表
        """
        # 清空所有图形
        self.clear_all_plots()
        
        # 更新数据
        self.stock_data = new_data
        self.dates_mpl = mdates.date2num(self.stock_data['Date'])
        
        # 重新绘制主图
        self.plot_price_chart()
        
        # 重新绘制交易量图（如果有）
        if hasattr(self, 'ax_volume'):
            self.ax_volume.clear()
            _, volume_colors = self.calculate_price_change()
            self.plot_volume_chart(volume_colors)
        
        # 重新格式化
        self.format_chart()
        
        # 重绘
        self.fig.canvas.draw_idle()


## 方法4: 动态更新数据（最高效）
def update_data_dynamically(self, new_data):
    """
    动态更新数据而不重新创建图形对象（最高效）
    """
    # 更新内部数据
    self.stock_data = new_data
    new_dates_mpl = mdates.date2num(self.stock_data['Date'])
    
    # 更新价格线的数据
    if hasattr(self, 'price_line'):
        self.price_line.set_data(new_dates_mpl, self.stock_data['Close'])
    
    # 更新x轴范围
    self.ax_price.set_xlim(new_dates_mpl[0], new_dates_mpl[-1])
    
    # 更新y轴范围
    y_min = self.stock_data['Close'].min() * 0.95
    y_max = self.stock_data['Close'].max() * 1.05
    self.ax_price.set_ylim(y_min, y_max)
    
    # 如果有交易量图，也需要更新
    if hasattr(self, 'ax_volume'):
        # 清除旧的柱状图
        self.ax_volume.clear()
        
        # 重新绘制交易量图
        _, volume_colors = self.calculate_price_change()
        self.plot_volume_chart(volume_colors)
        
        # 更新x轴范围
        self.ax_volume.set_xlim(new_dates_mpl[0], new_dates_mpl[-1])
    
    # 重绘
    self.fig.canvas.draw_idle()


## 方法5: 完整的更新系统
def update_chart_complete(self, new_data, update_type='all'):
    """
    完整的图表更新系统
    
    参数:
    ----------
    new_data : DataFrame
        新的股票数据
    update_type : str
        'all': 更新所有内容
        'price': 只更新价格线
        'volume': 只更新交易量
        'both': 更新价格和交易量
    """
    
    # 保存旧的数据范围用于动画效果
    old_dates = self.dates_mpl.copy() if hasattr(self, 'dates_mpl') else None
    
    # 更新数据
    self.stock_data = new_data
    self.dates_mpl = mdates.date2num(self.stock_data['Date'])
    
    # 根据更新类型执行不同的更新
    if update_type in ['all', 'price']:
        # 更新价格线
        if hasattr(self, 'price_line'):
            self.price_line.set_data(self.dates_mpl, self.stock_data['Close'])
        else:
            self.plot_price_chart()
    
    if update_type in ['all', 'volume'] and hasattr(self, 'ax_volume'):
        # 更新交易量图
        self.ax_volume.clear()
        _, volume_colors = self.calculate_price_change()
        self.plot_volume_chart(volume_colors)
    
    if update_type == 'both':
        # 同时更新价格和交易量
        if hasattr(self, 'price_line'):
            self.price_line.set_data(self.dates_mpl, self.stock_data['Close'])
        else:
            self.plot_price_chart()
        
        if hasattr(self, 'ax_volume'):
            self.ax_volume.clear()
            _, volume_colors = self.calculate_price_change()
            self.plot_volume_chart(volume_colors)
    
    # 更新轴范围
    for ax in [self.ax_price, self.ax_volume] if hasattr(self, 'ax_volume') else [self.ax_price]:
        if ax:
            ax.set_xlim(self.dates_mpl[0], self.dates_mpl[-1])
            
            if ax == self.ax_price:
                y_min = self.stock_data['Close'].min() * 0.95
                y_max = self.stock_data['Close'].max() * 1.05
                ax.set_ylim(y_min, y_max)
    
    # 如果有动画效果，可以添加过渡
    if old_dates is not None and len(old_dates) == len(self.dates_mpl):
        self.animate_transition(old_dates)
    else:
        self.fig.canvas.draw_idle()

def animate_transition(self, old_dates):
    """
    添加简单的动画过渡效果
    """
    import matplotlib.animation as animation
    
    def animate(frame):
        # 计算插值因子
        t = frame / 20.0
        
        # 插值日期（如果需要）
        interp_dates = old_dates * (1 - t) + self.dates_mpl * t
        
        # 更新价格线
        if hasattr(self, 'price_line'):
            self.price_line.set_data(interp_dates, self.stock_data['Close'])
        
        return [self.price_line] if hasattr(self, 'price_line') else []
    
    # 创建动画
    anim = animation.FuncAnimation(
        self.fig, animate, frames=20, 
        interval=50, blit=True, repeat=False
    )
    
    # 确保动画被保存
    self.fig.canvas.draw_idle()
    return anim


## 方法6: 使用 set_visible 控制显示
def toggle_data_display(self, show_original=True, show_new=True):
    """
    通过设置可见性来切换不同数据集的显示
    """
    # 存储多条数据线
    if not hasattr(self, 'data_lines'):
        self.data_lines = {}
    
    if show_original and hasattr(self, 'price_line'):
        self.price_line.set_visible(True)
    elif hasattr(self, 'price_line'):
        self.price_line.set_visible(False)
    
    # 显示新数据线
    line_key = 'new_data_line'
    if show_new:
        if line_key in self.data_lines:
            self.data_lines[line_key].set_visible(True)
        else:
            # 创建新的数据线
            new_line, = self.ax_price.plot(
                self.dates_mpl,
                self.stock_data['Close'] * 1.1,  # 示例：新数据
                color='red',
                linestyle='--',
                label='新数据',
                alpha=0.7
            )
            self.data_lines[line_key] = new_line
    elif line_key in self.data_lines:
        self.data_lines[line_key].set_visible(False)
    
    self.fig.canvas.draw_idle()

########################## 实际使用示例：##################
# 创建初始图表
data1 = create_sample_stock_data(days=50)
plotter = StockChartPlotter(data1)
plotter.create_plot()
plotter.show()

# 方法1: 完全清空并重绘
new_data = create_sample_stock_data(days=60)
plotter.update_chart_complete(new_data, update_type='all')

# 方法2: 只更新价格线
new_data2 = create_sample_stock_data(days=70)
plotter.update_chart_complete(new_data2, update_type='price')

# 方法3: 动态更新（最平滑）
for i in range(10):
    # 模拟实时数据更新
    new_data = create_sample_stock_data(days=50 + i)
    plotter.update_data_dynamically(new_data)
    plt.pause(0.5)  # 暂停0.5秒