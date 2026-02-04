import abc
from chain_data import ChainData, Node
from Common.AutoNumber import AutoIndex

class PLOT_TYPE(AutoIndex):
    MARKER = ()
    ANNOTATION = ()
    GUIDELINE = ()
    ORIGINAL = ()

class IPlotData(abc.ABCMeta):
    @abc.abstractmethod
    def plot(self):
        pass

class Marker(IPlotData):
    def __init__(self, data):
        self._marker = Node(data)

class OriginalCurve(IPlotData):
    def __init__(self, data):
        self._curve = Node(data)

class GuideLine(IPlotData):
    def __init__(self, line_data, line_text):
        self._line = {'line': Node(line_data), 'text': Node(line_text)}

class Annotation(IPlotData):
    def __init__(self, data):
        self._annotation = Node(data)

class PlotLayer(IPlotData):
    def __init__(self):
        self._plot_layers = dict(zip([t for t in PLOT_TYPE], ChainData() * len(PLOT_TYPE)))

    def get_layer_data(self, type:PLOT_TYPE) -> ChainData|None:
        return self._plot_layers[type]

    def add_layer_data(self, type:PLOT_TYPE, data:Node)->any:
        self._plot_layers[type].add(data)
        return data

    def remove_layer_data(self, type:PLOT_TYPE, index:int)->any:
        data = self._plot_layers[type].get(index)
        self._plot_layers[type].remove(index)
        return data

    def clear_layer_data(self, type:PLOT_TYPE)->None:
        self._plot_layers[type].clear()
        # remove from plot???

    def clear_all_layer_data(self)->None:
        for data in self._plot_layers.values():
            data.clear()

    def plot()->bool:
        pass

class PlotData(IPlotData):
    def __init__(self):
        self._plot_data = {}

    def get_layer(self, LayerName:str)->PlotLayer:
        for name, layer in self._plot_data.items():
            if name == LayerName:
                return layer   
        return None

    def add_layer(self, LayerName:str, data:PlotLayer)->bool:
        self._plot_data[LayerName] = data

    def remove_layer(self, LayerName:str)->bool:
        layer = self.get_layer(LayerName)
        if layer:
            layer.clear_all_layer_data()

    def clear_all_layers(self)->None:
        self._plot_data = {}

    def plot()->bool:
        pass