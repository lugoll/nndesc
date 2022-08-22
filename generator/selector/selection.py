import onnx

from generator.loader.onnx_loader import ACTIVATION_FN, CONV, LSTM, GRU, MAXPOOL, DENSE, KerasGraphExtractor
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Optional

# Network Types Constants
FFN = 'FeedForwardNetwork'
RNN = 'RecurrentNeuralNetwork'
CNN = 'ConvolutionalNeuralNetwork'


class NoSummaryException(Exception):
    def __str__(self):
        return f'Nothing left to summarize!'


class GraphData(BaseModel):
    structure: List[str]
    end_layer: str
    activation_fn: List[str]
    layer_num: int
    weighted_layer_num: int
    detailed_layers: List[Dict]
    conv_size_reduction: Optional[List[int]] = None
    conv_filter_amount: Optional[List[int]] = None

    def get_layer_pairs(self):
        for i in range(len(self.detailed_layers) - 1):
            yield self.detailed_layers[i], self.detailed_layers[i+1]


class GraphSummary(BaseModel):
    network_type: str
    properties: List

    @property
    def length(self):
        return len(self.properties)

    def __len__(self):
        return self.length


class Summarizer(object):
    """
    Summarizes the given graph
    """
    graph: GraphData = None
    last_summary: GraphSummary = None
    graph_features: List[Tuple[str, List]] = None
    _meta_information: dict = {}

    def __init__(self, graph: GraphData, selector):
        self.graph = graph
        self.selector = selector

    def summarize(self) -> GraphSummary:
        # Layers will be listed and only activations and layers will be summed, for small nets this fits
        properties = [
            # ('LayerNum', self.graph.layer_num)
        ]

        for layer, next_layer in self.graph.get_layer_pairs():
            if layer['type'] in self.graph.activation_fn:
                continue
            elif layer['type'] in [CONV, MAXPOOL, DENSE]:
                if layer['type'] == CONV:
                    layer = self._clean_conv(layer)
                elif layer['type'] == MAXPOOL:
                    layer = self._clean_maxpool(layer)

                if next_layer['type'] in self.graph.activation_fn:
                    layer['activation'] = next_layer['type']

                properties.append(('LayerDetailed', layer))
            else:
                properties.append(('LayerDetailed', layer))

        properties[-1] = 'LayerLastDetailed', properties[-1][1]

        self.last_summary = GraphSummary(
            network_type=self._determine_nn_type(),
            properties=properties
        )

        return self.last_summary

    def extract_graph_features(self):
        keys = {k for t, layer in self.last_summary.properties for k in layer.keys()}
        features = {k: [layer.get(k) for t, layer in self.last_summary.properties] for k in keys }
        self.graph_features = features
        pass

    # Use one, not used summarizer each time, determine summarizer, based on constraints
    def summarize_ext(self) -> GraphSummary:
        properties = []
        self.extract_graph_features()
        # Call Sum Methods
        for key in self.graph_features.keys():
            fn = getattr(self, f'sum_{key}', None)
            if fn:
                # print(f"used {fn} ({key})")
                # Call fn to sum stuff
                if res := fn():
                    # Only append if summary is applied
                    properties.append(res)

        # Add TotalLayer Count
        properties = [('LayerNum', self.graph.weighted_layer_num)] + properties

        # For NetworkType Convolutional Network add data dimension changes
        properties.append(self.sum_dimension_changes())

        # Add OutputLayer
        properties.append(self.last_summary.properties[-1])

        self.last_summary.properties = properties
        return self.last_summary

    def sum_type(self):
        """Aims all Layers and all types"""
        types = dict(self.graph_features)['type']
        types_count = {t: types.count(t) for t in set(types)}
        self._meta_information['types_count'] = types_count
        if MAXPOOL in types:
            types.remove(MAXPOOL)
        summary = {
            'most_used': max(set(types), key=types.count),
        }
        summary['most_used_count'] = types_count[summary['most_used']]

        if self._determine_nn_type() == CNN:
            # It is unlikely that CNN aren't build this way
            # conv/maxpool - flatten - dense
            if types_count.get(CONV, 0) == types_count.get(MAXPOOL, 0):
                summary['maxpool'] = 'equal'
            elif types_count.get(CONV, 0) > types_count.get(MAXPOOL, 0):
                summary['maxpool'] = 'some'

            if DENSE in types:
                summary['last'] = DENSE
                summary['last_count'] = types_count[summary['last']]

        if self._determine_nn_type() == RNN:
            types.remove(summary['most_used'])
            summary['also_used'] = types

        return 'LayerTypeSummary', summary

    def sum_filter(self):
        """Aims Conv Layers, give direction about the network, decreasing and increasing number, will be skipped if no CNN"""
        if self._determine_nn_type() == CNN:
            filters = [el for el in dict(self.graph_features)['filter'] if el]
            direction = {}
            check_increasing = True
            check_decreasing = True
            for p, n in zip(filters[:-1], filters[1:]):
                if p > n and check_decreasing:
                    direction[len(direction)+1] = 'decrease'
                    check_increasing = True
                    check_decreasing = False
                elif p < n and check_increasing:
                    direction[len(direction)+1] = 'increase'
                    check_increasing = False
                    check_decreasing = True

            if len(direction) == 1:
                direction = {
                    'all': direction[1],
                    'from': filters[0],
                    'to': filters[-1]
                }
            elif len(direction) == 0:
                direction = {
                    'all': 'are equal',
                    'amount': filters[0]
                }
            return 'FilterSummary', direction

    def sum_kernel_shape(self):
        """Aims Conv Layers, mostly used, also used, if just one, then give position"""
        if self._determine_nn_type() == CNN:
            # Lexicalize Kernels at this step to hash them for set
            kernels = ['x'.join(str(c) for c in k) for k in dict(self.graph_features)['kernel_shape'] if k]
            kernels_set = set(kernels)
            summary = {}
            if len(set(kernels)) > 2:
                summary['several_used'] = set(kernels)
            elif len(kernels_set) == 2:
                summary = {
                    'most_used': max(kernels_set, key=kernels.count),
                }
                summary['most_used_count'] = kernels.count(summary['most_used'])
                kernels_set.remove(summary['most_used'])
                summary['also_used'] = max(kernels_set, key=kernels.count)
                summary['also_used_count'] = kernels.count(summary['also_used'])
            elif len(set(kernels)) == 1:
                summary = {
                    'only_used': max(kernels_set, key=kernels.count),
                }

            return 'KernelSummary', summary

    def sum_pads(self):
        """Aims Conv Layers, mostly used, also used"""
        NotImplemented

    def sum_output(self):
        """Aims Dense Layers, Todo: find proper solution"""
        NotImplemented

    def sum_activation(self):
        if self._determine_nn_type() == CNN:
            activation = [a for a in dict(self.graph_features)['activation'] if a]
            activation_set = set(activation)
            summary = {}
            if len(set(activation)) > 2:
                summary['several_used'] = set(activation)
            elif len(activation_set) == 2:
                summary = {
                    'most_used': max(activation_set, key=activation.count),
                }
                summary['most_used_count'] = activation.count(summary['most_used'])
                activation_set.remove(summary['most_used'])
                summary['also_used'] = max(activation_set, key=activation.count)
                summary['also_used_count'] = activation.count(summary['also_used'])
            elif len(set(activation)) == 1:
                summary = {
                    'only_used': max(activation_set, key=activation.count),
                }

            return 'ActivationSummary', summary

    def sum_dimension_changes(self):
        """Aims Conv/Maxpool Layers, mostly used, minus last one"""
        if self._determine_nn_type() == CNN:
            maxpool_strides = {str(2) for l in self.graph.detailed_layers if l['type'] == MAXPOOL}
            conv_strides = {str(l.get('strides', 1)) for l in self.graph.detailed_layers if l['type'] == CONV and l.get('strides', 1) > 2}

            summary = {
                'maxpool_downsampling': False,
                'conv_downsampling': False,
            }

            if len(maxpool_strides) > 0:
                summary['maxpool_downsampling'] = True
            if len(conv_strides) > 0:
                summary['conv_downsampling'] = True
                if len(set(conv_strides)) > 1:
                    c_s_str = ', '.join(str(s) for s in conv_strides[:-1]) + f'and {conv_strides[-1]}'
                    summary['conv_strides'] = f'the strides {c_s_str} are'
                else:
                    summary['conv_strides'] = f'a stride of {conv_strides[0]} is'
            return 'DimensionSummary', summary

    def _clean_conv(self, layer_dict: dict) -> dict:
        assert layer_dict['type'] == CONV, layer_dict['type']
        strides = set(layer_dict['strides'])
        padding = set(layer_dict['pads'])
        # Reduce dimensions for Layers
        if len(strides) == 1:
            layer_dict['strides'] = list(strides)[0]
        # Reduce dimensions for Layers
        if len(strides) == 1 and list(strides)[0] == 1:
            del(layer_dict['strides'])
        elif len(strides) == 1:
            layer_dict['strides'] = list(strides)[0]
        # For no Padding
        if len(padding) == 1 and list(padding)[0] == 0:
            del(layer_dict['pads'])
        elif len(padding) == 1:
            layer_dict['pads'] = list(padding)[0]

        return layer_dict

    def _clean_maxpool(self, layer_dict: dict) -> dict:
        assert layer_dict['type'] == MAXPOOL, layer_dict['type']
        strides = set(layer_dict['strides'])
        kernel = set(layer_dict['strides'])
        # Reduce dimensions for Layers
        if len(strides) == 1 and list(strides)[0] == 2:
            del(layer_dict['strides'])
        elif len(strides) == 1:
            layer_dict['strides'] = list(strides)[0]
        # Remove Information for default settings
        if len(kernel) == 1 and list(kernel)[0] == 2:
            del(layer_dict['kernel_shape'])
        elif len(kernel) == 1:
            layer_dict['kernel_shape'] = list(kernel)[0]

        return layer_dict

    def _determine_nn_type(self) -> str:
        if CONV in self.graph.structure:
            return CNN
        elif LSTM in self.graph.structure:
            return RNN
        elif GRU in self.graph.structure:
            return RNN
        else:
            return FFN


class Selector(object):
    """
    Decides which properties are important and calls the summarizer as long as necessary to meet the expectations.
    Decides if layer information should be summarized extensivly based on property_num parameter
    """
    property_num: int = 6  # Six as default for Template-Based approach
    graph: List[Tuple[str, Dict]] = None
    graph_data: GraphData = None

    def __init__(self, graph: List[Tuple[str, Dict]], propert_num: int = 6):
        super(Selector, self).__init__()
        self.graph = graph
        self.property_num = propert_num
        structure = [data['type'] for name, data in graph]
        activation_fn = {layer for layer in structure if layer in ACTIVATION_FN}
        conv_size_reduction = [layer['strides'][0] for name, layer in graph if layer['type'] in (CONV, MAXPOOL)]
        conv_filter_amount = [layer['filter'] for name, layer in graph if layer['type'] == CONV]
        self.graph_data = GraphData(
            structure=structure,
            end_layer=structure[-1],
            activation_fn=activation_fn,
            conv_size_reduction=conv_size_reduction,
            conv_filter_amount=conv_filter_amount,
            layer_num=len(structure),
            weighted_layer_num=len([l for l in structure if l in (CONV, DENSE, LSTM, GRU)]),
            detailed_layers=[value for n, value in graph]
        )
        self.summarizer = Summarizer(self.graph_data, self)

    def select(self):
        # Call Summarizer until expectations are met
        summary = self.summarizer.summarize()
        # If property_num is negative, don't use the extended summary
        if len(summary) > self.property_num > 0:
            summary = self.summarizer.summarize_ext()

        return summary


if __name__ == '__main__':
    conv_onnx = onnx.load('../../test/models/keras/saved_models/vgg16.onnx')

    extractor = KerasGraphExtractor(conv_onnx)

    selector = Selector(extractor.graph_structure, propert_num=-1)

    result = selector.select()

    print(result)
    print(len(result))

    pass
