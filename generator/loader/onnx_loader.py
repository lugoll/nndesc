import onnx

# Constants for layer description
CONV = 'conv'
MAXPOOL = 'maxpool'
DENSE = 'dense'
FLATTEN = 'flatten'
LSTM = 'lstm'
GRU = 'gru'
# Activation Functions
RELU = 'relu'
SOFTMAX = 'softmax'
SIGMOID = 'sigmoid'
LOGSOFTMAX = 'logsoftmax'

ACTIVATION_FN = {
    RELU,
    SOFTMAX,
    SIGMOID,
    LOGSOFTMAX
}


class GraphExtractor(object):
    graph_structure = []
    ops_map = {}

    _var_dims = None
    _skipped_nodes = []

    def __init__(self, onnx_object, onnx_mode='training'):
        self.extract_all_var_dims(onnx_object)
        self.parse_onnx(onnx_object, onnx_mode)

    @property
    def supported_ops(self):
        return self.ops_map.keys()

    def parse_onnx(self, onnx_object, onnx_mode):
        self.graph_structure = None

    def find_node_parser(self, op_type):
        if op_type not in self.ops_map.keys():
            return None
        else:
            op_name = self.ops_map[op_type]
            return getattr(self, f'parse_{op_name}', op_name)

    def extract_all_var_dims(self, onnx_object, with_params=True):
        var_dims = {}
        if with_params:
            for var in onnx_object.graph.initializer:
                var_dims[var.name] = var.dims
        else:
            pass

        self._var_dims = var_dims

    # Some Common Parse Methods between PyTorch and Keras ONNX Representation
    def parse_lstm(self, node):
        assert node.op_type == 'LSTM'
        params = {
            'type': 'lstm',
            # Set default for direction if not in attributes
            'direction': 'unidirectional'
        }

        for attr in node.attribute:
            # Extract INTS
            if attr.name in ('hidden_size'):
                params[attr.name] = getattr(attr, 'i')
            elif attr.name in ('direction'):
                params[attr.name] = getattr(attr, 's').decode()
        return params

    def parse_gru(self, node):
        assert node.op_type == 'GRU'
        params = {
            'type': 'gru',
            # Set default for direction if not in attributes
            'direction': 'unidirectional'
        }

        for attr in node.attribute:
            # Extract INTS
            if attr.name in ('hidden_size'):
                params[attr.name] = getattr(attr, 'i')
            elif attr.name in ('direction'):
                params[attr.name] = getattr(attr, 's').decode()
        return params

    def parse_maxpool(self, node):
        assert node.op_type == 'MaxPool'
        params = {'type': 'maxpool'}

        for attr in node.attribute:
            # Extract INTS
            if attr.name in ('kernel_shape', 'strides'):
                params[attr.name] = getattr(attr, 'ints')

        return params


class PyTorchGraphExtractor(GraphExtractor):
    ops_map = {
        'Conv': CONV,
        'MaxPool': MAXPOOL,
        'Relu': RELU,
        'Flatten': FLATTEN,
        'Gemm': DENSE,
        'Softmax': SOFTMAX,
        'LSTM': LSTM,
        'Sigmoid': SIGMOID,
        'GRU': GRU,
        'LogSoftmax': LOGSOFTMAX,
    }

    def __init__(self, onnx_object, onnx_mode='training'):
        super().__init__(onnx_object, onnx_mode)

    def parse_onnx(self, onnx_object, onnx_mode):
        self.graph_structure = []
        for node in onnx_object.graph.node:
            # print(node.op_type)
            parser = self.find_node_parser(node.op_type)
            if type(parser) is str:
                self.graph_structure.append((node.name, {'type': parser}))

            if callable(parser):
                self.graph_structure.append((node.name, parser(node)))

    def parse_conv(self, node):
        assert node.op_type == 'Conv'
        params = {'type': 'conv'}

        # Extract Filter Size via bias size
        for in_val in node.input:
            if 'bias' in in_val.split('.'):
                params['filter'] = self._var_dims[in_val][0]

        for attr in node.attribute:
            # Extract INTS
            if attr.name in ('kernel_shape', 'strides', 'pads'):
                params[attr.name] = getattr(attr, 'ints')

        return params

    def parse_dense(self, node):
        assert node.op_type == 'Gemm', node.op_type
        params = {'type': 'dense'}

        # Extract Input/output vie weight/bias size
        for in_val in node.input:
            if 'bias' in in_val.split('.'):
                params['output'] = self._var_dims[in_val][0]
            if 'weight' in in_val.split('.'):
                # Second Value probably always the input
                params['input'] = self._var_dims[in_val][1]

        return params

    def parse_lstm(self, node):
        assert node.op_type == 'LSTM'
        params = {
            'type': 'lstm',
            # Set default for direction if not in attributes
            'direction': 'unidirectional'
        }

        for attr in node.attribute:
            # Extract INTS
            if attr.name in ('hidden_size'):
                params[attr.name] = getattr(attr, 'i')
            elif attr.name in ('direction'):
                params[attr.name] = getattr(attr, 's').decode()
        return params

    def parse_gru(self, node):
        assert node.op_type == 'GRU'
        params = {
            'type': 'gru',
            # Set default for direction if not in attributes
            'direction': 'unidirectional'
        }

        for attr in node.attribute:
            # Extract INTS
            if attr.name in ('hidden_size'):
                params[attr.name] = getattr(attr, 'i')
            elif attr.name in ('direction'):
                params[attr.name] = getattr(attr, 's').decode()
        return params


class KerasGraphExtractor(GraphExtractor):
    ops_map = {
        'Conv': CONV,
        'MaxPool': MAXPOOL,
        'Relu': RELU,
        'Reshape': FLATTEN,
        'MatMul_Add': DENSE,
        'Softmax': SOFTMAX,
        'LSTM': LSTM,
        'Sigmoid': SIGMOID,
        'GRU': GRU,
        'LogSoftmax': LOGSOFTMAX,
    }

    def __init__(self, onnx_object, onnx_mode='training'):
        super().__init__(onnx_object, onnx_mode)

    def parse_onnx(self, onnx_object, onnx_mode):
        self.graph_structure = []
        
        previous_nodes = []

        for node in onnx_object.graph.node:
            op_type = node.op_type
            node_name = node.name
            
            # Handling Dense Layer here and maybe other Layer over multiple operations
            if op_type == 'MatMul' or op_type == 'Add':
                previous_nodes.append(node)
            # Currently only Operations about max 2 nodes are supported like Dense with Matmul/Add Ops
            if len(previous_nodes) > 1:
                # Overriding node an op_type
                op_type = "_".join([n.op_type for n in previous_nodes])
                node_name = "_".join([n.name for n in previous_nodes])
                node = previous_nodes
                # Delete previous_nodes
                previous_nodes = []

            parser = self.find_node_parser(op_type)
            if type(parser) is str:
                self.graph_structure.append((node_name, {'type': parser}))

            if callable(parser):
                self.graph_structure.append((node_name, parser(node)))

            # Clean out the None Nodes
            self.graph_structure = [(name, val) for name, val in self.graph_structure if val]



    def parse_conv(self, node):
        assert node.op_type == 'Conv'
        filters = self._var_dims.get(f'{node.name}/ReadVariableOp:0', [None])[0]

        params = {
            'type': 'conv',
            # Extract Filter Size via bias size
            'filter': filters,
            # Set default for Padding if not in attributes
            'pads': [0, 0, 0, 0]
        }

        for attr in node.attribute:
            # Extract INTS
            if attr.name in ('kernel_shape', 'strides', 'pads'):
                params[attr.name] = getattr(attr, 'ints')

        return params

    def parse_dense(self, node):
        assert type(node) is list
        assert len(node) > 1

        return {
            'type': 'dense',
            'input': self._var_dims[f'{node[0].name}/ReadVariableOp:0'][0],
            'output': self._var_dims[f'{node[1].name}/ReadVariableOp:0'][0]
        }

    def parse_flatten(self, node):
        shape_var_name = node.input[1]

        dims = self._var_dims[shape_var_name]
        # A Flatten Layer reduces to maximum 2 Dimensions (batch_size and output)
        if dims[0] > 2:
            return None
        else:
            return {'type': FLATTEN}


if __name__ == '__main__':

    conv_onnx = onnx.load('../../test/models/keras/saved_models/my_lstm.onnx')

    extractor = KerasGraphExtractor(conv_onnx)
    graph = extractor.graph_structure

    pass