import onnx
import re
from generator.loader.onnx_loader import KerasGraphExtractor, PyTorchGraphExtractor
from generator.selector.selection import Selector, GraphSummary
from generator.lexicalizer.maps import Languages, layer_type_lexicalizer, activation_type_lexicalizer
from jinja2 import PackageLoader, Environment


class NoMatchingTemplate(Exception):
    def __str__(self):
        return f'No Template matched the summarization pattern!'


class TemplateEngine(object):
    language = Languages.EN

    selected_template = None
    lexicalized_properties = None
    lexicalized_layers = None

    def __init__(self, graph: GraphSummary, language: Languages = None):
        if language:
            self.language = language

        self.graph = graph
        self.lexicalize_properties()
        self.select_template()

    def render(self) -> str:
        """
        Put the details of the graph summary into the selected template and render it with jinja2
        """
        env = Environment()
        env.loader = PackageLoader('generator.lexicalizer', 'templates')

        template = env.get_template(self.selected_template)

        return re.sub("\s\s+" , " ", template.render(
            layers=self.lexicalized_layers,
            property=dict(self.lexicalized_properties)
        ).replace('\n', ''))

    def select_template(self, language: Languages = None) -> str:
        """
        Select Template based on graph_summary, network_type and language save it in object, may reselect for other language
        """
        lang = self.language
        if language:
            lang = language

        base_path = f'{self.graph.network_type}/{lang.value}'

        information_types = {t for t, e in self.graph.properties}

        if set(information_types) == {'LayerDetailed', 'LayerLastDetailed'}:
            path = f'{base_path}/only_layers.jj2'
        elif 'LayerTypeSummary' in information_types:
            path = f'{base_path}/summed_network.jj2'
        else:
            raise NoMatchingTemplate

        self.selected_template = path

    def lexicalize_properties(self):
        self.lexicalized_properties = []
        self.lexicalized_layers = []
        for t, layer in self.graph.properties:
            # Apply different lexicalizer based on property type
            if t in {'LayerDetailed', 'LayerLastDetailed'}:
                layer['type'] = layer_type_lexicalizer[self.language.value][layer['type']]
                if 'kernel_shape' in layer.keys():
                    layer['kernel_shape'] = 'x'.join([str(x) for x in layer['kernel_shape']])
                if 'activation' in layer.keys():
                    layer['activation'] = activation_type_lexicalizer[self.language.value][layer['activation']]
                self.lexicalized_layers.append(layer)
            elif t == 'LayerTypeSummary':
                # most_used, also_used, last_used, only_used are relevand keywords which could exist and contain types
                if 'most_used' in layer.keys():
                    layer['most_used'] = layer_type_lexicalizer[self.language.value][layer['most_used']]
                if 'last_used' in layer.keys():
                    layer['last_used'] = layer_type_lexicalizer[self.language.value][layer['last_used']]
                if 'only_used' in layer.keys():
                    layer['only_used'] = layer_type_lexicalizer[self.language.value][layer['only_used']]
                if 'also_used' in layer.keys() and type(layer['also_used']) == str:
                    layer['also_used'] = layer_type_lexicalizer[self.language.value][layer['also_used']]
                if 'also_used' in layer.keys() and type(layer['also_used']) == list:
                    layer['also_used'] = [layer_type_lexicalizer[self.language.value][el] for el in layer['also_used']]
                self.lexicalized_properties.append((t, layer))
            elif t == 'ActivationSummary':
                # most_used, also_used, last_used, only_used are relevand keywords which could exist and contain types
                if 'most_used' in layer.keys():
                    layer['most_used'] = activation_type_lexicalizer[self.language.value][layer['most_used']]
                if 'several_used' in layer.keys():
                    layer['several_used'] = [activation_type_lexicalizer[self.language.value][el] for el in layer['several_used']]
                if 'only_used' in layer.keys():
                    layer['only_used'] = activation_type_lexicalizer[self.language.value][layer['only_used']]
                if 'also_used' in layer.keys() and type(layer['also_used']) == str:
                    layer['also_used'] = activation_type_lexicalizer[self.language.value][layer['also_used']]
                if 'also_used' in layer.keys() and type(layer['also_used']) == list:
                    layer['also_used'] = [activation_type_lexicalizer[self.language.value][el] for el in layer['also_used']]
                self.lexicalized_properties.append((t, layer))
            else:
                self.lexicalized_properties.append((t, layer))


if __name__ == '__main__':
    conv_onnx = onnx.load('../../test/models/pytorch/saved_models/my_complex_cnn_3.onnx')
    extractor = PyTorchGraphExtractor(conv_onnx)

    selector = Selector(extractor.graph_structure)
    selected_graph_summary = selector.select()

    text_engine = TemplateEngine(selected_graph_summary)
    text = text_engine.render()

    print(text)
