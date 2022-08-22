import onnx
import re
from generator.loader.onnx_loader import KerasGraphExtractor, PyTorchGraphExtractor
from generator.selector.selection import Selector, GraphSummary
from generator.lexicalizer.maps import Languages, layer_type_lexicalizer, activation_type_lexicalizer, get_pos_word_map_default, network_type_map
from jinja2 import PackageLoader, Environment


class NoMatchingTemplate(Exception):
    def __init__(self, position):
        self.position = position

    def __str__(self):
        return f'No Template matched for the position "{self.position}"!'


class SimpleLayerLexicalizer(object):
    lexicalized_layers: list = []
    language = Languages.EN

    def __init__(self, graph: GraphSummary, language: Languages = None):
        if language:
            self.language = language

        self.graph = graph
        self.lexicalize_properties()

    def lexicalize_properties(self):
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

    def select_template(self, position: str = 'middle', language: Languages = None) -> str:
        """
        Select Template based on graph_summary, network_type and language save it in object, may reselect for other language
        """
        lang = self.language
        if language:
            lang = language

        base_path = f'SimpleLayer/{lang.value}'

        if position == 'middle':
            path = f'{base_path}/MiddleLayer.jj2'
        elif position == 'first':
            path = f'{base_path}/FirstLayer.jj2'
        elif position == 'last':
            path = f'{base_path}/LastLayer.jj2'
        else:
            raise

        return path

    def render(self) -> str:
        """
        Applies a jinja template to each layer an render this description
        """

        env = Environment()
        env.loader = PackageLoader('generator.lexicalizer', 'templates')
        layer_sentences = []

        _, first_layer = self.graph.properties[0]
        template = env.get_template(self.select_template(position='first'))
        layer_sentences.append(template.render(
            layer=first_layer,
            pos_word=get_pos_word_map_default(1, lang=self.language),
            network_type=network_type_map[self.language.value][self.graph.network_type]
        ))

        for i, (_, layer) in enumerate(self.graph.properties[1:-1]):
            template = env.get_template(self.select_template())
            layer_sentences.append(template.render(
                layer=layer,
                pos_word=get_pos_word_map_default(i+2, lang=self.language)
            ))

        _, last_layer = self.graph.properties[-1]
        template = env.get_template(self.select_template(position='last'))
        layer_sentences.append(template.render(
            layer=last_layer,
            pos_word=get_pos_word_map_default(len(self.graph.properties), lang=self.language)
        ))

        layers_text = ' '.join(layer_sentences)
        return re.sub("\s\s+" , " ", layers_text).replace('\n', '')


if __name__ == "__main__":
    conv_onnx = onnx.load('../../test/models/pytorch/saved_models/my_complex_cnn_2.onnx')
    extractor = PyTorchGraphExtractor(conv_onnx)

    selector = Selector(extractor.graph_structure, propert_num=-1)
    selected_graph_summary = selector.select()

    text_engine = SimpleLayerLexicalizer(selected_graph_summary)
    text = text_engine.render()

    print(text)