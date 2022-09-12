import onnx

from generator.loader.onnx_loader import KerasGraphExtractor, PyTorchGraphExtractor
from generator.selector.selection import Selector
from generator.lexicalizer.template import TemplateEngine


def run_pipeline(onnx_filepath, model_origin='keras'):
    conv_onnx = onnx.load(onnx_filepath)
    if model_origin == 'pytorch':
        extractor = PyTorchGraphExtractor(conv_onnx)
    elif model_origin == 'keras':
        extractor = KerasGraphExtractor(conv_onnx)
    else:
        return None

    selector = Selector(extractor.graph_structure)
    selected_graph_summary = selector.select()

    text_engine = TemplateEngine(selected_graph_summary)
    text = text_engine.render()
    return text
