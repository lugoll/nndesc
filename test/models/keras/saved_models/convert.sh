#!/bin/bash
python -m tf2onnx.convert --saved-model my_cnn.model --output my_cnn.onnx
python -m tf2onnx.convert --saved-model my_ffn.model --output my_ffn.onnx

python -m tf2onnx.convert --saved-model my_lstm.model --output my_lstm.onnx

python -m tf2onnx.convert --saved-model vgg16.model --output vgg16.onnx
python -m tf2onnx.convert --saved-model mobile_net.model --output mobile_net.onnx
python -m tf2onnx.convert --saved-model mobile_net_v3_small.model --output mobile_net_v3_small.onnx
python -m tf2onnx.convert --saved-model resnet.model --output resnet.onnx