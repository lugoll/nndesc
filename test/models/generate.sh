#!/bin/bash

# Generate PyTorch ONNX models
mkdir -p pytorch/saved_models/
python pytorch/pos_gru_net.py
python pytorch/mnist_conv_net.py
python pytorch/mnist_complex_conv_net_1.py
python pytorch/mnist_complex_conv_net_2.py
python pytorch/mnist_complex_conv_net_3.py

# Save Keras Models
python keras/textclass_lstm.py
python keras/mnist_conv_net.py
python keras/mnist_feed_forward.py
python keras/mobile_net.py
python keras/resnet.py
python keras/vgg16.py

