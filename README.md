# Neural Network Description (nndesc)

## Description
This project was built during with the purpose to automatically generate textual description for neural networks through there implementation based on established frameworks.
To make as many frameworks compatible as possible, [ONNX](https://github.com/onnx/onnx) with the operation set 13 is used as base representation for the neural network. 
With this, it is possible to generate descriptions like the following for the [VGG16](https://keras.io/api/applications/vgg/) model:

> The Convolutional Neural Network consists of 16 weighted Layer, for which mostly Convolutional Layer and some MaxPool Layer are used . In all Convolutional Layers a kernel size of 3x3 gets applied to the data. Through the whole network, the amount of output filters in the Convolutional Layers increases from 64 to 512 filters. For downsampling only the MaxPool Layers are used. Overall in the network mostly the ReLU activation function is used, but also the Softmax is used 1 times.The last layer in the network is a Dense Layer with the Softmax activation function.

## How to Use
Firstly install the prerequisites:

Afterwards you can run the following code to generate a description:
```python
from generator import run_pipeline
run_pipeline('../test/models/keras/saved_models/vgg16.onnx', model_origin='keras')
```

The ONNX file is generated differently based on the used framework. The documantation of the framework will surely provide more information on to accomplish that.

### Generate Test Files
Due to the size, the onnx files for the test models are not part of this repository but could be generated with the follwing script:
```commandline
cd test/models && ./generate.sh
cd keras/saved_models && ./convert.sh
```

## Project Status
This project was developed as final thesis, thus it is currently just more of a prototype than package for real-world usage. 
There are still some topics which should be improved for further usage of this tool.

## Roadmap
- [ ] improve extraction of the graph data from ONNX
- [ ] recognize some complex structures e. g. residual blocks
- [ ] improve lexicalization (merge similar sentences and add more phrases to select from)
- [ ] build real python package for easier usage maybe add commandline script

## Project structure
`generator/` <br>
The python library for generation

`notebooks` <br>
A collection of notebooks for development, testing and visualization.<br>
Also alternative approaches for the lexicalization can be found here.

`test`: <br>
A collection of test models with onnx files.

## License
[MIT License](LICENSE)
