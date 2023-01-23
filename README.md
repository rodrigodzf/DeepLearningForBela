# DeepLearningForBela

This repository contains a wrapper library and samples that can be used as a starting point for running deep learning models on the Bela plattform. It also includes some benchmarking tools and utilities to faciliate protoyping.

Currently it supports models created with Pytorch and Tensorflow. All the models are loaded dynamically at runtime and optimized with the desired frontend/backend configuration.

## Requirements

This repository asssumes you have available the precompiled libraries ArmNN and TensorflowLite for Arm. The instructions for compiling these frameworks can be found [here](https://www.tensorflow.org/lite/guide/build_arm) for TFLite and [here](https://developer.arm.com/documentation/102649/2202/Overview) for ArmNN.

Since compiling these libraries on Bela can become diffcult it is recommended to use a crosscompiler. For this you can use [this repository](https://github.com/rodrigodzf/xc-bela-container) to download an image that includes the precompiled libraries. Optionally you can also build the image from scratch (following the instructions for each framework). Note that the currently provided TFLite was compiled without XNNPACK due to this [issue](https://github.com/abhiTronix/raspberry-pi-cross-compilers/issues/90).

## Building

To build the included samples and libraries with cmake use:

```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=Toolchain.cmake
cd build
cmake --build .
```

## Samples

The samples assume you have a pretrained network exported in the `.tflite` or `.onnx` formats. There are some python scripts included that can be used to train and export such networks.

wavetable - This is a simple generator that feds a sawtooth (or a sine) wave to a neural network. It is meant to be used in combination with a MLP and a TCN.
linear - This is a simple linear Tensorflow Lite model. This sample does not use any of the frontends provided. The model can be generated using [this code](https://github.com/pelinski/bela-tflite-example/blob/main/python/model.ipynb) or downloaded from [here](https://github.com/pelinski/bela-tflite-example/blob/main/bela-tflite-linear/model.tflite).

<!-- threaded - A threaded sample that shows how to run inference neural network on a separate thread  -->

## Frontend and backend selection

There are several frameworks for deploying neural networks on embedded devices such as the Bela or Raspberry Pi.

Currently, the following frontends are available:

- Tensorflow Lite
- ArmNN
- RTNeural

Each Frontend supports different formats and also different options for optimization and backend selection.

### Tensorflow Lite

Tensorflow Lite uses [delegates](https://www.tensorflow.org/lite/performance/delegates) to accelerate certain operations on different hardware. By default, operators are optimized for Neon on ARM devices and the [default delegate is XNNPACK](https://blog.tensorflow.org/2020/07/accelerating-tensorflow-lite-xnnpack-integration.html).

ArmNN also provides a custom delegate that can be used with TFLite.

### ArmNN

ArmNN provides 3 backends: CpuRef, CpuAcc and GpuAcc. However, is is also possible to [implement custom backends](https://arm-software.github.io/armnn/20.02/backends.xhtml).

### RTNeural

RTNeural provides 3 backends, STL, xsimd, and Eigen. By default Eigen is used, more information about the backends and their selection can be found [here](https://github.com/jatinchowdhury18/RTNeural#choosing-a-backend)
