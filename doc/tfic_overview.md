# TensorflowImageClassifier Overview

### What is image classification?

Image classification is an AI task where a model takes in an image and tries to determine a label for the image.

This is useful for tasks where you need to know something about what is in an image, such as if there are 0, 1, or 4 objects in a stack or if an intake has cubes or balls in it.

### What models are supported?

Most Tensorflow 2.0 models can be run using this API.

Tensorflow 1.0 models are theoretically supported with back compatibility, but I would highly suggest using Tensorflow 2.0 models for accuracy and inference time reasons

### Creating a TensorImageClassifier instance

TensorImageClassifier instances are created using the 'TFICBuilder'

Here is the most basic usage
```java
TensorImageClassifier tfic = new TFICBuilder(hardwareMap, "model.tflite", "Label 1", "Label 2").build();
```

In this example, we pass to it the hardwareMap, the name of our model in the **assets** folder of the FtcRobotController module ("model.tflite"), and the label(s) the model uses ("Label 1" and "Label 2")

However, the following options are also available in the builder

**keepTopKResults(int numResults)**: Number of results to keep. By default, this should be greater then or equal to the length of the labels, but it can be lowered if you only want a certain number of results each time.

**useXNNPack(boolean xnnPack)**: Sets if the model should use XNNPack. XNNPack is a set of neural network operations highly optimized for running floating-point models

XNNPack works on ARM, x86, and WebAssembly systems, and can make running non-quantized models faster

See here for more information https://github.com/google/XNNPACK/

**useNNAPI()**: Use the NNAPI delegate. NNAPI automagically uses hardware accelerators like GPUS, DSPs, or NPUs to speed up model inference

Theoretically this should be used *instead* of GPU Acceleration, but requires backend architecture to work correctly, so it may not work on all devices

**Update** Testing done on September 10, 2021 indicates that NNAPI does not make a difference on the Rev Robotics Control Hub. Phones may benefit from this option, but the Control Hub will not, due to the GPU not supporting OpenCL or OpenGL 3.0

See here for more information https://developer.android.com/ndk/guides/neuralnetworks/

In this example two parameters are needed, the hardwareMap, and the name of the model (in this example it is "model.tflite")

**allowBufferHandleOutput(boolean allow)**: On some devices the result of the hardware accelerated model can be read directly from the accelerator's buffer. In this case allowBufferHandleOutput can be set to FALSE

This gives a slight improvement to run time, but is highly experimental and may not be stable

It is not necessary to use this on the FTC Control Hub, since GPU acceleration is not supported

**useGPUAcceleration()**: Uses GPU acceleration to run models. ***THIS IS A LEGACY METHOD AND SHOULD NOT BE NORMALLY USED, USE NNAPI INSTEAD***

GPUs are usually able to run models much faster then a CPU, so they can provide a speed improvement.

NNAPI can use GPU acceleration, so this method should mostly be used for devices where using NNAPI isn't possible

In addition, this method can actually *force* the system to use a GPU over the CPU, if that is needed.

However, *only* using a GPU has the potential to cause crashes from unsupported operations, or even slow down the model due to bottlenecks

This option **may be required** to use NNAPI acceleration on some older phones. It requires a GPU with OpenCL or OpenGL 3.0. Due to this requirement, GPU Acceleration is **not** supported on the Control Hub and may cause instability if attempted to be used anyway


**setLabels(String... labels)**: ***DEPRICATED*** Sets the labels that the model will assign to the detections. This should match the labels that the model is trained on. For example, if you trained a model to detect "cubes" and "balls", you should pass "cubes", and "balls" here

Order matters! The order of the labels here should match the order of the labels used to train

This has been depricated, pass labels in the constructor instead

### How do I run the model?

The model is run using the recognize() method
This method has one parameter, the mat to run the model on

The method returns a list of "recognitions" that the model things the image contains

Each recognition has the following attributes

**ID**: A unique id for each recognition

**Title**: The label of the recognition

**Confidence**: The confidence level from 0-100% of the detection, basically how confident is the model the image fits the label
