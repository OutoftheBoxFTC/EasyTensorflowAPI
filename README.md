# EasyTensorflowAPI

[![](https://jitpack.io/v/OutoftheBoxFTC/EasyTensorflowAPI.svg)](https://jitpack.io/#OutoftheBoxFTC/EasyTensorflowAPI)

An easy to use API to use Tensorflow 2.0 in FTC

FIRST provides a decent Tensorflow API already packaged in the app, but it currently only supports Tensorflow 1.0 and does not expose many backend features for user customizability.

In addition, due to the nature of tensorflow models, being able to see the implementation of Tensorflow allows for people to make much more specific models for FTC use

## Disclaimer

As of right now, the Tensor Image Classification system is UNTESTED, testing will come soon.

## Current Features:
EasyTensorflowAPI currently supports both tensorflow object detection and tensorflow image classification models in the form of a .tflite model
The API supports loading models from the assets folder of FTCRobotController, setting used threads, managing NNAPI delegate usage, GPU acceleration, XNNPack, and more
Models are all run on Bitmap inputs, allowing you to use a variety of camera input options, from the built in webcam api in the FTC Robot Controller to external solutions like EasyOpenCV

## Documentation:
[TensorImageClassifier](https://github.com/OutoftheBoxFTC/EasyTensorflowAPI/blob/main/doc/tfic_overview.md)

[TensorObjectDetector](https://github.com/OutoftheBoxFTC/EasyTensorflowAPI/blob/main/doc/tfod_overview.md)

How to make a Tensorflow Object Detection Model: WIP

How to make a Tensorflow Image Classification Model: WIP

## Installing
Installation is simple. Just go to build.dependencies.gradle in your project and add the following

```
maven { url 'https://jitpack.io' }
```

in repositories and

```
implementation 'com.github.OutoftheBoxFTC:EasyTensorflowAPI:v1.0.5-Alpha'
```

in dependencies. Then run a gradle sync, and everything should download!

## OnBotJava:
This library can be ported to OnBotJava, however I strongly recommend anyone using this library to use it in Android Studio to avoid a lot of pain that will come from porting it.



## Important Note:
This API is still a work in progress and is subject to change. The basic usage of the API will remain the same for the forseeable future, but new features will be added as teams share feedback on things they want to see. 
