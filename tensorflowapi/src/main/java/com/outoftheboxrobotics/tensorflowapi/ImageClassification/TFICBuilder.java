package com.outoftheboxrobotics.tensorflowapi.ImageClassification;

import com.outoftheboxrobotics.tensorflowapi.ObjectDetection.TensorObjectDetector;
import com.qualcomm.robotcore.hardware.HardwareMap;
import com.qualcomm.robotcore.util.RobotLog;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.IOException;

public class TFICBuilder {
    private final HardwareMap map;
    private final String modelName;
    private boolean quantized;
    private final Interpreter.Options options;
    private String[] labels;
    private int numRecognitions;

    public TFICBuilder(HardwareMap map, String modelName){
        this.map = map;
        this.modelName = modelName;
        quantized = false;
        options = new Interpreter.Options();
        labels = new String[0];
        options.setCancellable(true);
        this.numRecognitions = 10;
    }

    public TFICBuilder setQuantized(boolean quantized){
        this.quantized = quantized;
        return this;
    }

    public TFICBuilder setLabels(String... labels){
        this.labels = labels;
        return this;
    }

    public TFICBuilder setNumThreads(int numThreads){
        this.options.setNumThreads(numThreads);
        return this;
    }

    public TFICBuilder keepTopKResults(int numResults){
        this.numRecognitions = numResults;
        return this;
    }

    public TFICBuilder useXNNPack(boolean xnnPack){
        this.options.setUseXNNPACK(xnnPack);
        return this;
    }

    public TFICBuilder useNNAPI(){
        this.options.setUseNNAPI(true);
        return this;
    }

    public TFICBuilder allowBufferHandleOutput(){
        this.options.setAllowBufferHandleOutput(true);
        return this;
    }

    public TFICBuilder useGPUAcceleration(){
        CompatibilityList compatList = new CompatibilityList();

        if(compatList.isDelegateSupportedOnThisDevice()){
            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
        } else {
            RobotLog.addGlobalWarningMessage("WARNING! GPU Acceleration is NOT supported on this device. Disabling GPU Acceleration.");
        }
        return this;
    }

    public TensorImageClassifier build() throws IOException {
        return new TensorImageClassifier(map, modelName, quantized, options, labels, numRecognitions);
    }
}
