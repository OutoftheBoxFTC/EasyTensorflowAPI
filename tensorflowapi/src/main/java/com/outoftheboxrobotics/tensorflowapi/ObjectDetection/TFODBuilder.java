package com.outoftheboxrobotics.tensorflowapi.ObjectDetection;

import com.qualcomm.robotcore.hardware.HardwareMap;
import com.qualcomm.robotcore.util.RobotLog;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.IOException;

public class TFODBuilder {
    private final HardwareMap map;
    private final String modelName;
    private boolean quantized;
    private final Interpreter.Options options;
    private String[] labels;

    public TFODBuilder(HardwareMap map, String modelName){
        this.map = map;
        this.modelName = modelName;
        quantized = false;
        options = new Interpreter.Options();
        labels = new String[0];
        options.setCancellable(true);
    }

    public TFODBuilder setQuantized(boolean quantized){
        this.quantized = quantized;
        return this;
    }

    public TFODBuilder setLabels(String... labels){
        this.labels = labels;
        return this;
    }

    public TFODBuilder setNumThreads(int numThreads){
        this.options.setNumThreads(numThreads);
        return this;
    }

    public TFODBuilder useXNNPack(boolean xnnPack){
        this.options.setUseXNNPACK(xnnPack);
        return this;
    }

    public TFODBuilder useNNAPI(){
        this.options.setUseNNAPI(true);
        return this;
    }

    public TFODBuilder allowBufferHandleOutput(){
        this.options.setAllowBufferHandleOutput(true);
        return this;
    }

    public TFODBuilder useGPUAcceleration(){
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

    public TensorObjectDetector build() throws IOException {
        return new TensorObjectDetector(map, modelName, quantized, options, labels);
    }
}
