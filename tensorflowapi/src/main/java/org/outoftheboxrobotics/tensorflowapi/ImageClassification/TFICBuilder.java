package org.outoftheboxrobotics.tensorflowapi.ImageClassification;

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

    /**
     * Sets if the model is quantized
     *
     * Quantization of a model processes the model so that it uses integer inputs instead of floating point inputs
     * This speeds up inference on the model, but also (slightly) decreases accuracy
     */
    public TFICBuilder setQuantized(boolean quantized){
        this.quantized = quantized;
        return this;
    }

    /**
     * Sets the labels that the model uses.
     *
     * This is the labels that the model use, like "ring" or "goal"
     */
    public TFICBuilder setLabels(String... labels){
        this.labels = labels;
        return this;
    }

    /**
     * Number of threads that the model will use
     *
     * Generally, it is recommended to use somewhere between 1-4 threads
     * On the Control Hub, it seems the most optimal number of threads is somewhere between 2-3
     * The optimal number of threads may change depending on the model
     */
    public TFICBuilder setNumThreads(int numThreads){
        this.options.setNumThreads(numThreads);
        return this;
    }

    /**
     * How many results should be kept
     *
     * The Image Classifier will automatically keep the top k results, discarding the rest
     */
    public TFICBuilder keepTopKResults(int numResults){
        this.numRecognitions = numResults;
        return this;
    }

    /**
     * Sets if the XNNPack delegate is used
     *
     * XNNPack is a set of neural network operators that are highly optimized for running floating point models on ARM, x86, and WebAssembly
     * This has the potential to run floating point (i.e non-quantized) models faster
     */
    public TFICBuilder useXNNPack(boolean xnnPack){
        this.options.setUseXNNPACK(xnnPack);
        return this;
    }

    /**
     * Uses a NNAPI delegate
     *
     * NNAPI utilizes dedicated hardware accelerators, such as Graphics Processing Units, Digital Signal Processors, or Neural Processing Units to speed up model inference
     * NNAPI requires backend architecture to allow for acceleration
     * At this time it is unknown if the FTC Control Hub allows for NNAPI acceleration
     */
    public TFICBuilder useNNAPI(){
        this.options.setUseNNAPI(true);
        return this;
    }

    /**
     * Sets if the buffer handle output is used
     *
     * By default, when using hardware acceleration, data is copied to the CPU buffer before reading
     * Setting this value to FALSE allows for the device to attempt to read data directly from the hardware accelerated buffers
     * This is not supported on all devices but may give an advantage in inference time when used on a supported device
     */
    public TFICBuilder allowBufferHandleOutput(){
        this.options.setAllowBufferHandleOutput(true);
        return this;
    }

    /**
     * Uses GPU Acceleration
     *
     * NOTE: This is for legacy implementation, usage of NNAPI is high recommended instead of GPU acceleration
     *
     * If the device is capable of GPU (Graphics Processing Unit) Acceleration, it will be used
     * Not all models can be GPU accelerated, and some may throw an error when run on a GPU
     */
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
