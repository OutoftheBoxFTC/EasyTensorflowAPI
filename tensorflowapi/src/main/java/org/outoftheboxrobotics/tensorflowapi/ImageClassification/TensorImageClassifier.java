package org.outoftheboxrobotics.tensorflowapi.ImageClassification;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;

import com.qualcomm.robotcore.hardware.HardwareMap;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.outoftheboxrobotics.tensorflowapi.TensorProcessingException;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class TensorImageClassifier {
    private final Interpreter interpreter;
    private final boolean quantized;
    private final int width, height;
    private int numRecognitions;

    private TensorImage inputImageBuffer;
    private final TensorBuffer outputProbabilityBuffer;
    private final TensorProcessor probabilityProcessor;

    private final String[] labels;

    protected TensorImageClassifier(HardwareMap map, String modelName, boolean quantized, Interpreter.Options options, String[] labels, int numRecognitions) throws IOException {
        MappedByteBuffer model = loadModelFile(map.appContext.getAssets(), modelName);
        this.quantized = quantized;

        this.interpreter = new Interpreter(model, options);

        this.width = this.interpreter.getInputTensor(0).shape()[1];
        this.height = this.interpreter.getInputTensor(0).shape()[2];

        this.numRecognitions = 10;

        DataType imageDataType = this.interpreter.getInputTensor(0).dataType();

        int probabilityTensorIndex = 0;
        int[] probabilityShape = interpreter.getOutputTensor(probabilityTensorIndex).shape();
        DataType probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType();

        this.inputImageBuffer = new TensorImage(imageDataType);

        this.outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        this.probabilityProcessor = new TensorProcessor.Builder().add(quantized ? new NormalizeOp(0, 255) : new NormalizeOp(0, 1)).build();

        this.labels = labels;
        if(numRecognitions == 0){
            numRecognitions = labels.length;
        }
        this.numRecognitions = numRecognitions;
    }
    //TODO: Validate that the Tensor Image Classifier works

    /**
     * Runs inference on a given image
     * @param in the image to run the model on
     * @return a list possible recognitions for the image
     */

    public List<Recognition> recognize(Mat in){
        if(in.type() != CvType.CV_8UC3 && in.type() != CvType.CV_8UC4){
            //While some models *technically* have grayscale inputs
            //All official TFOD models require three channels of image input
            //We could convert grayscale images to 3 channel, but there is no standard
            //So we would be guessing if the model wants single, double, or triple channel grayscale
            //We could try to guess, but iterating through a NN to find that would take an unreasonable amount of time
            //The specific type of mat is to prevent data casting type errors
            throw new TensorProcessingException("At this time only mats of type CV_8UC3 are supported");
        }

        long timestamp = System.currentTimeMillis();

        Mat in_32SC3 = new Mat();
        if(in.channels() == 3) {
            //Cast to integer type for processing
            in.convertTo(in_32SC3, CvType.CV_32SC3);
        }else{
            //TFOD models do not process Alpha data, so we need to get rid of the fourth channel
            //For some reason EOCV passes four channel mats even though most cameras
            //Do not have an alpha channel ¯\_(ツ)_/¯
            Imgproc.cvtColor(in, in_32SC3, Imgproc.COLOR_RGBA2RGB);
            in_32SC3.convertTo(in_32SC3, CvType.CV_32SC3);
        }

        int[] data = new int[(int) (in_32SC3.channels() * in_32SC3.total())];
        in_32SC3.get(0, 0, data);
        inputImageBuffer.load(data, new int[]{in_32SC3.width(), in_32SC3.height(), in_32SC3.channels()}); //TF Size is width, height, channels

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
                .add(quantized ? new NormalizeOp(0, 1) : new NormalizeOp(127.5f, 127.5f)).build();
        inputImageBuffer = imageProcessor.process(inputImageBuffer);

        interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

        Map<String, Float> labeledProbability = new TensorLabel(Arrays.asList(labels), probabilityProcessor.process(outputProbabilityBuffer)).getMapWithFloatValue();
        PriorityQueue<Recognition> pq = new PriorityQueue<>(numRecognitions, (o1, o2) -> Float.compare(o2.getConfidence(), o1.getConfidence()));

        for(Map.Entry<String, Float> entry : labeledProbability.entrySet()){
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue()));
        }

        ArrayList<Recognition> recognitions = new ArrayList<>();
        int recogSize = Math.min(pq.size(), numRecognitions);
        for(int i = 0; i < recogSize; i ++){
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    @Deprecated
    public List<Recognition> recognize(Bitmap bitmap){
        inputImageBuffer.load(bitmap);

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
                .add(quantized ? new NormalizeOp(0, 1) : new NormalizeOp(127.5f, 127.5f)).build();
        inputImageBuffer = imageProcessor.process(inputImageBuffer);

        interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

        Map<String, Float> labeledProbability = new TensorLabel(Arrays.asList(labels), probabilityProcessor.process(outputProbabilityBuffer)).getMapWithFloatValue();
        PriorityQueue<Recognition> pq = new PriorityQueue<>(numRecognitions, (o1, o2) -> Float.compare(o2.getConfidence(), o1.getConfidence()));

        for(Map.Entry<String, Float> entry : labeledProbability.entrySet()){
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue()));
        }

        ArrayList<Recognition> recognitions = new ArrayList<>();
        int recogSize = Math.min(pq.size(), numRecognitions);
        for(int i = 0; i < recogSize; i ++){
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static class Recognition {
        private final String id;

        private final String title;

        private final Float confidence;

        public Recognition(
                final String id, final String title, final Float confidence) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            return resultString.trim();
        }
    }
}
