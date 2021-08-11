package com.outoftheboxrobotics.tensorflowapi.ObjectDetection;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;

import com.qualcomm.robotcore.hardware.HardwareMap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TensorObjectDetector {
    private final Interpreter interpreter;
    private final boolean quantized;
    private final int width, height, numDetections;

    private final int[] intValues;
    private final ByteBuffer buffer;

    private final String[] labels;

    protected TensorObjectDetector(HardwareMap map, String modelName, boolean quantized, Interpreter.Options options, String[] labels) throws IOException {
        MappedByteBuffer model = loadModelFile(map.appContext.getAssets(), modelName);
        this.quantized = quantized;

        this.interpreter = new Interpreter(model, options);

        this.width = this.interpreter.getInputTensor(0).shape()[1];
        this.height = this.interpreter.getInputTensor(0).shape()[2];
        this.numDetections = this.interpreter.getOutputTensor(0).shape()[1];

        this.intValues = new int[this.width * this.height];
        this.buffer = ByteBuffer.allocateDirect(this.width * this.height * 3 * (quantized ? 1 : 4));
        this.buffer.order(ByteOrder.nativeOrder());

        this.labels = labels;
    }

    public List<Detection> recognize(Bitmap bitmap){
        Bitmap clone = bitmap.copy(bitmap.getConfig(), true);
        if(bitmap.getWidth() != width || bitmap.getHeight() != height){
            clone = Bitmap.createScaledBitmap(bitmap, width, height, false);
        }
        clone.getPixels(intValues, 0, width, 0, 0, width, height);

        long timestamp = System.currentTimeMillis();

        buffer.rewind();
        for(int i = 0; i < width; i ++){
            for(int j = 0; j < height; j ++){
                int pixel = intValues[i * width + j];
                if(quantized){
                    buffer.put((byte) ((pixel >> 16) & 0xFF));
                    buffer.put((byte) ((pixel >> 8) & 0xFF));
                    buffer.put((byte) (pixel & 0xFF));
                }else{
                    buffer.putFloat((((pixel >> 16) & 0xFF) - 127.5f) / 127.5f);
                    buffer.putFloat((((pixel >> 8) & 0xFF) - 127.5f) / 127.5f);
                    buffer.putFloat(((pixel & 0xFF) - 127.5f) / 127.5f);
                }
            }
        }

        Object[] inputArray = {buffer};
        float[][][] outputLocations = new float[1][numDetections][4];
        float[][] outputClasses = new float[1][numDetections];
        float[][] outputScores = new float[1][numDetections];
        float[] numDetections = new float[1];

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);

        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        int numDetectionsOutput = Math.min(this.numDetections, (int) numDetections[0]);

        final ArrayList<Detection> detections = new ArrayList<>(numDetectionsOutput);
        for (int i = 0; i < numDetectionsOutput; ++i) {
            final RectF detection = new RectF(
                            outputLocations[0][i][1] * width,
                            outputLocations[0][i][0] * height,
                            outputLocations[0][i][3] * width,
                            outputLocations[0][i][2] * height);

            detections.add(
                    new Detection(
                            "" + i, labels[((int) outputClasses[0][i])], outputScores[0][i], detection, bitmap, timestamp));
        }
        clone.recycle();
        return detections;
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

    public static class Detection {
        private final String id;

        private final String title;

        private final Float confidence;

        private RectF location;

        private final Bitmap bitmap;

        private final long imageTimestamp;

        public Detection(
                final String id, final String title, final Float confidence, final RectF location, Bitmap bitmap, long imageTimestamp) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
            this.bitmap = bitmap;
            this.imageTimestamp = imageTimestamp;
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

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        public long getImageTimestamp() {
            return imageTimestamp;
        }

        public Bitmap getBitmap() {
            return bitmap;
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
            if (location != null) {
                resultString += location + " ";
            }
            return resultString.trim();
        }
    }
}
