package org.outoftheboxrobotics.tensorflowapi.ObjectDetection;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import com.qualcomm.robotcore.hardware.HardwareMap;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.outoftheboxrobotics.tensorflowapi.TensorProcessingException;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

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
    private final boolean quantized, drawOnImage;
    private final int width, height, numDetections;

    private final int[] intValues;
    private final ByteBuffer buffer;

    private final String[] labels;

    protected TensorObjectDetector(HardwareMap map, String modelName, boolean quantized, boolean drawOnImage, Interpreter.Options options, String[] labels) throws IOException {
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
        this.drawOnImage = drawOnImage;
    }

    /**
     * Runs inference on a given image
     *
     * This method IGNORES drawOnImage!
     *
     * DEPRECATED: Use recognize(Mat in) instead when possible
     * @param bitmap the image to run the model on
     * @return a list of detected objects in the image
     */
    @Deprecated
    public List<Detection> recognize(Bitmap bitmap){
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);
        List<Detection> dets = recognize(mat);
        mat.release();
        return dets;
    }

    /**
     * Runs inference on a given image
     * @param in the image to run the model on
     * @return a list of detected objects in the image
     */
    public List<Detection> recognize(Mat in){
        if(in.type() != CvType.CV_8UC3 && in.type() != CvType.CV_8UC4){
            throw new TensorProcessingException("At this time only mats of type CV_8UC3 are supported");
        }

        long timestamp = System.currentTimeMillis();

        Mat in_32SC3 = new Mat();
        if(in.type() == CvType.CV_8UC3) {
            in.convertTo(in_32SC3, CvType.CV_32SC3);
        }else{
            Imgproc.cvtColor(in, in_32SC3, Imgproc.COLOR_RGBA2RGB);
            in_32SC3.convertTo(in_32SC3, CvType.CV_32SC3);
        }

        int[] data = new int[(int) (in.channels() * in.total())];
        in_32SC3.get(0, 0, data);

        TensorImage image = new TensorImage();
        image.load(data, new int[]{in_32SC3.width(), in_32SC3.height()});

        ImageProcessor.Builder processorBuilder = new ImageProcessor.Builder()
                .add(new ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR));

        if(quantized){
            processorBuilder.add(new QuantizeOp(127.5f, 127.5f));
        }

        image = processorBuilder.build().process(image);

        Object[] inputArray = {image.getBuffer()};

        //Copy output buffers into a tensorflow readable object map
        //TODO: Verify order of operations here or make output tensor assignment automatic
        //This is the most common order of output tensors, but there have been models with different output tensor orders
        float[][][] outputLocations = new float[1][numDetections][4];
        float[][] outputClasses = new float[1][numDetections];
        float[][] outputScores = new float[1][numDetections];
        float[] numDetections = new float[1];

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);

        //Run inference
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        //Only process the number of outputs found by the model since some models will return less then numDetections detections
        //Uses min because some models will return null detections greater then numDetections
        int numDetectionsOutput = Math.min(this.numDetections, (int) numDetections[0]);

        final ArrayList<Detection> detections = new ArrayList<>(numDetectionsOutput);
        for (int i = 0; i < numDetectionsOutput; ++i) {
            final RectF detection = new RectF(
                    outputLocations[0][i][1] * in.width(),
                    outputLocations[0][i][0] * in.height(),
                    outputLocations[0][i][3] * in.width(),
                    outputLocations[0][i][2] * in.height());

            detections.add(
                    new Detection(
                            "" + i, labels[((int) outputClasses[0][i])], outputScores[0][i], detection, timestamp));
            if(drawOnImage){
                Rect r = new Rect(
                        new Point((detection.right) * in.width(), (detection.top) * in.height()),
                        new Point((detection.left) * in.width(), (detection.bottom) * in.height())
                );
                Imgproc.rectangle(in, r, new Scalar(57, 255, 20));
            }
        }
        in_32SC3.release();
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


        private final long imageTimestamp;

        public Detection(
                final String id, final String title, final Float confidence, final RectF location, long imageTimestamp) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
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
