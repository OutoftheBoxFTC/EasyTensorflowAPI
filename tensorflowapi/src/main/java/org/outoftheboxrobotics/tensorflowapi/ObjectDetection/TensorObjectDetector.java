package org.outoftheboxrobotics.tensorflowapi.ObjectDetection;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import com.qualcomm.robotcore.hardware.HardwareMap;
import com.qualcomm.robotcore.util.RobotLog;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.outoftheboxrobotics.tensorflowapi.TensorProcessingException;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.CastOp;
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
    private final boolean drawOnImage;
    private boolean quantized;
    private final int width, height, numDetections;
    private float minConfidence;

    private final String[] labels;

    protected TensorObjectDetector(HardwareMap map, String modelName, boolean quantized, boolean drawOnImage, float minConfidence, Interpreter.Options options, String[] labels) throws IOException {
        MappedByteBuffer model = loadModelFile(map.appContext.getAssets(), modelName);
        this.quantized = quantized;

        this.interpreter = new Interpreter(model, options);

        this.width = this.interpreter.getInputTensor(0).shape()[1];
        this.height = this.interpreter.getInputTensor(0).shape()[2];
        this.numDetections = this.interpreter.getOutputTensor(0).shape()[1];

        //The number of bytes in the input tensor should equal the number of bytes to allocate
        if(this.interpreter.getInputTensor(0).numBytes() != (this.width * this.height * 3 * (quantized ? 1 : 4))){
            //Dang it, the user seems to have messed up quantization settings. Right now, the model is guaranteed to fail
            //So we might as well attempt to change it to get the run to work
            RobotLog.addGlobalWarningMessage("Warning! Quantization was not set correctly, changing settings to avoid error");
            this.quantized = !this.quantized;
        }

        this.labels = labels;
        this.drawOnImage = drawOnImage;
        this.minConfidence = minConfidence;
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

        TensorImage image = new TensorImage(DataType.FLOAT32); //Load the data with FLOAT32 type
        //For some reason, casting from FLOAT32 to UINT8 is slightly faster then the other way around on some devices

        image.load(data, new int[]{in_32SC3.width(), in_32SC3.height(), in_32SC3.channels()}); //TF Size is width, height, channels

        ImageProcessor.Builder processorBuilder = new ImageProcessor.Builder()
                .add(new ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR));
        //We cannot strictly trust that the passed image is the right size, we we bilinear scale to the right dims
        //We could add a crop or something along those lines here, but it would be faster and more efficient
        //For the user to do so in EOCV before passing, since we do not know *where* they want to crop

        if(quantized){
            processorBuilder.add(new QuantizeOp(127.5f, 127.5f)); //Quantization formula
            processorBuilder.add(new CastOp(DataType.UINT8)); //QuantizeOp does not cast to int, need to do so here
        }

        image = processorBuilder.build().process(image);

        Object[] inputArray = {image.getBuffer()};
        //TFOD expects an object array for multi-run, so we just wrap it here

        //Copy output buffers into a tensorflow readable object map
        float[][][] outputLocations = new float[1][numDetections][4];
        float[][] outputClasses = new float[1][numDetections];
        float[][] outputScores = new float[1][numDetections];
        float[] numDetections = new float[1];

        //TODO: Verify order of operations here or make output tensor assignment automatic
        //This is the most common order of output tensors, but there have been models with different output tensor orders
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

        float[][] tmp = new float[1][this.numDetections];
        float maxClass = 0, maxScore = 0;
        boolean checkScores = false, swap = true;
        for(int i = 0; i < numDetectionsOutput; i ++){
            if(outputClasses[0][i] >= labels.length){
                //There are two possibilities here
                //Either the user inputted the wrong number of labels (possible)
                //Or the output classes and scores somehow got switched (also possible)
                //Unfortunately this means we need to loop through the other list to check them too
                checkScores = true;
                maxClass = Math.max(maxClass, outputClasses[0][i]);
            }
            if(outputScores[0][i] >= labels.length){
                swap = false;
                maxScore = Math.max(maxScore, outputScores[0][i]);
            }
        }

        if(checkScores && swap){
            //Swapping scores and classes should fix things
            tmp = outputClasses;
            outputClasses = outputScores;
            outputScores = tmp;
        }
        if(checkScores && !swap){
            //Sigh, they didn't give us the right amount of labels
            //Technically, we don't know how many labels there should be
            //So we give both numbers and let them figure it out
            throw new TensorProcessingException("Processing output failed. Is the number of labels passed correct? Expected either " + maxClass + " or " + maxScore + " labels!");
        }

        final ArrayList<Detection> detections = new ArrayList<>(numDetectionsOutput);
        for (int i = 0; i < numDetectionsOutput; ++i) {
            if(outputScores[0][i] > minConfidence) {
                //TF outputs location as a number from 0-1 for width and height, most TF apis scale this to the internal model size
                //I.E 300x300, but this does not make sense in this context, so we scale them to the input image size
                final RectF detection = new RectF(
                        outputLocations[0][i][1] * in.width(),
                        outputLocations[0][i][0] * in.height(),
                        outputLocations[0][i][3] * in.width(),
                        outputLocations[0][i][2] * in.height());

                detections.add(
                        new Detection(
                                "" + i, labels[((int) outputClasses[0][i])], outputScores[0][i], detection, timestamp));
                if (drawOnImage) {
                    Rect r = new Rect(
                            new Point((detection.right), (detection.top)),
                            new Point((detection.left), (detection.bottom))
                    );
                    //Neon green, most likely colour to both stand out
                    //And not be used on the field
                    Imgproc.rectangle(in, r, new Scalar(57, 255, 20));
                    Imgproc.putText(in, labels[((int) outputClasses[0][i])] + " " + ((int)(outputScores[0][i] * 100)) + "%", new Point(detection.centerX(), detection.centerY()), Imgproc.FONT_HERSHEY_COMPLEX, 0.4, new Scalar(57, 255, 20));
                }
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
        //Could this be streamed on the fly for branched models?
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
