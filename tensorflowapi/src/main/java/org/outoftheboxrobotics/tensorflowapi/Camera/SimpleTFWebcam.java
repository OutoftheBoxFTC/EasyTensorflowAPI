package org.outoftheboxrobotics.tensorflowapi.Camera;

import android.graphics.ImageFormat;

import androidx.annotation.NonNull;

import com.qualcomm.robotcore.util.RobotLog;

import org.firstinspires.ftc.robotcore.external.ClassFactory;
import org.firstinspires.ftc.robotcore.external.android.util.Size;
import org.firstinspires.ftc.robotcore.external.function.Continuation;
import org.firstinspires.ftc.robotcore.external.hardware.camera.Camera;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraCaptureRequest;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraCaptureSequenceId;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraCaptureSession;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraCharacteristics;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraException;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraFrame;
import org.firstinspires.ftc.robotcore.external.hardware.camera.CameraName;
import org.firstinspires.ftc.robotcore.internal.camera.CameraManagerInternal;
import org.firstinspires.ftc.robotcore.internal.camera.ImageFormatMapper;
import org.firstinspires.ftc.robotcore.internal.system.Deadline;
import org.firstinspires.ftc.robotcore.internal.vuforia.externalprovider.CameraMode;
import org.firstinspires.ftc.robotcore.internal.vuforia.externalprovider.FrameFormat;
import org.jetbrains.annotations.NotNull;

import java.util.concurrent.TimeUnit;

public abstract class SimpleTFWebcam implements CameraCaptureSession.CaptureCallback{
    private final CameraName cameraName;
    private final CameraManagerInternal manager;
    private Camera camera;
    private final RESOLUTION resolution;

    private final Size res;

    private CameraCharacteristics characteristics;

    private CameraCaptureSession session = null;

    public SimpleTFWebcam(CameraName name){
        this(name, RESOLUTION.USE_LOWEST);
    }

    public SimpleTFWebcam(CameraName name, RESOLUTION resolution){
        this(name, resolution, new Size(-1, -1));
    }

    public SimpleTFWebcam(CameraName name, RESOLUTION resolution, Size res){
        this.cameraName = name;
        this.manager = (CameraManagerInternal) ClassFactory.getInstance().getCameraManager();
        this.resolution = resolution;
        this.res = res;
    }

    /**
     * Attempts to initialize and open the webcam
     * @return true if the camera was opened and false if it wasn't
     */
    public boolean initializeCamera(){
        try {
            camera = manager.requestPermissionAndOpenCamera(new Deadline(2, TimeUnit.SECONDS), cameraName, null);
            if (camera != null) {
                this.characteristics = camera.getCameraName().getCameraCharacteristics();
                return true;
            }else{
                return false;
            }
        }catch (Exception e){
            camera = null;
            throw e;
        }
    }

    public void start(){
        Size size = new Size(-1, -1);
        boolean resSupported = false;

        for(Size s : characteristics.getSizes(ImageFormat.YUY2)){
            if(size.getWidth() == -1 && size.getHeight() == -1){
                size = s;
            }else{
                if(resolution == RESOLUTION.USE_LOWEST) {
                    if ((s.getHeight() * s.getWidth()) < (size.getHeight() * size.getWidth())) {
                        size = s;
                        resSupported = true;
                    }
                }else if(resolution == RESOLUTION.USE_HIGHEST){
                    if ((s.getHeight() * s.getWidth()) > (size.getHeight() * size.getWidth())) {
                        size = s;
                        resSupported = true;
                    }
                }else{
                    if(s.getWidth() == this.res.getWidth() && s.getHeight() == this.res.getHeight()){
                        size = s;
                        resSupported = true;
                    }
                }
            }
        }

        if(!resSupported){
            if(characteristics.getSizes(ImageFormat.YUY2).length == 0){
                throw new TFCameraException("Camera does not support the YUY2 image type, which the API requires to work");
            }else{
                StringBuilder sb = new StringBuilder();
                for(Size s : characteristics.getSizes(ImageFormat.YUY2)){
                    sb.append(s).append(",");
                }
                throw new TFCameraException("Camera does not support requested resolution " + this.res + ". Supported resolutions are " + sb.toString());
            }
        }

        final Size finalSize = size;
        try {
            camera.createCaptureSession(Continuation.create(manager.getSerialThreadPool(), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull @NotNull CameraCaptureSession session) {
                    try {
                        CameraMode mode = new CameraMode(finalSize.getWidth(), finalSize.getHeight(), characteristics.getMaxFramesPerSecond(ImageFormatMapper.androidFromVuforiaWebcam(FrameFormat.YUYV), finalSize), FrameFormat.YUYV);

                        CameraCaptureRequest request = camera.createCaptureRequest(mode.getAndroidFormat(), mode.getSize(), mode.getFramesPerSecond());

                        session.startCapture(request, SimpleTFWebcam.this, Continuation.create(manager.getSerialThreadPool(), (session1, cameraCaptureSequenceId, lastFrameNumber) -> {

                        }));
                        SimpleTFWebcam.this.session = session;
                    }catch (CameraException | RuntimeException e){
                        e.printStackTrace();
                        session.close();
                    }
                }

                @Override
                public void onClosed(@NonNull @NotNull CameraCaptureSession session) {

                }
            }));
        } catch (CameraException e) {
            e.printStackTrace();
        }
    }

    public void stopCamera(){

    }

    @Override
    public void onNewFrame(@NonNull @org.jetbrains.annotations.NotNull CameraCaptureSession session, @NonNull @org.jetbrains.annotations.NotNull CameraCaptureRequest request, @NonNull @org.jetbrains.annotations.NotNull CameraFrame cameraFrame) {

    }

    enum RESOLUTION{
        USE_HIGHEST,
        USE_LOWEST
    }
}
