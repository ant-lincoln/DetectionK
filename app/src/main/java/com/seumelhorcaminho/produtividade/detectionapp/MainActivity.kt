package com.seumelhorcaminho.produtividade.detectionapp

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.seumelhorcaminho.produtividade.detectionapp.databinding.ActivityMainBinding
import com.seumelhorcaminho.produtividade.detectionapp.ml.AutoModel1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraManager: CameraManager
    private lateinit var handler: Handler
    private lateinit var cameraDevice: CameraDevice
    lateinit var bitmap: Bitmap
    lateinit var imageProcessor: ImageProcessor
    lateinit var model: AutoModel1
    lateinit var labels: List<String>
    var colors = listOf(
        Color.BLUE, Color.GREEN,Color.RED,Color.GRAY,Color.YELLOW,Color.BLACK
    )
    val paint = Paint()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        getPermission()

        labels = FileUtil.loadLabels(this,"labels.txt")
        imageProcessor =
            ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()



        model = AutoModel1.newInstance(this)


        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        binding.textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(s: SurfaceTexture, p1: Int, p2: Int) {
                openCam()
            }

            override fun onSurfaceTextureSizeChanged(s: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(s: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(s: SurfaceTexture) {
                bitmap = binding.textureView.bitmap!!
                var image = TensorImage.fromBitmap(bitmap)
                image = imageProcessor.process(image)

                // Runs model inference and gets result.
                val outputs = model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes = outputs.classesAsTensorBuffer.floatArray
                val scores = outputs.scoresAsTensorBuffer.floatArray
                val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray


                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width
                paint.textSize = h/15f
                paint.strokeWidth = h/85f
                var x = 0
                scores.forEachIndexed { index, fl ->
                    x = index
                    x *= 4
                    if(fl > 0.5){
                        paint.setColor(colors.get(index))
                        paint.style = Paint.Style.STROKE
                        canvas.drawRect(RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h), paint)
                        paint.style = Paint.Style.FILL
                        canvas.drawText(labels.get(classes.get(index).toInt())+" "+fl.toString(), locations.get(x+1)*w, locations.get(x)*h, paint)
                    }
                }

                binding.imageView.setImageBitmap(mutable)


            }


        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    @SuppressLint("MissingPermission")
    private fun openCam() {
        cameraManager.openCamera(
            cameraManager.cameraIdList[0],
            object : CameraDevice.StateCallback() {
                override fun onOpened(cam: CameraDevice) {
                    cameraDevice = cam

                    var surfaceTexture = binding.textureView.surfaceTexture
                    var surface = Surface(surfaceTexture)

                    var captureRequest =
                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                    captureRequest.addTarget(surface)

                    cameraDevice.createCaptureSession(
                        listOf(surface),
                        object : CameraCaptureSession.StateCallback() {
                            override fun onConfigured(cam: CameraCaptureSession) {
                                cam.setRepeatingRequest(captureRequest.build(), null, null)
                            }

                            override fun onConfigureFailed(p0: CameraCaptureSession) {
                            }
                        },
                        handler
                    )
                }

                override fun onDisconnected(p0: CameraDevice) {}

                override fun onError(p0: CameraDevice, p1: Int) {}

            },
            handler
        )
    }

    private fun getPermission() {
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            getPermission()
        }
    }
}