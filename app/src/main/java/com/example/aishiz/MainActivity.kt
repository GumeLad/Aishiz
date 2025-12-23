package com.example.aishiz

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.aishiz.ml.MobilenetV2Imagenet
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val runModelButton: Button = findViewById(R.id.btnRunModel)
        val resultTextView: TextView = findViewById(R.id.tvResult)
        val progressBar: ProgressBar = findViewById(R.id.progressBar)
        val openChatButton: Button = findViewById(R.id.btnOpenChat)

        // Open AI Chat Activity
        openChatButton.setOnClickListener {
            startActivity(Intent(this, ChatActivity::class.java))
        }

        runModelButton.setOnClickListener {
            resultTextView.text = "Running model..."
            progressBar.visibility = View.VISIBLE

            val model = MobilenetV2Imagenet.newInstance(this)

            val inputFeature0 = TensorBuffer.createFixedSize(
                intArrayOf(1, 224, 224, 3),
                DataType.FLOAT32
            )

            val byteCount = 1 * 224 * 224 * 3 * 4
            val byteBuffer = ByteBuffer.allocateDirect(byteCount)
            byteBuffer.order(ByteOrder.nativeOrder())
            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            val outputArray = outputFeature0.floatArray
            val firstValue = outputArray.firstOrNull()

            resultTextView.text = "Model ran. First value: $firstValue"
            progressBar.visibility = View.GONE

            model.close()
        }
    }
}
