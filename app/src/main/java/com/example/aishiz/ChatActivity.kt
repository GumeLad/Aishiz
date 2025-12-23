package com.example.aishiz

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.aishiz.databinding.ActivityChatBinding
import com.example.aishiz.databinding.DialogModelSettingsBinding
import com.google.android.material.slider.Slider
import kotlinx.coroutines.launch

class ChatActivity : AppCompatActivity() {

    private lateinit var binding: ActivityChatBinding
    private lateinit var modelManager: ModelManager
    private lateinit var chatAdapter: ChatAdapter
    private val viewModel: MainViewModel by viewModels()
    
    private var currentModelPath: String? = null
    private var currentConfig = ModelManager.ModelConfig(
        modelPath = "",
        temperature = 0.8f,
        topK = 40,
        topP = 0.95f,
        maxTokens = 1024
    )

    private val modelFilePickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { loadModelFromUri(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChatBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize
        modelManager = ModelManager(this)
        viewModel.setModelManager(modelManager)
        
        // Setup RecyclerView
        chatAdapter = ChatAdapter(viewModel.messages)
        binding.chatRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@ChatActivity).apply {
                stackFromEnd = true
            }
            adapter = chatAdapter
        }

        // Setup buttons
        binding.btnLoadModel.setOnClickListener {
            openModelFilePicker()
        }

        binding.btnModelSettings.setOnClickListener {
            showModelSettingsDialog()
        }

        binding.btnSend.setOnClickListener {
            sendMessage()
        }

        // Load last used model if exists
        loadSavedModel()
    }

    private fun openModelFilePicker() {
        modelFilePickerLauncher.launch(arrayOf("*/*"))
    }

    private fun loadModelFromUri(uri: Uri) {
        binding.progressBar.visibility = View.VISIBLE
        binding.tvModelStatus.text = "Loading model..."

        lifecycleScope.launch {
            try {
                // Copy model to internal storage
                val inputStream = contentResolver.openInputStream(uri)
                    ?: throw IllegalStateException("Cannot open file")

                val fileName = "phi35_mini_${System.currentTimeMillis()}.bin"
                val copyResult = modelManager.copyModelFromUri(inputStream, fileName)

                copyResult.fold(
                    onSuccess = { modelPath ->
                        loadModel(modelPath)
                    },
                    onFailure = { error ->
                        showError("Failed to copy: ${error.message}")
                        binding.progressBar.visibility = View.GONE
                    }
                )
            } catch (e: Exception) {
                showError("Error: ${e.message}")
                binding.progressBar.visibility = View.GONE
            }
        }
    }

    private suspend fun loadModel(modelPath: String) {
        // Check available memory and configure
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val availableMb = memoryInfo.availMem / (1024 * 1024)

        // Update config with new path
        currentConfig = currentConfig.copy(modelPath = modelPath)

        modelManager.loadModel(currentConfig).fold(
            onSuccess = {
                currentModelPath = modelPath
                binding.tvModelStatus.text = "✓ Model loaded successfully"
                binding.btnModelSettings.visibility = View.VISIBLE
                binding.progressBar.visibility = View.GONE

                // Save model path
                getSharedPreferences("aishiz_prefs", Context.MODE_PRIVATE)
                    .edit()
                    .putString("last_model_path", modelPath)
                    .apply()

                Toast.makeText(this, "Model ready! Start chatting.", Toast.LENGTH_SHORT).show()
            },
            onFailure = { error ->
                showError("Failed to load: ${error.message}")
                binding.progressBar.visibility = View.GONE
            }
        )
    }

    private fun sendMessage() {
        val message = binding.etMessage.text.toString().trim()
        if (message.isEmpty()) return

        if (!modelManager.isModelLoaded()) {
            Toast.makeText(this, "Please load a model first", Toast.LENGTH_SHORT).show()
            return
        }

        binding.etMessage.text?.clear()
        
        viewModel.sendMessage(message) {
            runOnUiThread {
                chatAdapter.notifyDataSetChanged()
                binding.chatRecyclerView.scrollToPosition(viewModel.messages.size - 1)
            }
        }
    }

    private fun showModelSettingsDialog() {
        val dialogBinding = DialogModelSettingsBinding.inflate(layoutInflater)
        
        // Set current values
        dialogBinding.sliderTemperature.value = currentConfig.temperature
        dialogBinding.sliderTopK.value = currentConfig.topK.toFloat()
        dialogBinding.sliderTopP.value = currentConfig.topP
        dialogBinding.sliderMaxTokens.value = currentConfig.maxTokens.toFloat()
        
        // Update value labels
        dialogBinding.tvTemperatureValue.text = String.format("%.1f", currentConfig.temperature)
        dialogBinding.tvTopKValue.text = currentConfig.topK.toString()
        dialogBinding.tvTopPValue.text = String.format("%.2f", currentConfig.topP)
        dialogBinding.tvMaxTokensValue.text = currentConfig.maxTokens.toString()

        // Setup sliders
        dialogBinding.sliderTemperature.addOnChangeListener { _, value, _ ->
            dialogBinding.tvTemperatureValue.text = String.format("%.1f", value)
        }
        dialogBinding.sliderTopK.addOnChangeListener { _, value, _ ->
            dialogBinding.tvTopKValue.text = value.toInt().toString()
        }
        dialogBinding.sliderTopP.addOnChangeListener { _, value, _ ->
            dialogBinding.tvTopPValue.text = String.format("%.2f", value)
        }
        dialogBinding.sliderMaxTokens.addOnChangeListener { _, value, _ ->
            dialogBinding.tvMaxTokensValue.text = value.toInt().toString()
        }

        // Preset buttons
        dialogBinding.btnPresetPrecise.setOnClickListener {
            dialogBinding.sliderTemperature.value = 0.3f
            dialogBinding.sliderTopK.value = 10f
            dialogBinding.sliderTopP.value = 0.85f
        }
        dialogBinding.btnPresetBalanced.setOnClickListener {
            dialogBinding.sliderTemperature.value = 0.8f
            dialogBinding.sliderTopK.value = 40f
            dialogBinding.sliderTopP.value = 0.95f
        }
        dialogBinding.btnPresetCreative.setOnClickListener {
            dialogBinding.sliderTemperature.value = 1.2f
            dialogBinding.sliderTopK.value = 50f
            dialogBinding.sliderTopP.value = 0.95f
        }

        val dialog = AlertDialog.Builder(this)
            .setView(dialogBinding.root)
            .create()

        dialogBinding.btnCancel.setOnClickListener {
            dialog.dismiss()
        }

        dialogBinding.btnApply.setOnClickListener {
            applySettings(
                dialogBinding.sliderTemperature.value,
                dialogBinding.sliderTopK.value.toInt(),
                dialogBinding.sliderTopP.value,
                dialogBinding.sliderMaxTokens.value.toInt()
            )
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun applySettings(temperature: Float, topK: Int, topP: Float, maxTokens: Int) {
        currentModelPath?.let { path ->
            currentConfig = ModelManager.ModelConfig(
                modelPath = path,
                temperature = temperature,
                topK = topK,
                topP = topP,
                maxTokens = maxTokens
            )

            binding.progressBar.visibility = View.VISIBLE
            binding.tvModelStatus.text = "Reloading with new settings..."

            lifecycleScope.launch {
                modelManager.loadModel(currentConfig).fold(
                    onSuccess = {
                        binding.tvModelStatus.text = "✓ Settings applied"
                        binding.progressBar.visibility = View.GONE
                        Toast.makeText(
                            this@ChatActivity,
                            "Settings updated successfully",
                            Toast.LENGTH_SHORT
                        ).show()
                    },
                    onFailure = { error ->
                        showError("Failed to apply settings: ${error.message}")
                        binding.progressBar.visibility = View.GONE
                    }
                )
            }
        }
    }

    private fun loadSavedModel() {
        val savedPath = getSharedPreferences("aishiz_prefs", Context.MODE_PRIVATE)
            .getString("last_model_path", null)

        savedPath?.let { path ->
            if (java.io.File(path).exists()) {
                lifecycleScope.launch {
                    binding.tvModelStatus.text = "Loading previous model..."
                    binding.progressBar.visibility = View.VISIBLE
                    loadModel(path)
                }
            }
        }
    }

    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        binding.tvModelStatus.text = "Error: $message"
    }

    override fun onDestroy() {
        super.onDestroy()
        modelManager.close()
    }
}
