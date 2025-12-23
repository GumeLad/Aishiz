# Quick Start Implementation Guide

This document provides step-by-step instructions to integrate offline AI model loading into your Aishiz application.

## Step 1: Add Dependencies

### Option A: MediaPipe LLM Inference (Recommended)

Update `app/build.gradle.kts`:

```kotlin
dependencies {
    // Existing dependencies...
    
    // Add MediaPipe for LLM inference
    implementation("com.google.mediapipe:tasks-genai:0.10.14")
}
```

Then sync the project.

### Option B: Direct TensorFlow Lite

If you prefer more control, use TensorFlow Lite directly:

```kotlin
dependencies {
    // Existing dependencies...
    
    // Add TensorFlow Lite GPU delegate for better performance
    implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")
}
```

## Step 2: Enable the ModelManager

The `ModelManager.kt` class has been created for you. To enable it:

1. Open `app/src/main/java/com/example/aishiz/ModelManager.kt`
2. Add MediaPipe dependency (see Step 1)
3. Uncomment the MediaPipe-specific code (marked with TODO comments)

## Step 3: Add Model Loading UI

### Update activity_main.xml

Add a button to select and load models:

```xml
<!-- Add after existing buttons -->
<Button
    android:id="@+id/btnSelectModel"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Load AI Model"
    app:layout_constraintTop_toBottomOf="@+id/btnRunModel"
    app:layout_constraintStart_toStartOf="parent"
    app:layout_constraintEnd_toEndOf="parent" />

<TextView
    android:id="@+id/tvModelStatus"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="No model loaded"
    app:layout_constraintTop_toBottomOf="@+id/btnSelectModel"
    app:layout_constraintStart_toStartOf="parent"
    app:layout_constraintEnd_toEndOf="parent" />
```

### Update MainActivity.kt

Add the following to your `MainActivity`:

```kotlin
import android.content.Intent
import android.net.Uri
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    
    private lateinit var modelManager: ModelManager
    private var currentModelPath: String? = null
    
    companion object {
        private const val REQUEST_MODEL_FILE = 1001
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize ModelManager
        modelManager = ModelManager(this)
        
        // Existing code...
        
        // Add model selection button
        val selectModelButton: Button = findViewById(R.id.btnSelectModel)
        val modelStatusText: TextView = findViewById(R.id.tvModelStatus)
        
        selectModelButton.setOnClickListener {
            openModelFilePicker()
        }
    }
    
    private fun openModelFilePicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf(
                "application/octet-stream",  // .bin files
                "application/x-tflite",       // .tflite files
                "*/*"
            ))
        }
        startActivityForResult(intent, REQUEST_MODEL_FILE)
    }
    
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_MODEL_FILE && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                loadModelFromUri(uri)
            }
        }
    }
    
    private fun loadModelFromUri(uri: Uri) {
        val modelStatusText: TextView = findViewById(R.id.tvModelStatus)
        val progressBar: ProgressBar = findViewById(R.id.progressBar)
        
        lifecycleScope.launch {
            try {
                progressBar.visibility = View.VISIBLE
                modelStatusText.text = "Loading model..."
                
                // Copy model to internal storage
                val inputStream = contentResolver.openInputStream(uri)
                    ?: throw IllegalStateException("Cannot open file")
                
                val fileName = "custom_model_${System.currentTimeMillis()}.bin"
                val copyResult = modelManager.copyModelFromUri(inputStream, fileName)
                
                copyResult.fold(
                    onSuccess = { modelPath ->
                        // Load the model
                        loadModel(modelPath)
                    },
                    onFailure = { error ->
                        modelStatusText.text = "Failed to copy: ${error.message}"
                        progressBar.visibility = View.GONE
                    }
                )
            } catch (e: Exception) {
                modelStatusText.text = "Error: ${e.message}"
                progressBar.visibility = View.GONE
            }
        }
    }
    
    private suspend fun loadModel(modelPath: String) {
        val modelStatusText: TextView = findViewById(R.id.tvModelStatus)
        val progressBar: ProgressBar = findViewById(R.id.progressBar)
        
        // Check available memory and configure accordingly
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val availableMb = memoryInfo.availMem / (1024 * 1024)
        
        // Create config based on available memory
        val config = ModelManager.ModelConfig.forAvailableMemory(availableMb).copy(
            modelPath = modelPath
        )
        
        modelManager.loadModel(config).fold(
            onSuccess = {
                currentModelPath = modelPath
                modelStatusText.text = "✓ Model loaded"
                progressBar.visibility = View.GONE
                
                // Save model path to preferences
                getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
                    .edit()
                    .putString("last_model_path", modelPath)
                    .apply()
            },
            onFailure = { error ->
                modelStatusText.text = "Failed to load: ${error.message}"
                progressBar.visibility = View.GONE
            }
        )
    }
    
    override fun onDestroy() {
        super.onDestroy()
        modelManager.close()
    }
}
```

## Step 4: Integrate with Chat Interface

Update `MainViewModel.kt` to use the ModelManager:

```kotlin
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch

class MainViewModel : ViewModel() {
    val messages = mutableListOf<Message>()
    private var modelManager: ModelManager? = null
    private val conversationHistory = mutableListOf<Pair<String, String>>()
    
    fun setModelManager(manager: ModelManager) {
        modelManager = manager
    }
    
    fun sendMessage(
        userMessage: String,
        onUpdate: () -> Unit
    ) = viewModelScope.launch {
        val manager = modelManager ?: run {
            messages.add(Message("Error: No model loaded", Role.ASSISTANT))
            onUpdate()
            return@launch
        }
        
        // Add user message
        messages.add(Message(userMessage, Role.USER))
        onUpdate()
        
        // Add typing indicator
        messages.add(Message("", Role.TYPING))
        onUpdate()
        
        try {
            // Generate response with streaming
            val fullResponse = StringBuilder()
            
            manager.generateContextualResponse(
                messages = conversationHistory,
                newMessage = userMessage,
                onPartialResult = { partial ->
                    fullResponse.append(partial)
                    
                    // Update typing indicator with partial response
                    val lastIndex = messages.indexOfLast { it.role == Role.TYPING }
                    if (lastIndex >= 0) {
                        messages[lastIndex] = Message(fullResponse.toString(), Role.ASSISTANT)
                        onUpdate()
                    }
                }
            ).fold(
                onSuccess = { response ->
                    // Remove typing indicator
                    messages.removeAll { it.role == Role.TYPING }
                    
                    // Ensure final message is present
                    if (messages.lastOrNull()?.text != response) {
                        messages.add(Message(response, Role.ASSISTANT))
                    }
                    
                    // Add to conversation history
                    conversationHistory.add(userMessage to response)
                    
                    onUpdate()
                },
                onFailure = { error ->
                    // Remove typing indicator
                    messages.removeAll { it.role == Role.TYPING }
                    
                    // Add error message
                    messages.add(Message("Error: ${error.message}", Role.ASSISTANT))
                    onUpdate()
                }
            )
        } catch (e: Exception) {
            // Remove typing indicator
            messages.removeAll { it.role == Role.TYPING }
            
            // Add error message
            messages.add(Message("Error: ${e.message}", Role.ASSISTANT))
            onUpdate()
        }
    }
    
    fun clearConversation() {
        messages.clear()
        conversationHistory.clear()
    }
}
```

Then in your chat activity, connect the ViewModel:

```kotlin
// In your chat activity
private val viewModel: MainViewModel by viewModels()

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    
    // Set the model manager
    viewModel.setModelManager(modelManager)
    
    // Send button click
    sendButton.setOnClickListener {
        val message = inputEditText.text.toString()
        if (message.isNotBlank()) {
            inputEditText.text.clear()
            
            viewModel.sendMessage(message) {
                // Update UI
                chatAdapter.notifyDataSetChanged()
                recyclerView.scrollToPosition(viewModel.messages.size - 1)
            }
        }
    }
}
```

## Step 5: Download and Test a Model

### Option 1: Gemma 2B (Recommended for testing)

1. Visit [Kaggle Models - Gemma](https://www.kaggle.com/models/google/gemma)
2. Download Gemma 2B IT (instruction-tuned) model
3. Convert to MediaPipe format using:
   ```bash
   pip install ai-edge-torch
   python convert_gemma_to_mediapipe.py
   ```
4. Transfer the `.bin` file to your Android device
5. Use the "Load AI Model" button to select it

### Option 2: Use a Smaller Test Model

For quick testing, you can use a smaller quantized model:

1. Download TinyLlama-1.1B (quantized): ~600MB
2. Convert to TFLite format
3. Load into the app

### Option 3: Bundle Model with App

For distribution, bundle the model in assets:

```kotlin
// In MainActivity onCreate
lifecycleScope.launch {
    val modelPath = filesDir.absolutePath + "/bundled_model.bin"
    modelManager.copyModelFromAssets(
        "models/gemma-2b-it.bin",
        modelPath
    ).fold(
        onSuccess = { path ->
            loadModel(path)
        },
        onFailure = { error ->
            Log.e(TAG, "Failed to load bundled model", error)
        }
    )
}
```

## Step 6: Customize Inference Parameters

Create a settings UI to let users customize the model behavior:

```kotlin
// In your settings activity or dialog
fun applyModelSettings(
    temperature: Float,
    topK: Int,
    topP: Float,
    maxTokens: Int
) {
    currentModelPath?.let { path ->
        val config = ModelManager.ModelConfig(
            modelPath = path,
            temperature = temperature,
            topK = topK,
            topP = topP,
            maxTokens = maxTokens
        )
        
        lifecycleScope.launch {
            modelManager.loadModel(config)
        }
    }
}
```

## Step 7: Add Permission (if loading from external storage)

Update `AndroidManifest.xml`:

```xml
<!-- Only needed if loading from external storage (not recommended for security) -->
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" 
    android:maxSdkVersion="32" />
```

Note: Using `ACTION_OPEN_DOCUMENT` doesn't require this permission as it uses the system picker.

## Step 8: Test Your Implementation

### Basic Test Checklist

- [ ] App builds without errors
- [ ] "Load AI Model" button appears
- [ ] File picker opens when button is clicked
- [ ] Model file can be selected
- [ ] Model loads successfully (check status text)
- [ ] Can send messages and receive responses
- [ ] Typing indicator shows during generation
- [ ] Responses appear in chat
- [ ] App doesn't crash on model loading errors

### Performance Test Checklist

- [ ] Check memory usage in Android Studio Profiler
- [ ] Verify inference speed is acceptable (< 5s for short prompts)
- [ ] Test with different model sizes
- [ ] Test on different devices (if possible)

## Troubleshooting

### "Model not loaded" error
- Ensure MediaPipe dependency is added
- Uncomment the MediaPipe code in ModelManager.kt
- Rebuild the project

### "Out of memory" error
- Use a smaller model (e.g., 2B instead of 7B)
- Reduce `maxTokens` in ModelConfig
- Close other apps to free memory

### Slow inference
- Enable GPU acceleration (should be default with MediaPipe)
- Use quantized models (INT8 instead of FP32)
- Reduce `maxTokens`

### Model produces gibberish
- Check model format (must be MediaPipe-compatible TFLite)
- Verify model was converted correctly
- Try a different model

## Next Steps

Once basic functionality is working:

1. **Add model management**: List, delete, and switch between loaded models
2. **Implement RAG**: Add document/knowledge base support
3. **Add voice**: Integrate speech-to-text and text-to-speech
4. **Fine-tune**: Add on-device training/fine-tuning
5. **Optimize**: Profile and optimize for battery life

## Resources

- Full guide: `OFFLINE_AI_MODEL_GUIDE.md`
- ModelManager source: `app/src/main/java/com/example/aishiz/ModelManager.kt`
- MediaPipe docs: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference

## Support

If you encounter issues:

1. Check the logs: `adb logcat | grep ModelManager`
2. Review the full guide: `OFFLINE_AI_MODEL_GUIDE.md`
3. Verify MediaPipe dependency version
4. Test with a known-good model (Gemma 2B)
