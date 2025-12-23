# Offline AI Model Integration Guide for Aishiz

## Current State Analysis

### Existing Infrastructure
Your Android application currently has:
- ✅ **TensorFlow Lite** runtime (v2.17.0) with ML model binding
- ✅ **MobileNet V2** for image classification (3.8 MB)
- ✅ **MobileBERT** model (24 shards, ~100 MB total) in TensorFlow.js format
- ✅ **Chat UI** with RecyclerView adapter and message models
- ✅ **Native C++ support** via CMake/NDK
- ✅ **Gradle 8.13.1** with Kotlin 1.9.24

### Current Limitations
- ❌ MobileBERT is in TFJS format (not optimized for Android)
- ❌ No dynamic model loading capability
- ❌ MobileNet only does image classification (not conversational AI)
- ❌ No customizable inference parameters
- ❌ No token streaming support

## Recommended Approach: MediaPipe LLM Inference API

### Why MediaPipe LLM?
For **maximum flexibility and customization** with **fully offline** operation:

1. **Most Flexible**: Supports Gemma, Phi, Falcon, and custom models
2. **Easy Model Loading**: Load models from files at runtime
3. **Customizable Parameters**: Temperature, top-k, top-p, max tokens
4. **Token Streaming**: Real-time response generation
5. **Optimized**: GPU acceleration and memory-efficient
6. **Well-Maintained**: Official Google solution for on-device LLM

### Supported Models (All Fully Offline)
- **Gemma 2B** (2GB) - Google's most flexible small model
- **Gemma 7B** (7GB) - More capable, requires high-end device
- **Phi-2** (2.7GB) - Microsoft's efficient model
- **Custom models** - Convert your own with model conversion tools

## Implementation Plan

### Phase 1: Add MediaPipe Dependencies

Update `app/build.gradle.kts`:
```kotlin
dependencies {
    // Existing dependencies...
    
    // MediaPipe LLM Inference
    implementation("com.google.mediapipe:tasks-genai:0.10.14")
}
```

Update `gradle/libs.versions.toml`:
```toml
[versions]
mediapipe = "0.10.14"

[libraries]
mediapipe-tasks-genai = { module = "com.google.mediapipe:tasks-genai", version.ref = "mediapipe" }
```

### Phase 2: Create Model Manager

Create `ModelManager.kt`:
```kotlin
package com.example.aishiz

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class ModelManager(private val context: Context) {
    
    private var llmInference: LlmInference? = null
    
    data class ModelConfig(
        val modelPath: String,
        val temperature: Float = 0.8f,
        val topK: Int = 40,
        val topP: Float = 0.95f,
        val maxTokens: Int = 1024,
        val randomSeed: Int = 0
    )
    
    suspend fun loadModel(config: ModelConfig): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(config.modelPath)
                .setTemperature(config.temperature)
                .setTopK(config.topK)
                .setTopP(config.topP)
                .setMaxTokens(config.maxTokens)
                .setRandomSeed(config.randomSeed)
                .build()
            
            llmInference?.close()
            llmInference = LlmInference.createFromOptions(context, options)
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun generateResponse(
        prompt: String,
        onPartialResult: ((String) -> Unit)? = null
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val inference = llmInference ?: return@withContext Result.failure(
                IllegalStateException("Model not loaded")
            )
            
            val fullResponse = StringBuilder()
            
            if (onPartialResult != null) {
                // Streaming response
                inference.generateResponseAsync(prompt)
                for (partialResult in inference.generateResponseAsync(prompt)) {
                    partialResult?.let {
                        fullResponse.append(it)
                        withContext(Dispatchers.Main) {
                            onPartialResult(it)
                        }
                    }
                }
            } else {
                // Non-streaming response
                val response = inference.generateResponse(prompt)
                fullResponse.append(response)
            }
            
            Result.success(fullResponse.toString())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    fun close() {
        llmInference?.close()
        llmInference = null
    }
    
    fun isModelLoaded(): Boolean = llmInference != null
}
```

### Phase 3: Update MainActivity for Model Selection

Add to `MainActivity.kt`:
```kotlin
private lateinit var modelManager: ModelManager
private var currentModelPath: String? = null

// In onCreate:
modelManager = ModelManager(this)

// Add button to select model file
val selectModelButton: Button = findViewById(R.id.btnSelectModel)
selectModelButton.setOnClickListener {
    openModelFilePicker()
}

private fun openModelFilePicker() {
    val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
        addCategory(Intent.CATEGORY_OPENABLE)
        type = "*/*"
        putExtra(Intent.EXTRA_MIME_TYPES, arrayOf(
            "application/octet-stream",  // .bin files
            "*/*"
        ))
    }
    startActivityForResult(intent, REQUEST_MODEL_FILE)
}

override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    if (requestCode == REQUEST_MODEL_FILE && resultCode == RESULT_OK) {
        data?.data?.let { uri ->
            copyModelToInternalStorage(uri)
        }
    }
}

private fun copyModelToInternalStorage(uri: Uri) {
    lifecycleScope.launch {
        try {
            val modelFile = File(filesDir, "custom_model.bin")
            contentResolver.openInputStream(uri)?.use { input ->
                modelFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            loadCustomModel(modelFile.absolutePath)
        } catch (e: Exception) {
            showError("Failed to load model: ${e.message}")
        }
    }
}

private suspend fun loadCustomModel(modelPath: String) {
    val config = ModelManager.ModelConfig(
        modelPath = modelPath,
        temperature = 0.8f,
        topK = 40,
        topP = 0.95f,
        maxTokens = 1024
    )
    
    modelManager.loadModel(config).fold(
        onSuccess = {
            currentModelPath = modelPath
            showSuccess("Model loaded successfully!")
        },
        onFailure = { error ->
            showError("Failed to load model: ${error.message}")
        }
    )
}
```

### Phase 4: Integrate with Chat UI

Update `MainViewModel.kt`:
```kotlin
class MainViewModel : ViewModel() {
    val messages = mutableListOf<Message>()
    private var modelManager: ModelManager? = null
    
    fun setModelManager(manager: ModelManager) {
        modelManager = manager
    }
    
    suspend fun sendMessage(userMessage: String): Result<String> {
        messages.add(Message(userMessage, Role.USER))
        
        val manager = modelManager ?: return Result.failure(
            IllegalStateException("Model not initialized")
        )
        
        // Add typing indicator
        messages.add(Message("", Role.TYPING))
        
        val result = manager.generateResponse(userMessage) { partial ->
            // Update the last message with partial response
            val lastIndex = messages.lastIndex
            if (lastIndex >= 0 && messages[lastIndex].role == Role.TYPING) {
                messages[lastIndex] = Message(partial, Role.ASSISTANT)
            }
        }
        
        // Remove typing indicator if still there
        if (messages.lastOrNull()?.role == Role.TYPING) {
            messages.removeAt(messages.lastIndex)
        }
        
        result.onSuccess { response ->
            // Ensure final response is added
            if (messages.lastOrNull()?.text != response) {
                messages.add(Message(response, Role.ASSISTANT))
            }
        }
        
        return result
    }
}
```

## Model Preparation

### Option 1: Use Pre-converted Models

Download pre-converted models:
- **Gemma 2B**: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference
- **Phi-2**: Convert using instructions below

### Option 2: Convert Your Own Model

#### Converting GGUF to TFLite for MediaPipe

1. Install conversion tools:
```bash
pip install ai-edge-torch
pip install transformers
```

2. Convert PyTorch model to TFLite:
```python
import ai_edge_torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-2"  # Or your model
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Convert to TFLite
edge_model = ai_edge_torch.convert(model, sample_args)
edge_model.export("model.bin")
```

3. Place the `.bin` file in your device storage or app assets

### Option 3: Bundle Model with App

For models under 100MB, bundle in `app/src/main/assets/`:
```
app/src/main/assets/
  └── models/
      └── gemma-2b-it.bin
```

Access in code:
```kotlin
val modelPath = context.getExternalFilesDir(null)?.absolutePath + "/gemma-2b-it.bin"
// Copy from assets to files dir first time
copyAssetToFile("models/gemma-2b-it.bin", modelPath)
```

## Customization Options

### Inference Parameters

```kotlin
// Creative writing
ModelConfig(
    temperature = 1.2f,  // More random/creative
    topK = 50,
    topP = 0.95f
)

// Precise answers
ModelConfig(
    temperature = 0.3f,  // More deterministic
    topK = 10,
    topP = 0.85f
)

// Balanced
ModelConfig(
    temperature = 0.8f,
    topK = 40,
    topP = 0.95f
)
```

### Context Window Management

```kotlin
class ConversationManager {
    private val maxContextTokens = 2048
    private val conversationHistory = mutableListOf<Pair<String, String>>()
    
    fun buildPrompt(newMessage: String): String {
        val context = conversationHistory.takeLast(5)  // Last 5 exchanges
        val prompt = buildString {
            context.forEach { (user, assistant) ->
                append("User: $user\n")
                append("Assistant: $assistant\n")
            }
            append("User: $newMessage\n")
            append("Assistant: ")
        }
        return prompt
    }
    
    fun addExchange(userMsg: String, assistantMsg: String) {
        conversationHistory.add(userMsg to assistantMsg)
    }
}
```

## Alternative Approach: TensorFlow Lite with XNNPACK

If MediaPipe doesn't meet your needs, use direct TensorFlow Lite:

### Advantages
- Even more control over model execution
- Can use custom operators
- Direct access to tensor manipulation

### Implementation

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

class TFLiteModelManager(context: Context) {
    private var interpreter: Interpreter? = null
    
    fun loadModel(modelPath: String, useGpu: Boolean = true) {
        val options = Interpreter.Options().apply {
            if (useGpu) {
                addDelegate(GpuDelegate())
            }
            setNumThreads(4)
        }
        
        val modelFile = File(modelPath)
        interpreter = Interpreter(modelFile, options)
    }
    
    fun generateText(input: IntArray): IntArray {
        val interpreter = interpreter ?: throw IllegalStateException("Model not loaded")
        
        val outputShape = interpreter.getOutputTensor(0).shape()
        val output = IntArray(outputShape[1])
        
        interpreter.run(input, output)
        return output
    }
}
```

## Performance Considerations

### Model Size vs. Device Constraints

| Model | Size | RAM Required | Recommended Device |
|-------|------|--------------|-------------------|
| Gemma 2B | 2GB | 4GB+ | Mid-range (2021+) |
| Phi-2 | 2.7GB | 6GB+ | High-end (2022+) |
| Gemma 7B | 7GB | 12GB+ | Flagship (2023+) |

### Memory Management

```kotlin
class ModelConfig {
    companion object {
        fun recommendedConfig(availableMemoryMb: Int): ModelConfig {
            return when {
                availableMemoryMb < 3000 -> ModelConfig(
                    maxTokens = 256,
                    temperature = 0.7f
                )
                availableMemoryMb < 6000 -> ModelConfig(
                    maxTokens = 512,
                    temperature = 0.8f
                )
                else -> ModelConfig(
                    maxTokens = 1024,
                    temperature = 0.8f
                )
            }
        }
    }
}

// Check available memory
val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
val memoryInfo = ActivityManager.MemoryInfo()
activityManager.getMemoryInfo(memoryInfo)
val availableMb = memoryInfo.availMem / (1024 * 1024)
```

### Inference Optimization

```kotlin
// Batch processing
class OptimizedInference(private val modelManager: ModelManager) {
    private val inferenceQueue = mutableListOf<String>()
    
    suspend fun queueInference(prompt: String) {
        inferenceQueue.add(prompt)
        if (inferenceQueue.size >= 3) {
            processBatch()
        }
    }
    
    private suspend fun processBatch() {
        val batch = inferenceQueue.toList()
        inferenceQueue.clear()
        
        batch.forEach { prompt ->
            modelManager.generateResponse(prompt)
        }
    }
}
```

## Testing Your Implementation

### Basic Test Cases

```kotlin
// Test 1: Model Loading
@Test
fun testModelLoading() = runBlocking {
    val modelManager = ModelManager(context)
    val config = ModelManager.ModelConfig(
        modelPath = "path/to/model.bin"
    )
    val result = modelManager.loadModel(config)
    assertTrue(result.isSuccess)
}

// Test 2: Simple Inference
@Test
fun testInference() = runBlocking {
    modelManager.loadModel(testConfig)
    val result = modelManager.generateResponse("Hello, how are you?")
    assertTrue(result.isSuccess)
    assertFalse(result.getOrNull().isNullOrEmpty())
}

// Test 3: Streaming
@Test
fun testStreamingInference() = runBlocking {
    val parts = mutableListOf<String>()
    modelManager.generateResponse("Tell me a story") { partial ->
        parts.add(partial)
    }
    assertTrue(parts.size > 1)
}
```

## Security Considerations

### Model Validation

```kotlin
fun validateModelFile(file: File): Boolean {
    // Check file size (prevent loading corrupted/malicious files)
    if (file.length() > 10L * 1024 * 1024 * 1024) {  // 10GB max
        return false
    }
    
    // Check magic bytes for TFLite format
    file.inputStream().use { input ->
        val header = ByteArray(4)
        input.read(header)
        // TFLite files start with "TFL3"
        return header.contentEquals(byteArrayOf(0x54, 0x46, 0x4C, 0x33))
    }
}
```

### Sandboxed Execution

```kotlin
// Run inference in isolated process
class SandboxedModelManager(context: Context) {
    private val executorService = Executors.newSingleThreadExecutor()
    
    fun generateResponseSafely(prompt: String): Future<Result<String>> {
        return executorService.submit<Result<String>> {
            try {
                // Inference happens in separate thread
                modelManager.generateResponse(prompt)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }
}
```

## Next Steps

### Immediate Actions
1. ✅ Add MediaPipe dependency to `build.gradle.kts`
2. ✅ Create `ModelManager.kt` class
3. ✅ Add model file picker UI
4. ✅ Test with Gemma 2B model
5. ✅ Integrate with existing chat UI

### Future Enhancements
- **Multi-model support**: Switch between models at runtime
- **Model marketplace**: Download models from curated list
- **Fine-tuning interface**: Train models on device
- **RAG support**: Add retrieval-augmented generation
- **Voice integration**: Speech-to-text and text-to-speech

## Resources

### Official Documentation
- [MediaPipe LLM Inference Guide](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)
- [TensorFlow Lite for Android](https://www.tensorflow.org/lite/android)
- [AI Edge Torch Conversion](https://github.com/google-ai-edge/ai-edge-torch)

### Model Sources
- [Hugging Face Models](https://huggingface.co/models?library=transformers)
- [Kaggle Models](https://www.kaggle.com/models)
- [Google AI Edge](https://ai.google.dev/edge)

### Community
- [TensorFlow Lite Discussion](https://discuss.tensorflow.org/c/lite)
- [MediaPipe GitHub](https://github.com/google/mediapipe)

## Troubleshooting

### Common Issues

**Issue**: "Failed to load model"
- **Solution**: Ensure model file is in correct format (TFLite binary)
- **Check**: File permissions and path accessibility

**Issue**: "Out of memory"
- **Solution**: Reduce `maxTokens` or use smaller model
- **Check**: Available RAM with ActivityManager

**Issue**: "Slow inference"
- **Solution**: Enable GPU acceleration
- **Check**: Use XNNPACK delegate for CPU optimization

**Issue**: "Model produces gibberish"
- **Solution**: Verify model quantization settings
- **Check**: Input preprocessing (tokenization)

## Conclusion

This guide provides a complete roadmap for integrating personal offline AI models into your Aishiz application with **maximum flexibility and customization**. The MediaPipe LLM Inference API is recommended as it offers the best balance of:

- ✅ **Flexibility**: Load any compatible model
- ✅ **Performance**: Optimized for mobile devices
- ✅ **Customization**: Full control over inference parameters
- ✅ **Offline**: 100% on-device execution
- ✅ **Ease of use**: Simple API with streaming support

Start with Gemma 2B for testing, then scale up to larger models or fine-tune your own based on your specific requirements.
