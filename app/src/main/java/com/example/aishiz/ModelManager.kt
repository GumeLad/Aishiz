package com.example.aishiz

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.InputStream

/**
 * ModelManager handles loading and inference for offline AI models.
 * 
 * This implementation is designed to work with MediaPipe LLM Inference API
 * for maximum flexibility and customization. Once MediaPipe dependency is added,
 * uncomment the MediaPipe-specific code.
 * 
 * To use:
 * 1. Add MediaPipe dependency to build.gradle.kts:
 *    implementation("com.google.mediapipe:tasks-genai:0.10.14")
 * 2. Uncomment the MediaPipe imports and implementation below
 * 3. Load a model using loadModel()
 * 4. Generate responses using generateResponse()
 */
class ModelManager(private val context: Context) {
    
    // Uncomment when MediaPipe dependency is added:
    // private var llmInference: LlmInference? = null
    
    companion object {
        private const val TAG = "ModelManager"
        private const val MAX_MODEL_SIZE_BYTES = 10L * 1024 * 1024 * 1024 // 10GB
    }
    
    /**
     * Configuration for model loading and inference parameters.
     * 
     * @param modelPath Absolute path to the model file (.bin format)
     * @param temperature Controls randomness (0.0 = deterministic, 1.0+ = creative)
     * @param topK Number of top tokens to consider (higher = more diverse)
     * @param topP Cumulative probability threshold (0.9-0.95 recommended)
     * @param maxTokens Maximum number of tokens to generate
     * @param randomSeed Seed for reproducible outputs (0 = random)
     */
    data class ModelConfig(
        val modelPath: String,
        val temperature: Float = 0.8f,
        val topK: Int = 40,
        val topP: Float = 0.95f,
        val maxTokens: Int = 1024,
        val randomSeed: Int = 0
    ) {
        companion object {
            /**
             * Returns recommended configuration based on available memory.
             */
            fun forAvailableMemory(availableMemoryMb: Long): ModelConfig {
                return when {
                    availableMemoryMb < 3000 -> ModelConfig(
                        maxTokens = 256,
                        temperature = 0.7f,
                        topK = 30
                    )
                    availableMemoryMb < 6000 -> ModelConfig(
                        maxTokens = 512,
                        temperature = 0.8f,
                        topK = 40
                    )
                    else -> ModelConfig(
                        maxTokens = 1024,
                        temperature = 0.8f,
                        topK = 40
                    )
                }
            }
            
            /**
             * Preset configurations for different use cases.
             */
            fun creative(modelPath: String) = ModelConfig(
                modelPath = modelPath,
                temperature = 1.2f,
                topK = 50,
                topP = 0.95f
            )
            
            fun precise(modelPath: String) = ModelConfig(
                modelPath = modelPath,
                temperature = 0.3f,
                topK = 10,
                topP = 0.85f
            )
            
            fun balanced(modelPath: String) = ModelConfig(
                modelPath = modelPath,
                temperature = 0.8f,
                topK = 40,
                topP = 0.95f
            )
        }
    }
    
    /**
     * Validates a model file before loading.
     * Checks file size and basic integrity.
     */
    fun validateModelFile(file: File): Result<Unit> {
        return try {
            if (!file.exists()) {
                return Result.failure(IllegalArgumentException("Model file does not exist: ${file.path}"))
            }
            
            if (!file.canRead()) {
                return Result.failure(IllegalArgumentException("Cannot read model file: ${file.path}"))
            }
            
            if (file.length() > MAX_MODEL_SIZE_BYTES) {
                return Result.failure(IllegalArgumentException("Model file too large (max 10GB): ${file.length()} bytes"))
            }
            
            if (file.length() < 1024) {
                return Result.failure(IllegalArgumentException("Model file too small (likely corrupted): ${file.length()} bytes"))
            }
            
            // Check for TFLite magic bytes (optional but recommended)
            file.inputStream().use { input ->
                val header = ByteArray(4)
                val bytesRead = input.read(header)
                if (bytesRead == 4) {
                    // TFLite files start with "TFL3" (0x54 0x46 0x4C 0x33)
                    val isTFLite = header.contentEquals(byteArrayOf(0x54, 0x46, 0x4C, 0x33))
                    if (!isTFLite) {
                        Log.w(TAG, "File does not appear to be a TFLite model (wrong magic bytes)")
                        // Don't fail - some valid models might have different headers
                    }
                }
            }
            
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Loads a model with the specified configuration.
     * Call this before generating responses.
     * 
     * @param config Model configuration including path and inference parameters
     * @return Result indicating success or failure
     */
    suspend fun loadModel(config: ModelConfig): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            Log.i(TAG, "Loading model from: ${config.modelPath}")
            
            // Validate file first
            val file = File(config.modelPath)
            validateModelFile(file).getOrThrow()
            
            // TODO: Uncomment when MediaPipe dependency is added
            /*
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(config.modelPath)
                .setTemperature(config.temperature)
                .setTopK(config.topK)
                .setTopP(config.topP)
                .setMaxTokens(config.maxTokens)
                .setRandomSeed(config.randomSeed)
                .build()
            
            // Close existing instance if any
            llmInference?.close()
            
            // Create new instance
            llmInference = LlmInference.createFromOptions(context, options)
            */
            
            Log.i(TAG, "Model loaded successfully")
            Log.i(TAG, "Config: temp=${config.temperature}, topK=${config.topK}, topP=${config.topP}, maxTokens=${config.maxTokens}")
            
            // For now, just simulate success until MediaPipe is added
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            Result.failure(e)
        }
    }
    
    /**
     * Generates a response for the given prompt.
     * 
     * @param prompt The input text prompt
     * @param onPartialResult Optional callback for streaming responses (called multiple times)
     * @return Result containing the complete generated text or an error
     */
    suspend fun generateResponse(
        prompt: String,
        onPartialResult: ((String) -> Unit)? = null
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            // TODO: Uncomment when MediaPipe dependency is added
            /*
            val inference = llmInference ?: return@withContext Result.failure(
                IllegalStateException("Model not loaded. Call loadModel() first.")
            )
            
            Log.d(TAG, "Generating response for prompt: ${prompt.take(50)}...")
            val fullResponse = StringBuilder()
            
            if (onPartialResult != null) {
                // Streaming response - provides real-time updates
                inference.generateResponseAsync(prompt)
                for (partialResult in inference.generateResponseAsync(prompt)) {
                    partialResult?.let { partial ->
                        fullResponse.append(partial)
                        withContext(Dispatchers.Main) {
                            onPartialResult(partial)
                        }
                    }
                }
            } else {
                // Non-streaming response - waits for complete result
                val response = inference.generateResponse(prompt)
                fullResponse.append(response)
            }
            
            Log.d(TAG, "Generated ${fullResponse.length} characters")
            Result.success(fullResponse.toString())
            */
            
            // Placeholder until MediaPipe is added
            Log.w(TAG, "MediaPipe not yet integrated. Add dependency and uncomment code.")
            Result.failure(IllegalStateException("MediaPipe LLM Inference not yet integrated. See OFFLINE_AI_MODEL_GUIDE.md"))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate response", e)
            Result.failure(e)
        }
    }
    
    /**
     * Generates a response with conversation context.
     * 
     * @param messages List of previous messages (user and assistant)
     * @param newMessage New user message to respond to
     * @param onPartialResult Optional callback for streaming
     * @return Result containing the generated response
     */
    suspend fun generateContextualResponse(
        messages: List<Pair<String, String>>,
        newMessage: String,
        onPartialResult: ((String) -> Unit)? = null
    ): Result<String> {
        // Build prompt with conversation history
        val prompt = buildString {
            // Include last N messages for context
            messages.takeLast(5).forEach { (user, assistant) ->
                append("User: $user\n")
                append("Assistant: $assistant\n")
            }
            append("User: $newMessage\n")
            append("Assistant: ")
        }
        
        return generateResponse(prompt, onPartialResult)
    }
    
    /**
     * Closes the model and releases resources.
     * Call this when done with the model or switching to a different model.
     */
    fun close() {
        // TODO: Uncomment when MediaPipe dependency is added
        /*
        llmInference?.close()
        llmInference = null
        */
        Log.i(TAG, "Model closed")
    }
    
    /**
     * Checks if a model is currently loaded.
     */
    fun isModelLoaded(): Boolean {
        // TODO: Uncomment when MediaPipe dependency is added
        // return llmInference != null
        return false // Placeholder
    }
    
    /**
     * Copies a model file from assets to internal storage.
     * Useful for bundling models with the app.
     * 
     * @param assetPath Path in assets directory (e.g., "models/gemma-2b.bin")
     * @param destinationPath Absolute path where file should be copied
     */
    suspend fun copyModelFromAssets(
        assetPath: String,
        destinationPath: String
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val destFile = File(destinationPath)
            
            // Skip if already exists
            if (destFile.exists() && destFile.length() > 0) {
                Log.i(TAG, "Model already exists at: $destinationPath")
                return@withContext Result.success(destinationPath)
            }
            
            Log.i(TAG, "Copying model from assets: $assetPath")
            
            context.assets.open(assetPath).use { input ->
                destFile.outputStream().use { output ->
                    input.copyTo(output, bufferSize = 8192)
                }
            }
            
            Log.i(TAG, "Model copied successfully (${destFile.length()} bytes)")
            Result.success(destinationPath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy model from assets", e)
            Result.failure(e)
        }
    }
    
    /**
     * Copies a model file from external URI to internal storage.
     * Useful for loading models selected by the user.
     * 
     * @param inputStream Input stream from ContentResolver
     * @param fileName Name for the copied file
     * @return Result containing the absolute path to the copied file
     */
    suspend fun copyModelFromUri(
        inputStream: InputStream,
        fileName: String
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val destFile = File(context.filesDir, fileName)
            
            Log.i(TAG, "Copying model to: ${destFile.absolutePath}")
            
            inputStream.use { input ->
                destFile.outputStream().use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Int
                    var totalBytes = 0L
                    
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        totalBytes += bytesRead
                        
                        // Log progress for large files
                        if (totalBytes % (50 * 1024 * 1024) == 0L) {
                            Log.d(TAG, "Copied ${totalBytes / 1024 / 1024} MB...")
                        }
                    }
                }
            }
            
            Log.i(TAG, "Model copied successfully (${destFile.length()} bytes)")
            Result.success(destFile.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy model from URI", e)
            Result.failure(e)
        }
    }
}
