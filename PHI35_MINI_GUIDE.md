# Phi-3.5-mini-instruct Model Integration Guide

## Overview

Microsoft's **Phi-3.5-mini-instruct** is an excellent choice for on-device AI with:
- **Size**: ~2.7GB (4-bit quantized) to ~7.8GB (FP16)
- **Performance**: High-quality instruction following
- **Efficiency**: Optimized for edge devices
- **Context**: 128K token context window

## Download and Conversion

### Step 1: Download the GGUF Model

Visit: https://huggingface.co/MaziyarPanahi/Phi-3.5-mini-instruct-GGUF/tree/main

Recommended versions:
- **For most devices**: `Phi-3.5-mini-instruct.Q4_K_M.gguf` (~2.7GB)
- **For high-end devices**: `Phi-3.5-mini-instruct.Q8_0.gguf` (~4.9GB)
- **Best quality**: `Phi-3.5-mini-instruct.f16.gguf` (~7.8GB)

Download directly from HuggingFace:
```bash
# Using huggingface-cli (recommended)
pip install huggingface-hub
huggingface-cli download MaziyarPanahi/Phi-3.5-mini-instruct-GGUF \
    Phi-3.5-mini-instruct.Q4_K_M.gguf \
    --local-dir ./models

# Or download via wget
wget https://huggingface.co/MaziyarPanahi/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct.Q4_K_M.gguf
```

### Step 2: Convert GGUF to MediaPipe TFLite

GGUF models need to be converted to TFLite format for use with MediaPipe.

#### Option A: Using AI Edge Torch (Recommended)

```python
# Install dependencies
pip install ai-edge-torch
pip install torch transformers
pip install gguf

# Convert GGUF to PyTorch, then to TFLite
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ai_edge_torch

# Load the non-GGUF version from HuggingFace
model_id = "microsoft/Phi-3.5-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Prepare sample input
sample_input = torch.randint(0, 50000, (1, 128))

# Convert to Edge model
edge_model = ai_edge_torch.convert(model, (sample_input,))

# Export to TFLite
edge_model.export("phi35_mini_instruct.tflite")

print("✓ Conversion complete!")
```

#### Option B: Use Pre-converted Models

Check if a pre-converted TFLite version is available:
- [Google AI Edge Models](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference#models)
- [Kaggle Models](https://www.kaggle.com/models)

### Step 3: Transfer to Android Device

1. **Via USB (ADB)**:
```bash
adb push phi35_mini_instruct.tflite /sdcard/Download/
```

2. **Via Cloud Storage**: Upload to Google Drive, Dropbox, etc., then download on device

3. **Direct Download**: Host on a server and download directly in the app

## Loading in Aishiz App

### Method 1: Using the App UI

1. Open the Aishiz app
2. Tap "Open AI Chat"
3. Tap "Load Model"
4. Navigate to and select your `phi35_mini_instruct.tflite` file
5. Wait for the model to load (~5-10 seconds)
6. Start chatting!

### Method 2: Bundling with App

If you want to include the model in your APK:

1. Place the model file in `app/src/main/assets/models/`:
```
app/src/main/assets/
  └── models/
      └── phi35_mini_instruct.tflite
```

2. Add code to copy from assets on first launch:
```kotlin
// In ChatActivity.onCreate()
lifecycleScope.launch {
    val modelPath = filesDir.absolutePath + "/phi35_mini_instruct.tflite"
    modelManager.copyModelFromAssets(
        "models/phi35_mini_instruct.tflite",
        modelPath
    ).fold(
        onSuccess = { path ->
            loadModel(path)
        },
        onFailure = { error ->
            Log.e("Chat", "Failed to load bundled model", error)
        }
    )
}
```

**Note**: Models over 100MB may require APK expansion files (OBB).

## Recommended Settings for Phi-3.5-mini

### Preset Configurations

**Precise (Q&A, Facts)**:
- Temperature: 0.3
- Top-K: 10
- Top-P: 0.85
- Max Tokens: 512

**Balanced (General Chat)**:
- Temperature: 0.8
- Top-K: 40
- Top-P: 0.95
- Max Tokens: 1024

**Creative (Stories, Ideas)**:
- Temperature: 1.2
- Top-K: 50
- Top-P: 0.95
- Max Tokens: 2048

### Adjust via Model Settings Dialog

1. Tap "Model Settings" button
2. Use sliders to adjust parameters
3. Or tap preset buttons (Precise/Balanced/Creative)
4. Tap "Apply" to reload model with new settings

## Performance Expectations

### On Mid-Range Device (6GB RAM)

**Q4_K_M (2.7GB) - Recommended**:
- Load time: 5-8 seconds
- First token: 0.5-0.8 seconds
- Inference speed: 5-7 tokens/second
- Memory usage: 3-4GB peak
- Battery: 3-4% per 10 min

### On High-End Device (8GB+ RAM)

**Q8_0 (4.9GB)**:
- Load time: 8-12 seconds
- First token: 0.6-1.0 seconds
- Inference speed: 4-6 tokens/second
- Memory usage: 5-6GB peak
- Battery: 4-5% per 10 min

### On Flagship Device (12GB+ RAM)

**F16 (7.8GB)**:
- Load time: 12-18 seconds
- First token: 0.8-1.2 seconds
- Inference speed: 3-5 tokens/second
- Memory usage: 8-9GB peak
- Battery: 5-6% per 10 min

## Troubleshooting

### "Failed to load model"
- **Check file size**: Ensure the file isn't corrupted (should be 2-8GB)
- **Check format**: Must be `.tflite` or `.bin` format, not `.gguf`
- **Check permissions**: App needs read access to file location

### "Out of memory"
- **Use smaller quantization**: Try Q4_K_M instead of F16
- **Reduce max tokens**: Set to 256 or 512 instead of 1024
- **Close other apps**: Free up device memory

### "Model produces gibberish"
- **Check conversion**: Ensure model was converted correctly
- **Try different settings**: Reduce temperature to 0.5-0.7
- **Verify model**: Re-download and convert if needed

### "Slow inference"
- **Normal for large models**: 3-5 tokens/sec is expected for Phi-3.5
- **Check battery saver**: Disable low-power mode
- **Reduce max tokens**: Lower values generate faster
- **Use Q4 quantization**: Smaller models run faster

## Model Capabilities

Phi-3.5-mini-instruct excels at:

✅ **Instruction Following**: Very good at understanding and following complex instructions
✅ **Reasoning**: Strong logical reasoning and problem-solving
✅ **Coding**: Can help with code explanations and simple programming tasks
✅ **Q&A**: Accurate factual question answering
✅ **Summarization**: Effective text summarization
✅ **Math**: Basic to intermediate mathematical problems
✅ **Context**: 128K context window supports long conversations

Limitations:
❌ **Not for**: Real-time translation, very specialized domains
❌ **Cutoff Date**: Training data up to October 2023
❌ **No Internet**: Completely offline, can't access current information

## Prompt Engineering Tips

### Effective Prompts

**Good**:
```
Explain quantum entanglement in simple terms that a high school student would understand.
```

**Better**:
```
You are a physics teacher. Explain quantum entanglement to a high school student using analogies and avoiding complex mathematical formulas.
```

### System Instructions

For consistent behavior, start conversations with:
```
You are a helpful, respectful, and honest AI assistant. Always answer as helpfully as possible while being safe and ethical.
```

## Comparison with Other Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **Phi-3.5-mini** | 2.7GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Balanced performance |
| Gemma 2B | 2GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Speed-critical apps |
| TinyLlama 1.1B | 600MB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Low-end devices |
| Gemma 7B | 7GB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | High-end devices |

## Additional Resources

- **Model Card**: https://huggingface.co/microsoft/Phi-3.5-mini-instruct
- **GGUF Variants**: https://huggingface.co/MaziyarPanahi/Phi-3.5-mini-instruct-GGUF
- **Phi-3 Cookbook**: https://github.com/microsoft/Phi-3CookBook
- **MediaPipe Docs**: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference

## Next Steps

1. ✅ Download Phi-3.5-mini-instruct.Q4_K_M.gguf
2. ✅ Convert to TFLite format
3. ✅ Load in Aishiz app
4. ✅ Adjust settings via Model Settings dialog
5. ✅ Start chatting and experimenting!

For more advanced features like RAG, fine-tuning, or multi-modal support, see the main documentation guides.
