# Model Conversion Guide

This guide explains how to convert various AI model formats to work with your Aishiz application.

## Overview

To use a personal AI model offline on Android, it must be in **MediaPipe-compatible TFLite format** (`.bin` or `.tflite`).

### Supported Source Formats

- ✅ PyTorch (`.pt`, `.pth`)
- ✅ Hugging Face Transformers
- ✅ ONNX (`.onnx`)
- ✅ TensorFlow SavedModel
- ✅ GGUF (via intermediate conversion)
- ⚠️ TensorFlow.js (requires conversion)

## Method 1: Using AI Edge Torch (Recommended)

**Best for**: PyTorch models and Hugging Face Transformers

### Prerequisites

```bash
pip install ai-edge-torch
pip install torch transformers
pip install numpy
```

### Convert Gemma 2B

```python
import ai_edge_torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load the model
model_id = "google/gemma-2b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use FP16 for smaller size
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Prepare sample inputs for tracing
sample_input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.long)

# 3. Convert to Edge model
edge_model = ai_edge_torch.convert(
    model,
    (sample_input_ids,)
)

# 4. Export to TFLite
edge_model.export("gemma-2b-it.tflite")

print("Conversion complete! File: gemma-2b-it.tflite")
```

### Convert Phi-2

```python
import ai_edge_torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load Phi-2
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# 2. Sample input
sample_input = torch.randint(0, 50000, (1, 128))

# 3. Convert
edge_model = ai_edge_torch.convert(model, (sample_input,))

# 4. Export
edge_model.export("phi-2.tflite")
print("Phi-2 converted successfully!")
```

### Convert Custom PyTorch Model

```python
import ai_edge_torch
import torch

# Load your model
model = torch.load("your_model.pt")
model.eval()

# Create sample input matching your model's input shape
sample_input = torch.randn(1, 128, 768)  # Adjust shape as needed

# Convert
edge_model = ai_edge_torch.convert(model, (sample_input,))

# Export
edge_model.export("custom_model.tflite")
```

## Method 2: Using TensorFlow Lite Converter

**Best for**: TensorFlow models

### Prerequisites

```bash
pip install tensorflow
```

### Convert TensorFlow SavedModel

```python
import tensorflow as tf

# 1. Load SavedModel
model = tf.saved_model.load("path/to/saved_model")

# 2. Create converter
converter = tf.lite.TFLiteConverter.from_saved_model("path/to/saved_model")

# 3. Set optimization (optional but recommended)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Convert
tflite_model = converter.convert()

# 5. Save
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion complete!")
```

### Convert Keras Model

```python
import tensorflow as tf

# 1. Load model
model = tf.keras.models.load_model("model.h5")

# 2. Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 3. Save
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

## Method 3: Using ONNX

**Best for**: Models exported from various frameworks

### Prerequisites

```bash
pip install onnx tf2onnx tensorflow
```

### PyTorch → ONNX → TFLite

```python
import torch
import onnx
import tf2onnx
import tensorflow as tf

# Step 1: PyTorch → ONNX
model = torch.load("model.pt")
model.eval()

dummy_input = torch.randn(1, 128, 768)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

# Step 2: ONNX → TensorFlow
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")

# Step 3: TensorFlow → TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

## Method 4: Quantization for Smaller Size

### Post-Training Quantization (Recommended)

Reduces model size by ~4x with minimal accuracy loss:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("model")

# Enable default optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Set supported ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # TensorFlow ops
]

tflite_model = converter.convert()

with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_model)
```

### Dynamic Range Quantization

```python
converter = tf.lite.TFLiteConverter.from_saved_model("model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# This will quantize weights to int8
tflite_model = converter.convert()
```

### Full Integer Quantization

For even smaller size (requires representative dataset):

```python
def representative_dataset():
    for _ in range(100):
        # Generate or load representative data
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_saved_model("model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
```

## Method 5: Converting GGUF Models

**Note**: GGUF is primarily for llama.cpp. Conversion is more complex.

### Approach 1: Via ONNX

```bash
# 1. Convert GGUF to PyTorch using llama.cpp tools
python convert-hf-to-gguf.py model_name --outtype f16

# 2. Load in PyTorch and export to ONNX
# (Similar to Method 3)

# 3. Convert ONNX to TFLite
```

### Approach 2: Use Pre-converted Models

It's often easier to find models already in HuggingFace format and convert those:

```python
# Instead of converting GGUF, use HuggingFace version
from transformers import AutoModelForCausalLM

# Example: LLaMA models are available in HF format
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# Then use Method 1 (AI Edge Torch) to convert
```

## Verification Tools

### Check Model Information

```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
print("Input shape:", input_details[0]['shape'])
print("Input type:", input_details[0]['dtype'])

# Get output details
output_details = interpreter.get_output_details()
print("Output shape:", output_details[0]['shape'])
print("Output type:", output_details[0]['dtype'])

# Get model size
import os
size_mb = os.path.getsize("model.tflite") / (1024 * 1024)
print(f"Model size: {size_mb:.2f} MB")
```

### Test Model Inference

```python
import numpy as np
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Prepare input
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
input_data = np.random.randn(*input_shape).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Inference successful!")
print("Output shape:", output_data.shape)
print("Output sample:", output_data[:5])
```

## Size Optimization Tips

### 1. Use Lower Precision

```python
# FP32 → FP16 (50% size reduction)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

### 2. Prune Unnecessary Layers

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Prune model before conversion
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
```

### 3. Knowledge Distillation

Train a smaller "student" model to mimic a larger "teacher" model:

```python
# This is more complex and requires training code
# See: https://keras.io/examples/vision/knowledge_distillation/
```

## Common Issues and Solutions

### Issue: "Unsupported operations in model"

**Solution**: Convert with SELECT_TF_OPS support:

```python
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
```

### Issue: Model too large (> 2GB)

**Solutions**:
1. Use quantization (INT8 instead of FP32)
2. Use a smaller base model (2B instead of 7B)
3. Prune unnecessary layers
4. Use distillation to create a smaller model

### Issue: Conversion fails with memory error

**Solutions**:
```python
# Use lower memory options
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
```

### Issue: Poor accuracy after quantization

**Solutions**:
1. Use dynamic range quantization instead of full integer
2. Provide better representative dataset
3. Use hybrid quantization (keep some layers in FP32)

## Recommended Models for Mobile

### Small (< 1GB) - Fast inference, basic capabilities
- **TinyLlama-1.1B**: ~600MB, good for simple tasks
- **DistilGPT-2**: ~350MB, basic conversation
- **MobileBERT**: ~100MB, Q&A and classification

### Medium (1-3GB) - Balanced performance
- **Gemma-2B-IT**: ~2GB, excellent instruction following
- **Phi-2**: ~2.7GB, strong reasoning
- **StableLM-2-1.6B**: ~1.6GB, good general purpose

### Large (3-8GB) - Best quality, slower
- **Gemma-7B-IT**: ~7GB, high-quality responses
- **LLaMA-7B**: ~7GB, very capable
- **Mistral-7B**: ~7GB, excellent performance

## Batch Conversion Script

For converting multiple models:

```python
import os
import ai_edge_torch
from transformers import AutoModelForCausalLM
import torch

models_to_convert = [
    "google/gemma-2b-it",
    "microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
]

for model_id in models_to_convert:
    try:
        print(f"Converting {model_id}...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Sample input
        sample_input = torch.randint(0, 1000, (1, 128))
        
        # Convert
        edge_model = ai_edge_torch.convert(model, (sample_input,))
        
        # Generate filename
        output_name = model_id.replace("/", "_") + ".tflite"
        
        # Export
        edge_model.export(output_name)
        
        print(f"✓ Converted {model_id} to {output_name}")
        
        # Clean up
        del model
        del edge_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ Failed to convert {model_id}: {e}")

print("Batch conversion complete!")
```

## Advanced: Custom Model Architecture

If you have a custom architecture, ensure compatibility:

```python
import torch
import torch.nn as nn

class CustomLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture here
        # Avoid operations not supported by TFLite
        
    def forward(self, x):
        # Forward pass
        return x

# Test conversion
model = CustomLLM()
model.eval()

sample_input = torch.randn(1, 128, 512)

try:
    edge_model = ai_edge_torch.convert(model, (sample_input,))
    edge_model.export("custom.tflite")
    print("✓ Custom model converted successfully!")
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    print("Check for unsupported operations")
```

## Next Steps

After conversion:

1. **Verify**: Test the model with verification tools
2. **Optimize**: Apply quantization if not done
3. **Test**: Try inference on desktop first
4. **Transfer**: Copy to Android device
5. **Integrate**: Load in Aishiz app
6. **Benchmark**: Measure performance and memory usage

## Resources

- **AI Edge Torch**: https://github.com/google-ai-edge/ai-edge-torch
- **TFLite Converter**: https://www.tensorflow.org/lite/convert
- **Model Optimization**: https://www.tensorflow.org/model_optimization
- **Hugging Face**: https://huggingface.co/models
- **MediaPipe Models**: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference

## Support

For conversion issues:
1. Check TensorFlow/PyTorch versions compatibility
2. Review error messages for unsupported ops
3. Try with a known-good model first (e.g., Gemma-2B)
4. Search TensorFlow Lite GitHub issues
