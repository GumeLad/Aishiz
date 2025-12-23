# Aishiz - Offline AI Assistant

An Android application for running personal AI models completely offline with maximum flexibility and customization.

## Overview

Aishiz allows you to load and run your own AI models directly on your Android device without requiring an internet connection. Built with TensorFlow Lite and designed for MediaPipe LLM Inference API, it provides a flexible foundation for offline conversational AI.

## Features

- ✅ **Fully Offline**: No internet connection required after model loading
- ✅ **Flexible Model Support**: Load any MediaPipe-compatible TFLite model
- ✅ **Customizable Parameters**: Control temperature, top-k, top-p, and more
- ✅ **Token Streaming**: Real-time response generation
- ✅ **Chat Interface**: Clean, intuitive conversation UI
- ✅ **Model Management**: Easy model selection and loading

## Current Status

The application infrastructure is ready with:
- TensorFlow Lite v2.17.0 integration
- Chat UI components (ChatAdapter, Message models)
- ModelManager implementation (ready to activate)
- Native C++ support via NDK
- Example models (MobileNet V2, MobileBERT)

## Quick Start

### 1. Add MediaPipe Dependency

```kotlin
// app/build.gradle.kts
dependencies {
    implementation("com.google.mediapipe:tasks-genai:0.10.14")
}
```

### 2. Enable ModelManager

Uncomment MediaPipe code in `app/src/main/java/com/example/aishiz/ModelManager.kt`

### 3. Load a Model

Download and convert a model (see guides below), then load it through the app UI.

## Documentation

### 📚 Comprehensive Guides

- **[OFFLINE_AI_MODEL_GUIDE.md](OFFLINE_AI_MODEL_GUIDE.md)** - Complete guide to offline AI model integration
  - MediaPipe LLM Inference API setup
  - Model loading and inference
  - Customization options
  - Performance optimization
  - Troubleshooting

- **[QUICK_START.md](QUICK_START.md)** - Step-by-step implementation guide
  - Add dependencies
  - Update UI components
  - Integrate with chat interface
  - Test your implementation

- **[MODEL_CONVERSION_GUIDE.md](MODEL_CONVERSION_GUIDE.md)** - How to convert your models
  - PyTorch to TFLite conversion
  - ONNX to TFLite conversion
  - Quantization techniques
  - Batch conversion scripts
  - Verification tools

- **[RECOMMENDATIONS.md](RECOMMENDATIONS.md)** - Strategic next steps and best practices
  - Priority recommendations
  - Model strategy
  - Architecture guidelines
  - Performance optimization
  - Security best practices

## Recommended Models

### For Testing
- **Gemma 2B IT** (~2GB) - Best balance of quality and speed
- **Phi-2** (~2.7GB) - Strong reasoning capabilities

### For Production
- **Small devices**: TinyLlama 1.1B (~600MB)
- **Modern devices**: Gemma 2B IT (~2GB)
- **Flagship devices**: Gemma 7B IT (~7GB)

## Architecture

```
UI Layer (MainActivity, ChatActivity)
    ↓
ViewModel Layer (MainViewModel)
    ↓
Business Logic (ModelManager)
    ↓
MediaPipe LLM Inference
    ↓
TFLite Model Files
```

## Requirements

- Android SDK 30+
- Minimum 4GB RAM (8GB+ recommended)
- 2GB+ free storage for models
- Android Studio Arctic Fox or later

## Building

```bash
./gradlew assembleDebug
```

## Technology Stack

- **Language**: Kotlin 1.9.24
- **UI**: Android Views with ViewBinding
- **ML Framework**: TensorFlow Lite 2.17.0
- **LLM Engine**: MediaPipe Tasks GenAI 0.10.14
- **Build System**: Gradle 8.13.1
- **Min SDK**: 30
- **Target SDK**: 36

## Project Structure

```
app/
├── src/main/
│   ├── java/com/example/aishiz/
│   │   ├── MainActivity.kt           # Main UI
│   │   ├── MainViewModel.kt          # State management
│   │   ├── ModelManager.kt           # Model loading/inference
│   │   ├── ChatAdapter.kt            # Chat UI adapter
│   │   └── Message.kt                # Message data model
│   ├── ml/                           # Model files
│   │   ├── mobilenet_v2_imagenet.tflite
│   │   └── mobilebert/
│   ├── res/                          # Resources
│   └── AndroidManifest.xml
├── build.gradle.kts
└── ...
```

## Features Roadmap

### Phase 1 (Week 1) ✅
- [x] MediaPipe integration documentation
- [x] ModelManager implementation
- [x] Comprehensive guides created
- [ ] Model loading UI
- [ ] Basic inference testing

### Phase 2 (Weeks 2-4)
- [ ] Streaming responses
- [ ] Customizable parameters UI
- [ ] Multi-model support
- [ ] Model persistence

### Phase 3 (Month 2)
- [ ] Voice integration (STT/TTS)
- [ ] RAG (Retrieval-Augmented Generation)
- [ ] Model benchmarking
- [ ] Advanced context management

### Phase 4 (Future)
- [ ] On-device fine-tuning
- [ ] Function calling / tool use
- [ ] Model compression pipeline
- [ ] Multi-modal support (images, audio)

## Performance Targets

- Model load time: < 10 seconds (2GB model)
- Inference speed: > 5 tokens/second
- Memory usage: < 80% available RAM
- Response latency: < 1 second (first token)

## Contributing

This is a personal project, but suggestions and feedback are welcome via GitHub issues.

## Security

- Models run completely offline (no data leaves device)
- Uses Android scoped storage for model files
- Input sanitization and output validation
- Sandboxed model execution

## License

[Add your license here]

## Support & Resources

### Official Documentation
- [MediaPipe LLM Inference](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)
- [TensorFlow Lite for Android](https://www.tensorflow.org/lite/android)
- [AI Edge Torch](https://github.com/google-ai-edge/ai-edge-torch)

### Model Sources
- [Hugging Face Models](https://huggingface.co/models)
- [Kaggle Models](https://www.kaggle.com/models)
- [Google AI Edge](https://ai.google.dev/edge)

### Community
- [TensorFlow Lite Discussion](https://discuss.tensorflow.org/c/lite)
- [MediaPipe GitHub](https://github.com/google/mediapipe)

## Acknowledgments

- Google MediaPipe team for LLM Inference API
- TensorFlow Lite team for mobile ML framework
- Hugging Face for model hub
- All open-source LLM projects

## Contact

[Add your contact information]

---

**Note**: This application is designed for running AI models locally on your device. Model performance depends on device capabilities. See RECOMMENDATIONS.md for device-specific guidance.
