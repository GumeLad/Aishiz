plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

val tensorflowLiteVersion by extra("2.17.0")

android {
    namespace = "com.example.aishiz"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.aishiz"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags("")
            }
        }
        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
        }
    }

    buildTypes {
        release {
                                                isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
        mlModelBinding = true
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildToolsVersion = "36.1.0"
    ndkVersion = "27.0.12077973"
    dependenciesInfo {
        includeInApk = true
        includeInBundle = true
    }
}


dependencies {
    // Core TF Lite runtime (version catalog)
    implementation(libs.tflite)
    implementation(libs.tflitesupport)
    implementation(libs.tensorflow.lite.metadata)

    // MediaPipe for LLM inference
    implementation("com.google.mediapipe:tasks-genai:0.10.14")

    // AndroidX / Material basics
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.recyclerview)
    implementation(libs.androidx.fragment)
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.activity:activity-ktx:1.9.3")

    testImplementation(libs.junit)

    // Android instrumented testing
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}