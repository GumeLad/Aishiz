plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.example.AiShiz"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.offlineai"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    buildToolsVersion = "35.0.0"
    ndkVersion = "27.0.12077973"
}


dependencies {
implementation("androidx.core:core-ktx:1.17.0")
implementation("androidx.appcompat:appcompat:1.7.1")
}
