package com.example.aishiz

import androidx.lifecycle.ViewModel

class MainViewModel : ViewModel() {
    val messages = mutableListOf<Message>()
}