package com.example.aishiz

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Launch ChatActivity directly - LLM chat is the main feature
        startActivity(Intent(this, ChatActivity::class.java))
        finish()
    }
}
