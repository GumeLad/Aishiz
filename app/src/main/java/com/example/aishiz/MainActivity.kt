package com.example.aishiz

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.aishiz.databinding.ActivityMainBinding
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: MainViewModel

    private val chatAdapter by lazy { ChatAdapter(viewModel.messages) }
    private val executor = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        viewModel = ViewModelProvider(this)[MainViewModel::class.java]

        binding.messages.layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        binding.messages.adapter = chatAdapter

        binding.sendButton.setOnClickListener {
            val prompt = binding.promptInput.text.toString()
            if (prompt.isNotBlank()) {
                addMessage(prompt, Role.USER)
                binding.promptInput.text.clear()
                // Show typing indicator
                addMessage("...", Role.TYPING)
                // Start generation
                startGeneration(prompt)
            }
        }
    }

    private fun addMessage(text: String, role: Role) {
        val message = Message(text, role)
        runOnUiThread {
            chatAdapter.addMessage(message)
            binding.messages.scrollToPosition(chatAdapter.itemCount - 1)
        }
    }

    private fun startGeneration(prompt: String) {
        executor.execute {
            // Simulate token generation
            val fullResponse = "This is a generated response to your prompt: '$prompt'"
            val words = fullResponse.split(" ")
            var isFirstToken = true

            for (word in words) {
                Thread.sleep(200) // Simulate delay for each token
                if (isFirstToken) {
                    isFirstToken = false
                    handler.post {
                        chatAdapter.removeTypingIndicator()
                        addMessage("$word ", Role.ASSISTANT)
                    }
                } else {
                    handler.post {
                        chatAdapter.appendContentToLastMessage("$word ")
                    }
                }
            }
        }
    }
}