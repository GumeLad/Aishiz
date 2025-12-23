package com.example.aishiz

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var viewModel: MainViewModel

    private lateinit var messages: RecyclerView
    private lateinit var promptInput: EditText
    private lateinit var sendButton: Button
    private val chatAdapter by lazy { ChatAdapter(viewModel.messages) }
    private val executor = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewModel = ViewModelProvider(this)[MainViewModel::class.java]

        messages = findViewById(R.id.messages)
        promptInput = findViewById(R.id.promptInput)
        sendButton = findViewById(R.id.sendButton)

        messages.layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        messages.adapter = chatAdapter

        sendButton.setOnClickListener {
            val prompt = promptInput.text.toString()
            if (prompt.isNotBlank()) {
                addMessage(prompt, Role.USER)
                promptInput.text.clear()
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
            messages.scrollToPosition(chatAdapter.itemCount - 1)
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