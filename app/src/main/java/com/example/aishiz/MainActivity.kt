package com.example.aishiz

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.aishiz.databinding.ActivityMainBinding
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val chatAdapter = ChatAdapter(mutableListOf())
    private val executor = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val contentMainBinding = binding.contentMain

        contentMainBinding.messages.layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        contentMainBinding.messages.adapter = chatAdapter

        binding.toolbar.setNavigationOnClickListener {
            binding.drawer.openDrawer(binding.navModels)
        }
        binding.toolbar.setOnMenuItemClickListener {
            binding.drawer.openDrawer(binding.navParams)
            true
        }

        contentMainBinding.sendButton.setOnClickListener {
            val prompt = contentMainBinding.promptInput.text.toString()
            if (prompt.isNotBlank()) {
                addMessage(prompt, Role.USER)
                contentMainBinding.promptInput.text.clear()
                addMessage("...", Role.TYPING)
                startGeneration(prompt)
            }
        }
    }

    private fun addMessage(text: String, role: Role) {
        val message = Message(text, role)
        runOnUiThread {
            chatAdapter.addMessage(message)
            binding.contentMain.messages.scrollToPosition(chatAdapter.itemCount - 1)
        }
    }

    private fun startGeneration(prompt: String) {
        executor.execute {
            val fullResponse = "This is a generated response to your prompt: '$prompt'"
            val words = fullResponse.split(" ")
            var isFirstToken = true

            for (word in words) {
                Thread.sleep(200)
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