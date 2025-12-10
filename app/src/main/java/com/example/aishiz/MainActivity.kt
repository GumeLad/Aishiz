package com.example.aishiz

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class MainActivity : AppCompatActivity() {

    private lateinit var messages: RecyclerView
    private lateinit var promptInput: EditText
    private lateinit var sendButton: Button
    private val chatAdapter = ChatAdapter()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        messages = findViewById(R.id.messages)
        promptInput = findViewById(R.id.promptInput)
        sendButton = findViewById(R.id.sendButton)

        messages.layoutManager = LinearLayoutManager(this)
        messages.adapter = chatAdapter

        sendButton.setOnClickListener {
            val prompt = promptInput.text.toString()
            if (prompt.isNotBlank()) {
                addMessage(prompt, "user")
                promptInput.text.clear()
                // TODO: Show typing indicator, start generation, and stream tokens
            }
        }
    }

    private fun addMessage(message: String, type: String) {
        chatAdapter.addMessage(Message(message, type))
        messages.scrollToPosition(chatAdapter.itemCount - 1)
    }
}

data class Message(val text: String, val type: String)

class ChatAdapter : RecyclerView.Adapter<ChatAdapter.MessageViewHolder>() {

    private val messages = mutableListOf<Message>()

    fun addMessage(message: Message) {
        messages.add(message)
        notifyItemInserted(messages.size - 1)
    }

    override fun getItemViewType(position: Int): Int {
        return when (messages[position].type) {
            "user" -> 0
            "assistant" -> 1
            else -> 2 // typing
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MessageViewHolder {
        val layout = when (viewType) {
            0 -> R.layout.item_message_user
            1 -> R.layout.item_message_assistant
            else -> R.layout.item_message_typing
        }
        val view = LayoutInflater.from(parent.context).inflate(layout, parent, false)
        return MessageViewHolder(view)
    }

    override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
        val message = messages[position]
        holder.bind(message)
    }

    override fun getItemCount() = messages.size

    inner class MessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val messageText: TextView? = itemView.findViewById(R.id.messageText)

        fun bind(message: Message) {
            messageText?.text = message.text
        }
    }
}