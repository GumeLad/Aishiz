package com.example.aishiz

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.aishiz.databinding.ItemMessageAssistantBinding
import com.example.aishiz.databinding.ItemMessageTypingBinding
import com.example.aishiz.databinding.ItemMessageUserBinding

class ChatAdapter(private val messages: MutableList<Message>) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    companion object {
        private const val VIEW_TYPE_USER = 1
        private const val VIEW_TYPE_ASSISTANT = 2
        private const val VIEW_TYPE_TYPING = 3
    }

    override fun getItemCount(): Int = messages.size

    override fun getItemViewType(position: Int): Int {
        return when (messages[position].role) {
            Role.USER -> VIEW_TYPE_USER
            Role.ASSISTANT -> VIEW_TYPE_ASSISTANT
            Role.TYPING -> VIEW_TYPE_TYPING
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return when (viewType) {
            VIEW_TYPE_USER -> UserViewHolder(
                ItemMessageUserBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_TYPING -> TypingViewHolder(
                ItemMessageTypingBinding.inflate(inflater, parent, false)
            )
            else -> AssistantViewHolder(
                ItemMessageAssistantBinding.inflate(inflater, parent, false)
            )
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val item = messages[position]
        when (holder) {
            is UserViewHolder -> holder.bind(item)
            is AssistantViewHolder -> holder.bind(item)
            is TypingViewHolder -> { /* No binding needed */ }
        }
    }

    fun addMessage(message: Message) {
        messages.add(message)
        notifyItemInserted(messages.size - 1)
    }

    fun removeTypingIndicator() {
        if (messages.isNotEmpty() && messages.last().role == Role.TYPING) {
            val lastIndex = messages.size - 1
            messages.removeAt(lastIndex)
            notifyItemRemoved(lastIndex)
        }
    }

    fun appendContentToLastMessage(content: String) {
        if (messages.isNotEmpty() && messages.last().role == Role.ASSISTANT) {
            val lastMessage = messages.last()
            lastMessage.text += content
            notifyItemChanged(messages.size - 1)
        }
    }

    class UserViewHolder(private val binding: ItemMessageUserBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(message: Message) {
            binding.messageText.text = message.text
        }
    }

    class AssistantViewHolder(private val binding: ItemMessageAssistantBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(message: Message) {
            binding.messageText.text = message.text
        }
    }

    class TypingViewHolder(private val binding: ItemMessageTypingBinding) : RecyclerView.ViewHolder(binding.root)
}