package com.example.aishiz

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch

class MainViewModel : ViewModel() {
    val messages = mutableListOf<Message>()
    private var modelManager: ModelManager? = null
    private val conversationHistory = mutableListOf<Pair<String, String>>()
    
    fun setModelManager(manager: ModelManager) {
        modelManager = manager
    }
    
    fun sendMessage(
        userMessage: String,
        onUpdate: () -> Unit
    ) = viewModelScope.launch {
        val manager = modelManager ?: run {
            messages.add(Message("Error: No model loaded", Role.ASSISTANT))
            onUpdate()
            return@launch
        }
        
        // Add user message
        messages.add(Message(userMessage, Role.USER))
        onUpdate()
        
        // Add typing indicator
        messages.add(Message("", Role.TYPING))
        onUpdate()
        
        try {
            // Generate response with streaming
            val fullResponse = StringBuilder()
            
            manager.generateContextualResponse(
                messages = conversationHistory,
                newMessage = userMessage,
                onPartialResult = { partial ->
                    fullResponse.append(partial)
                    
                    // Update typing indicator with partial response
                    val lastIndex = messages.indexOfLast { it.role == Role.TYPING }
                    if (lastIndex >= 0) {
                        messages[lastIndex] = Message(fullResponse.toString(), Role.ASSISTANT)
                        onUpdate()
                    }
                }
            ).fold(
                onSuccess = { response ->
                    // Remove typing indicator
                    messages.removeAll { it.role == Role.TYPING }
                    
                    // Ensure final message is present
                    if (messages.lastOrNull()?.text != response) {
                        messages.add(Message(response, Role.ASSISTANT))
                    }
                    
                    // Add to conversation history
                    conversationHistory.add(userMessage to response)
                    
                    onUpdate()
                },
                onFailure = { error ->
                    // Remove typing indicator
                    messages.removeAll { it.role == Role.TYPING }
                    
                    // Add error message
                    messages.add(Message("Error: ${error.message}", Role.ASSISTANT))
                    onUpdate()
                }
            )
        } catch (e: Exception) {
            // Remove typing indicator
            messages.removeAll { it.role == Role.TYPING }
            
            // Add error message
            messages.add(Message("Error: ${e.message}", Role.ASSISTANT))
            onUpdate()
        }
    }
    
    fun clearConversation() {
        messages.clear()
        conversationHistory.clear()
    }
}
