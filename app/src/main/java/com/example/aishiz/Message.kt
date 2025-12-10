package com.example.aishiz

/**
 * Simple chat message model representing either a user or assistant message.
 */
enum class Role { USER, ASSISTANT, TYPING }

data class Message(
    val text: String,
    val role: Role
)
