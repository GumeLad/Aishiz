package com.example.aishiz

enum class Role {
    USER, ASSISTANT, TYPING
}

data class Message(var text: String, val role: Role)
