'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useChat } from 'ai/react'
import { Card, ScrollArea, Button } from '@/components/ui'
import { Send, Loader } from 'lucide-react'

export function ChatInterface() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat()
  
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex h-screen bg-gradient-to-b from-gray-900 to-gray-800"
    >
      {/* Chat Sidebar */}
      <div className="w-64 bg-gray-800 p-4">
        <ChatSidebar />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages Area */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} {...message} />
            ))}
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="p-4 border-t border-gray-700">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              value={input}
              onChange={handleInputChange}
              placeholder="Ask your legal question..."
              className="flex-1 bg-gray-700 rounded-lg px-4 py-2"
            />
            <Button type="submit" disabled={isLoading}>
              {isLoading ? <Loader className="animate-spin" /> : <Send />}
            </Button>
          </form>
        </div>
      </div>
    </motion.div>
  )
} 