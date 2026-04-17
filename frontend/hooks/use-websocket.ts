'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { createQueryWebSocket, type StreamingMessage, type ChatMessage } from '@/lib/api'

export function useWebSocket(datasetId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const wsRef = useRef<ReturnType<typeof createQueryWebSocket> | null>(null)
  const messageIdRef = useRef(0)

  useEffect(() => {
    if (!datasetId) return

    const ws = createQueryWebSocket(
      datasetId,
      // onChunk
      (data: StreamingMessage) => {
        setIsStreaming(true)
        setMessages(prev => {
          const last = prev[prev.length - 1]
          if (last && last.role === 'assistant' && last.isStreaming) {
            return [
              ...prev.slice(0, -1),
              { ...last, content: last.content + (data.content || '') }
            ]
          }
          return prev
        })
      },
      // onComplete
      (data: StreamingMessage) => {
        setIsStreaming(false)
        setMessages(prev => {
          const last = prev[prev.length - 1]
          if (last && last.role === 'assistant' && data.data) {
            return [
              ...prev.slice(0, -1),
              {
                ...last,
                content: data.data?.response?.chat?.answer || data.data?.response?.answer || last.content,
                tables: data.data?.response?.tables,
                charts: data.data?.response?.plots,
                insights: data.data?.response?.insights?.map((i: { text: string }) => i.text),
                sources: data.data?.response?.sources,
              }
            ]
          }
          return prev
        })
      },
      // onError
      (error: any) => {
        console.error('WebSocket error:', error)
        setIsConnected(false)
        setIsStreaming(false)
      }
    )

    wsRef.current = ws
    setIsConnected(true)

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
      wsRef.current = null
      setIsConnected(false)
    }
  }, [datasetId])

  const sendMessage = useCallback(async (query: string) => {
    const messageId = ++messageIdRef.current
    const userMessage: ChatMessage = {
      role: 'user',
      content: query,
    }
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: '',
      isStreaming: true,
    }

    setMessages(prev => [...prev, userMessage, assistantMessage])

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ query }))
    }
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return {
    messages,
    isConnected,
    isStreaming,
    sendMessage,
    clearMessages,
  }
}