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
      (data: StreamingMessage) => {
        if (data.type === 'chunk') {
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
        } else if (data.type === 'complete' && data.data) {
          setIsStreaming(false)
          setMessages(prev => {
            const last = prev[prev.length - 1]
            if (last && last.role === 'assistant') {
              return [
                ...prev.slice(0, -1),
                {
                  ...last,
                  content: data.data?.response?.chat?.answer || data.data?.response?.answer || last.content,
                  isStreaming: false,
                  sources: data.data?.response?.sources,
                  tables: data.data?.response?.tables,
                  charts: data.data?.response?.plots,
                  insights: data.data?.response?.insights?.map(i => i.text),
                }
              ]
            }
            return prev
          })
        } else if (data.type === 'error') {
          setIsStreaming(false)
          setMessages(prev => [
            ...prev,
            {
              id: `error-${messageIdRef.current++}`,
              role: 'assistant',
              content: `Error: ${data.error}`,
              timestamp: new Date(),
              isStreaming: false,
            }
          ])
        }
      },
      (error) => {
        console.error('WebSocket error:', error)
        setIsConnected(false)
        setIsStreaming(false)
      },
      () => {
        setIsConnected(false)
        setIsStreaming(false)
      }
    )

    wsRef.current = ws
    setIsConnected(true)

    return () => {
      ws.close()
      wsRef.current = null
      setIsConnected(false)
    }
  }, [datasetId])

  const sendQuery = useCallback((query: string) => {
    const id = `msg-${messageIdRef.current++}`
    
    setMessages(prev => [
      ...prev,
      {
        id,
        role: 'user',
        content: query,
        timestamp: new Date(),
      }
    ])

    setTimeout(() => {
      setMessages(prev => [
        ...prev,
        {
          id: `assistant-${messageIdRef.current++}`,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        }
      ])
    }, 50)

    wsRef.current?.sendQuery(query)
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return {
    messages,
    isConnected,
    isStreaming,
    sendQuery,
    clearMessages,
  }
}
