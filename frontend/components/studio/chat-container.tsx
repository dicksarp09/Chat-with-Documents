'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, ChevronDown, BarChart3, Table2, Lightbulb } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useWebSocket } from '@/hooks/use-websocket'
import { queryDataset } from '@/lib/api'
import { DataTable } from './data-table'
import { ChartView } from './chart-view'
import { cn } from '@/lib/utils'

interface Dataset {
  dataset_id: string
  type: 'csv' | 'document'
  filename: string
}

interface ChatContainerProps {
  datasets: Dataset[]
  selectedDataset: string | null
  onSelectDataset: (id: string) => void
}

export function ChatContainer({ datasets, selectedDataset, onSelectDataset }: ChatContainerProps) {
  const [query, setQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [messages, setMessages] = useState<Array<{
    id: string
    role: 'user' | 'assistant'
    content: string
    timestamp: Date
    isStreaming?: boolean
    tables?: Array<{ columns: string[]; rows: (string | number)[][] }>
    charts?: Array<{ type: string; x: string; y: string; title?: string; data: { name: string; value: number }[] }>
    insights?: string[]
    sources?: string[]
  }>>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || !selectedDataset || isLoading) return

    const userMessage = query.trim()
    setQuery('')
    setIsLoading(true)

    // Add user message
    setMessages(prev => [
      ...prev,
      {
        id: `user-${Date.now()}`,
        role: 'user',
        content: userMessage,
        timestamp: new Date(),
      }
    ])

    // Add placeholder for assistant
    const assistantId = `assistant-${Date.now()}`
    setMessages(prev => [
      ...prev,
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      }
    ])

    try {
      // For now, use REST API (WebSocket can be added later)
      const response = await queryDataset(selectedDataset, userMessage)
      
      const answer = response.response?.chat?.answer || response.response?.answer || 'No response generated'
      
      setMessages(prev => prev.map(msg => 
        msg.id === assistantId
          ? {
              ...msg,
              content: answer,
              isStreaming: false,
              tables: response.response?.tables,
              charts: response.response?.plots,
              insights: response.response?.insights?.map((i: { text: string }) => i.text),
              sources: response.response?.sources,
            }
          : msg
      ))
    } catch (error) {
      setMessages(prev => prev.map(msg =>
        msg.id === assistantId
          ? {
              ...msg,
              content: `Error: ${error instanceof Error ? error.message : 'Failed to process query'}`,
              isStreaming: false,
            }
          : msg
      ))
    } finally {
      setIsLoading(false)
    }
  }

  if (datasets.length === 0) {
    return (
      <Card variant="paper">
        <CardContent className="py-16 text-center">
          <div className="w-16 h-16 mx-auto rounded-2xl bg-background-secondary flex items-center justify-center mb-4">
            <Bot className="w-8 h-8 text-foreground-muted" />
          </div>
          <h3 className="font-serif text-xl font-medium text-foreground mb-2">
            No datasets to chat with
          </h3>
          <p className="text-foreground-muted max-w-sm mx-auto">
            Upload a CSV file or document first, then select it below to start asking questions.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Dataset Selector */}
      <div className="lg:col-span-1">
        <Card variant="paper" className="sticky top-24">
          <CardContent className="p-4">
            <h3 className="font-serif font-medium text-foreground mb-3">Select Dataset</h3>
            <div className="space-y-2">
              {datasets.map(dataset => (
                <button
                  key={dataset.dataset_id}
                  onClick={() => onSelectDataset(dataset.dataset_id)}
                  className={cn(
                    'w-full text-left px-3 py-2 rounded-lg transition-all duration-150',
                    selectedDataset === dataset.dataset_id
                      ? 'bg-accent/10 text-accent'
                      : 'hover:bg-background-secondary text-foreground-secondary hover:text-foreground'
                  )}
                >
                  <div className="flex items-center gap-2">
                    {dataset.type === 'csv' ? (
                      <Table2 className="w-4 h-4 shrink-0" />
                    ) : (
                      <BarChart3 className="w-4 h-4 shrink-0" />
                    )}
                    <span className="text-sm truncate">{dataset.filename}</span>
                  </div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Chat Area */}
      <div className="lg:col-span-3">
        <Card variant="paper" className="h-[calc(100vh-16rem)] flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <div className="w-20 h-20 rounded-3xl bg-accent/10 flex items-center justify-center mb-6">
                  <Bot className="w-10 h-10 text-accent" />
                </div>
                <h3 className="font-serif text-2xl font-medium text-foreground mb-3">
                  Ask me anything
                </h3>
                <p className="text-foreground-muted max-w-md">
                  {selectedDataset
                    ? 'Start a conversation about your data. I can analyze patterns, answer questions, and generate insights.'
                    : 'Select a dataset from the panel to start chatting.'}
                </p>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={message.id}
                className={cn(
                  'flex gap-4 animate-fade-up',
                  message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                )}
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <div className={cn(
                  'w-10 h-10 rounded-xl flex items-center justify-center shrink-0',
                  message.role === 'user' ? 'bg-accent' : 'bg-background-secondary'
                )}>
                  {message.role === 'user' ? (
                    <User className="w-5 h-5 text-white" />
                  ) : (
                    <Bot className="w-5 h-5 text-foreground-muted" />
                  )}
                </div>

                <div className={cn(
                  'flex-1 space-y-4',
                  message.role === 'user' ? 'text-right' : ''
                )}>
                  <div className={cn(
                    'inline-block max-w-[85%] rounded-2xl px-5 py-3',
                    message.role === 'user'
                      ? 'bg-accent text-white'
                      : 'bg-background-secondary text-foreground'
                  )}>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">
                      {message.content}
                      {message.isStreaming && (
                        <span className="inline-block w-2 h-4 ml-1 bg-accent animate-pulse" />
                      )}
                    </p>
                  </div>

                  {/* Insights */}
                  {message.insights && message.insights.length > 0 && !message.isStreaming && (
                    <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <Lightbulb className="w-4 h-4 text-amber-600" />
                        <span className="text-sm font-medium text-amber-800">Key Insights</span>
                      </div>
                      <ul className="space-y-2">
                        {message.insights.map((insight, i) => (
                          <li key={i} className="text-sm text-amber-900 flex items-start gap-2">
                            <span className="text-amber-500 mt-1">•</span>
                            <span>{insight}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Charts */}
                  {message.charts && message.charts.length > 0 && !message.isStreaming && (
                    <div className="space-y-4">
                      {message.charts.map((chart, i) => (
                        <ChartView key={i} chart={chart} />
                      ))}
                    </div>
                  )}

                  {/* Tables */}
                  {message.tables && message.tables.length > 0 && !message.isStreaming && (
                    <div className="space-y-4">
                      {message.tables.map((table, i) => (
                        <DataTable key={i} data={table} />
                      ))}
                    </div>
                  )}

                  {/* Sources */}
                  {message.sources && message.sources.length > 0 && !message.isStreaming && (
                    <div className="flex flex-wrap gap-2">
                      {message.sources.map((source, i) => (
                        <Badge key={i} variant="info" className="text-xs">
                          Source {i + 1}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-border p-4">
            <form onSubmit={handleSubmit} className="flex gap-3">
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={selectedDataset ? 'Ask a question about your data...' : 'Select a dataset first...'}
                disabled={!selectedDataset || isLoading}
                className="flex-1"
              />
              <Button
                type="submit"
                disabled={!selectedDataset || !query.trim() || isLoading}
                className="shrink-0"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </form>
          </div>
        </Card>
      </div>
    </div>
  )
}
