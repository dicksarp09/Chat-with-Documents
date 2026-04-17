const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'

interface UploadResponse {
  dataset_id: string
  filename: string
  type: 'document' | 'csv'
  message: string
}

interface QueryResponse {
  answer: string
  sources: string[]
  dataset_id: string
  response?: {
    chat?: {
      answer?: string
    }
    answer?: string
    tables?: any[]
    plots?: any[]
    insights?: Array<{ text: string }>
    sources?: string[]
  }
}

interface Dataset {
  dataset_id: string
  filename: string
  type: 'document' | 'csv'
  created_at: string
  chunk_count?: number
}

export interface ChartData {
  type: string
  title?: string
  data: Record<string, any>[]
  xKey?: string
  yKey?: string
  x?: string
  y?: string
  name?: string
}

export interface StreamingMessage {
  type: 'chunk' | 'complete' | 'error'
  content?: string
  error?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

export interface TableData {
  columns?: string[]
  headers?: string[]
  rows?: string[][]
  title?: string
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_URL}/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(error.detail || 'Upload failed')
  }

  return response.json()
}

export async function queryDataset(datasetId: string, query: string): Promise<QueryResponse> {
  const response = await fetch(`${API_URL}/query?dataset_id=${datasetId}&q=${encodeURIComponent(query)}`, {
    method: 'POST',
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Query failed' }))
    throw new Error(error.detail || 'Query failed')
  }

  return response.json()
}

export async function getDatasets(): Promise<Dataset[]> {
  const response = await fetch(`${API_URL}/datasets`)
  
  if (!response.ok) {
    throw new Error('Failed to fetch datasets')
  }

  return response.json()
}

export async function deleteDataset(datasetId: string): Promise<void> {
  const response = await fetch(`${API_URL}/datasets/${datasetId}`, {
    method: 'DELETE',
  })

  if (!response.ok) {
    throw new Error('Failed to delete dataset')
  }
}

export function getWebSocketUrl(datasetId: string): string {
  return `${WS_URL}/ws/query/${datasetId}`
}

export function useApiUrl() {
  return API_URL
}

export function createQueryWebSocket(
  datasetId: string,
  onMessage: (data: StreamingMessage) => void
): WebSocket {
  const url = `${WS_URL}/ws/query/${datasetId}`
  const ws = new WebSocket(url)

  ws.onopen = () => {
    console.log('WebSocket connected')
  }

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as StreamingMessage
      onMessage(data)
    } catch (e) {
      console.error('Failed to parse WebSocket message:', e)
    }
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
    onMessage({ type: 'error', error: 'Connection error' })
  }

  ws.onclose = () => {
    console.log('WebSocket disconnected')
  }

  return ws
}

export async function sendQueryViaWebSocket(
  ws: WebSocket,
  query: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ query }))
      resolve()
    } else {
      reject(new Error('WebSocket not connected'))
    }
  })
}