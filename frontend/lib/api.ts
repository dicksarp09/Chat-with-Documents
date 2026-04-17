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
}

interface Dataset {
  dataset_id: string
  filename: string
  type: 'document' | 'csv'
  created_at: string
  chunk_count?: number
}

export interface ChartData {
  type: 'bar' | 'line' | 'pie' | 'scatter'
  title: string
  data: Record<string, any>[]
  xKey?: string
  yKey?: string
}

export interface TableData {
  headers: string[]
  rows: string[][]
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