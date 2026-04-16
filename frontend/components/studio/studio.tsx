'use client'

import { useState, useEffect, useCallback } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { UploadZone } from './upload-zone'
import { DatasetList } from './dataset-list'
import { ChatContainer } from './chat-container'
import { Upload, MessageSquare, Database } from 'lucide-react'

export default function Studio() {
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [datasets, setDatasets] = useState<Array<{
    dataset_id: string
    type: 'csv' | 'document'
    filename: string
    status: string
  }>>([])
  const [refreshKey, setRefreshKey] = useState(0)

  const loadDatasets = useCallback(async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'}/datasets`)
      const data = await response.json()
      setDatasets(data.datasets || [])
    } catch (error) {
      console.error('Failed to load datasets:', error)
    }
  }, [])

  useEffect(() => {
    loadDatasets()
  }, [loadDatasets, refreshKey])

  const handleUploadComplete = () => {
    setRefreshKey(k => k + 1)
  }

  const handleSelectDataset = (datasetId: string) => {
    setSelectedDataset(datasetId)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-accent flex items-center justify-center">
                <span className="text-white font-serif font-bold text-lg">DI</span>
              </div>
              <div>
                <h1 className="font-serif text-2xl font-semibold tracking-tight text-foreground">
                  Document Intelligence Studio
                </h1>
                <p className="text-sm text-foreground-muted">
                  Upload files and chat with your data
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs text-foreground-muted font-mono">
                {datasets.length} {datasets.length === 1 ? 'dataset' : 'datasets'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <Tabs defaultValue="chat" className="space-y-6">
          <TabsList className="bg-linen border border-border">
            <TabsTrigger value="upload">
              <Upload className="w-4 h-4" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="datasets">
              <Database className="w-4 h-4" />
              Datasets
            </TabsTrigger>
            <TabsTrigger value="chat">
              <MessageSquare className="w-4 h-4" />
              Chat
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <UploadZone onUploadComplete={handleUploadComplete} />
          </TabsContent>

          <TabsContent value="datasets">
            <DatasetList
              datasets={datasets}
              onSelect={handleSelectDataset}
              selectedId={selectedDataset}
              onRefresh={() => setRefreshKey(k => k + 1)}
            />
          </TabsContent>

          <TabsContent value="chat">
            <ChatContainer
              datasets={datasets}
              selectedDataset={selectedDataset}
              onSelectDataset={handleSelectDataset}
            />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-linen mt-auto">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <p className="text-center text-sm text-foreground-muted font-serif italic">
            Powered by Groq LLaMA · Built with FastAPI & Next.js
          </p>
        </div>
      </footer>
    </div>
  )
}
