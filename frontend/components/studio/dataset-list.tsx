'use client'

import { useState } from 'react'
import { FileText, Table2, Trash2, MessageSquare, RefreshCw, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { deleteDataset } from '@/lib/api'
import { cn } from '@/lib/utils'

interface Dataset {
  dataset_id: string
  type: 'csv' | 'document'
  filename: string
  status: string
}

interface DatasetListProps {
  datasets: Dataset[]
  selectedId: string | null
  onSelect: (id: string) => void
  onRefresh: () => void
}

export function DatasetList({ datasets, selectedId, onSelect, onRefresh }: DatasetListProps) {
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const handleDelete = async (e: React.MouseEvent, datasetId: string) => {
    e.stopPropagation()
    setDeletingId(datasetId)
    
    try {
      await deleteDataset(datasetId)
      onRefresh()
      if (selectedId === datasetId) {
        onSelect('')
      }
    } catch (error) {
      console.error('Failed to delete:', error)
    } finally {
      setDeletingId(null)
    }
  }

  if (datasets.length === 0) {
    return (
      <Card variant="paper">
        <CardContent className="py-16 text-center">
          <div className="w-16 h-16 mx-auto rounded-2xl bg-background-secondary flex items-center justify-center mb-4">
            <FileText className="w-8 h-8 text-foreground-muted" />
          </div>
          <h3 className="font-serif text-xl font-medium text-foreground mb-2">
            No datasets yet
          </h3>
          <p className="text-foreground-muted max-w-sm mx-auto">
            Upload a CSV file or document to get started with analysis.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-foreground-muted">
          {datasets.length} {datasets.length === 1 ? 'dataset' : 'datasets'}
        </p>
        <Button variant="ghost" size="sm" onClick={onRefresh}>
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {datasets.map((dataset, index) => (
          <Card
            key={dataset.dataset_id}
            className={cn(
              'cursor-pointer transition-all duration-200 hover:shadow-lifted',
              selectedId === dataset.dataset_id
                ? 'ring-2 ring-accent border-accent'
                : ''
            )}
            onClick={() => onSelect(dataset.dataset_id)}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <CardContent className="p-5">
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-center gap-3 min-w-0">
                  <div className={cn(
                    'w-10 h-10 rounded-lg flex items-center justify-center shrink-0',
                    dataset.type === 'csv' ? 'bg-emerald-100' : 'bg-amber-100'
                  )}>
                    {dataset.type === 'csv' ? (
                      <Table2 className={cn('w-5 h-5', dataset.type === 'csv' ? 'text-emerald-600' : 'text-amber-600')} />
                    ) : (
                      <FileText className="w-5 h-5 text-amber-600" />
                    )}
                  </div>
                  <div className="min-w-0">
                    <p className="font-medium text-foreground truncate">
                      {dataset.filename}
                    </p>
                    <p className="text-xs text-foreground-muted font-mono">
                      {dataset.dataset_id}
                    </p>
                  </div>
                </div>
                <Badge variant={dataset.type === 'csv' ? 'csv' : 'document'}>
                  {dataset.type === 'csv' ? 'CSV' : 'Document'}
                </Badge>
              </div>

              <div className="mt-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={cn(
                    'w-2 h-2 rounded-full',
                    dataset.status === 'profiled' || dataset.status === 'processed'
                      ? 'bg-emerald-500'
                      : 'bg-amber-500'
                  )} />
                  <span className="text-xs text-foreground-muted capitalize">
                    {dataset.status}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-foreground-muted hover:text-accent"
                    onClick={(e) => handleDelete(e, dataset.dataset_id)}
                    disabled={deletingId === dataset.dataset_id}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
