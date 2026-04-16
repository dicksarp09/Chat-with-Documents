'use client'

import { useState, useCallback } from 'react'
import { Upload, FileText, Table2, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import { cn, getFileType } from '@/lib/utils'
import { uploadFile } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface UploadZoneProps {
  onUploadComplete: () => void
}

interface UploadingFile {
  file: File
  progress: number
  status: 'uploading' | 'success' | 'error'
  error?: string
  result?: {
    dataset_id: string
    filename: string
    type: string
  }
}

export function UploadZone({ onUploadComplete }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const uploadFiles = useCallback(async (files: FileList) => {
    const newFiles: UploadingFile[] = Array.from(files).map(file => ({
      file,
      progress: 0,
      status: 'uploading' as const,
    }))

    setUploadingFiles(prev => [...prev, ...newFiles])

    for (let i = 0; i < newFiles.length; i++) {
      const fileItem = newFiles[i]
      
      try {
        // Simulate progress
        const progressInterval = setInterval(() => {
          setUploadingFiles(prev => 
            prev.map(f => 
              f.file === fileItem.file && f.status === 'uploading'
                ? { ...f, progress: Math.min(f.progress + 10, 90) }
                : f
            )
          )
        }, 100)

        const result = await uploadFile(fileItem.file)
        
        clearInterval(progressInterval)

        setUploadingFiles(prev =>
          prev.map(f =>
            f.file === fileItem.file
              ? { ...f, progress: 100, status: 'success', result }
              : f
          )
        )

        setTimeout(() => {
          onUploadComplete()
        }, 500)
      } catch (error) {
        setUploadingFiles(prev =>
          prev.map(f =>
            f.file === fileItem.file
              ? { ...f, status: 'error', error: error instanceof Error ? error.message : 'Upload failed' }
              : f
          )
        )
      }
    }
  }, [onUploadComplete])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      uploadFiles(files)
    }
  }, [uploadFiles])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      uploadFiles(files)
    }
    e.target.value = ''
  }, [uploadFiles])

  const removeFile = useCallback((file: File) => {
    setUploadingFiles(prev => prev.filter(f => f.file !== file))
  }, [])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Upload Area */}
      <Card variant="paper">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
              <Upload className="w-5 h-5 text-accent" />
            </div>
            Upload Files
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={cn(
              'relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200',
              isDragging
                ? 'border-accent bg-accent/5 scale-[1.02]'
                : 'border-border hover:border-border-strong hover:bg-background-secondary'
            )}
          >
            <input
              type="file"
              accept=".csv,.pdf,.docx,.doc"
              multiple
              onChange={handleFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            
            <div className="space-y-4">
              <div className="w-16 h-16 mx-auto rounded-2xl bg-linen flex items-center justify-center">
                <FileText className="w-8 h-8 text-foreground-muted" />
              </div>
              
              <div>
                <p className="font-serif text-lg font-medium text-foreground">
                  Drop files here or click to upload
                </p>
                <p className="text-sm text-foreground-muted mt-1">
                  Supports CSV, PDF, and DOCX files
                </p>
              </div>

              <div className="flex items-center justify-center gap-6 pt-4">
                <div className="flex items-center gap-2 text-sm text-foreground-muted">
                  <Table2 className="w-4 h-4" />
                  CSV
                </div>
                <div className="flex items-center gap-2 text-sm text-foreground-muted">
                  <FileText className="w-4 h-4" />
                  PDF
                </div>
                <div className="flex items-center gap-2 text-sm text-foreground-muted">
                  <FileText className="w-4 h-4" />
                  DOCX
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Upload Progress */}
      <Card variant="paper">
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
              <Loader2 className="w-5 h-5 text-accent animate-spin" />
            </div>
            Upload Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          {uploadingFiles.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-foreground-muted font-serif italic">
                No files uploading
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {uploadingFiles.map((item, index) => (
                <div
                  key={`${item.file.name}-${index}`}
                  className="bg-background-secondary rounded-lg p-4"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-center gap-3 min-w-0">
                      {item.status === 'uploading' && (
                        <Loader2 className="w-5 h-5 text-accent animate-spin shrink-0" />
                      )}
                      {item.status === 'success' && (
                        <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
                      )}
                      {item.status === 'error' && (
                        <AlertCircle className="w-5 h-5 text-red-600 shrink-0" />
                      )}
                      <div className="min-w-0">
                        <p className="font-medium text-foreground truncate">
                          {item.file.name}
                        </p>
                        <p className="text-sm text-foreground-muted">
                          {(item.file.size / 1024).toFixed(1)} KB
                          {item.status === 'error' && ` · ${item.error}`}
                        </p>
                      </div>
                    </div>
                    <Badge variant={getFileType(item.file.name) === 'csv' ? 'csv' : 'document'}>
                      {getFileType(item.file.name) === 'csv' ? 'CSV' : 'Document'}
                    </Badge>
                  </div>
                  
                  {item.status === 'uploading' && (
                    <div className="mt-3">
                      <div className="h-1.5 bg-border rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent transition-all duration-300"
                          style={{ width: `${item.progress}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
