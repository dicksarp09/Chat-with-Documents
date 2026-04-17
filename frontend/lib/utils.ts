import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatBytes(bytes: number, decimals = 2) {
  if (bytes === 0) return '0 Bytes'

  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB']

  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

export function formatTime(ms: number) {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export function truncate(str: string, length: number) {
  if (str.length <= length) return str
  return str.slice(0, length) + '...'
}

export function getFileType(filename: string): 'csv' | 'document' | 'unknown' {
  const ext = filename.toLowerCase().split('.').pop()
  if (ext === 'csv') return 'csv'
  if (['pdf', 'docx', 'doc'].includes(ext || '')) return 'document'
  return 'unknown'
}

export function getFileIcon(filename: string) {
  const type = getFileType(filename)
  if (type === 'csv') return 'table'
  if (type === 'document') return 'file-text'
  return 'file'
}
