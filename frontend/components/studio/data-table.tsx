'use client'

import { TableData } from '@/lib/api'
import { cn } from '@/lib/utils'

interface DataTableProps {
  data: TableData
  maxRows?: number
  className?: string
}

export function DataTable({ data, maxRows = 10, className }: DataTableProps) {
  const columns = data.columns || data.headers || []
  const rows = data.rows || []
  const displayedRows = rows.slice(0, maxRows)
  const hasMore = rows.length > maxRows

  return (
    <div className={cn('overflow-hidden rounded-lg border border-amber-200 bg-paper', className)}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-amber-200 bg-amber-50/50">
              {columns.map((col, i) => (
                <th
                  key={i}
                  className="px-4 py-3 text-left font-serif font-semibold text-amber-900"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayedRows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className="border-b border-amber-100 transition-colors hover:bg-amber-50/30"
              >
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="px-4 py-2 text-amber-800 font-mono text-xs"
                  >
                    {typeof cell === 'number' ? cell.toLocaleString() : cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hasMore && (
        <div className="px-4 py-2 text-center text-xs text-amber-600 bg-amber-50/30 border-t border-amber-200">
          Showing {maxRows} of {rows.length} rows
        </div>
      )}
    </div>
  )
}
