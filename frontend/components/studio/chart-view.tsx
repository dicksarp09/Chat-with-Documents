'use client'

import { ChartData } from '@/lib/api'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { cn } from '@/lib/utils'

interface ChartViewProps {
  charts: ChartData[]
  className?: string
}

const COLORS = [
  '#D97706', // amber-600
  '#F59E0B', // amber-500
  '#FBBF24', // amber-400
  '#92400E', // amber-800
  '#B45309', // amber-700
  '#FCD34D', // amber-300
]

const CHART_COLORS = [
  'hsl(38, 92%, 50%)',
  'hsl(32, 95%, 44%)',
  'hsl(45, 93%, 47%)',
  'hsl(25, 95%, 53%)',
  'hsl(40, 89%, 40%)',
  'hsl(51, 98%, 50%)',
]

export function ChartView({ charts, className }: ChartViewProps) {
  if (!charts || charts.length === 0) return null

  return (
    <div className={cn('space-y-6', className)}>
      {charts.map((chart, index) => (
        <div
          key={index}
          className="rounded-lg border border-amber-200 bg-paper p-4 shadow-sm"
        >
          {chart.title && (
            <h4 className="mb-4 text-center font-serif text-lg font-medium text-amber-900">
              {chart.title}
            </h4>
          )}
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <>
                {chart.type === 'bar' ? (
                <BarChart data={chart.data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#92400E', fontSize: 12 }}
                    tickLine={{ stroke: '#D97706' }}
                  />
                  <YAxis tick={{ fill: '#92400E', fontSize: 12 }} tickLine={{ stroke: '#D97706' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFBEB',
                      border: '1px solid #FCD34D',
                      borderRadius: '8px',
                      fontFamily: 'inherit',
                    }}
                  />
                  <Bar dataKey="value" fill="#D97706" radius={[4, 4, 0, 0]} />
                </BarChart>
              ) : chart.type === 'line' ? (
                <LineChart data={chart.data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#92400E', fontSize: 12 }}
                    tickLine={{ stroke: '#D97706' }}
                  />
                  <YAxis tick={{ fill: '#92400E', fontSize: 12 }} tickLine={{ stroke: '#D97706' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFBEB',
                      border: '1px solid #FCD34D',
                      borderRadius: '8px',
                      fontFamily: 'inherit',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#D97706"
                    strokeWidth={2}
                    dot={{ fill: '#F59E0B', strokeWidth: 2 }}
                  />
                </LineChart>
              ) : chart.type === 'pie' ? (
                <PieChart>
                  <Pie
                    data={chart.data}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {chart.data.map((_, i) => (
                      <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFBEB',
                      border: '1px solid #FCD34D',
                      borderRadius: '8px',
                      fontFamily: 'inherit',
                    }}
                  />
                </PieChart>
              ) : chart.type === 'scatter' ? (
                <ScatterChart margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: '#92400E', fontSize: 12 }}
                    name={chart.x}
                  />
                  <YAxis dataKey="value" tick={{ fill: '#92400E', fontSize: 12 }} name={chart.y} />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    contentStyle={{
                      backgroundColor: '#FFFBEB',
                      border: '1px solid #FCD34D',
                      borderRadius: '8px',
                      fontFamily: 'inherit',
                    }}
                  />
<Scatter data={chart.data} fill="#D97706" />
                </ScatterChart>
              ) : (
                <></>
              )}
            </>
          </ResponsiveContainer>
        </div>
        </div>
      ))}
    </div>
  )
}
