import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Document Intelligence Studio',
  description: 'Upload documents and CSV files, ask questions, and get AI-powered insights',
  icons: {
    icon: [
      {
        url: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="%23B45309" width="100" height="100" rx="20"/><text y="70" x="50" text-anchor="middle" font-size="60" fill="white" font-family="serif">DI</text></svg>',
        type: 'image/svg+xml',
      },
    ],
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        {children}
      </body>
    </html>
  )
}
