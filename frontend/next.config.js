/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  // Vercel-compatible env var handling
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || process.env.API_URL || 'http://localhost:8000/api/v1',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || process.env.WS_URL || 'ws://localhost:8000',
  },
  // Ensure proper asset handling
  trailingSlash: false,
}

module.exports = nextConfig
