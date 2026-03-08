import './globals.css'

export const metadata = {
  title: 'AI-NIDS Dashboard',
  description: 'AI Network Intrusion Detection System',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
