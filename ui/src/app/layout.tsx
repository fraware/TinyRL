import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Sidebar } from '@/components/layout/sidebar';
import { TopNav } from '@/components/layout/top-nav';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'TinyRL - Reinforcement Learning for Microcontrollers',
  description: 'Train, optimize, and deploy reinforcement learning models on resource-constrained devices with enterprise-grade reliability.',
  keywords: ['reinforcement learning', 'microcontrollers', 'embedded systems', 'AI', 'machine learning'],
  authors: [{ name: 'TinyRL Team' }],
  creator: 'TinyRL',
  publisher: 'TinyRL',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://tinyrl.dev'),
  openGraph: {
    title: 'TinyRL - Reinforcement Learning for Microcontrollers',
    description: 'Train, optimize, and deploy reinforcement learning models on resource-constrained devices.',
    url: 'https://tinyrl.dev',
    siteName: 'TinyRL',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'TinyRL - Reinforcement Learning for Microcontrollers',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'TinyRL - Reinforcement Learning for Microcontrollers',
    description: 'Train, optimize, and deploy reinforcement learning models on resource-constrained devices.',
    images: ['/og-image.png'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'your-google-verification-code',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full bg-gray-50`}>
        <div className="flex h-full">
          <Sidebar />
          <div className="flex flex-1 flex-col lg:pl-64">
            <TopNav 
              user={{
                name: 'John Doe',
                email: 'john@example.com',
                avatar: '/avatars/john.jpg',
              }}
              notifications={3}
            />
            <main className="flex-1 overflow-y-auto">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
} 