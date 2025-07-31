'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Copy, Download, Eye, CheckCircle, AlertCircle } from 'lucide-react';
import dynamic from 'next/dynamic';

// Dynamically import Monaco editor to avoid SSR issues
const MonacoEditor = dynamic(() => import('@monaco-editor/react'), {
  ssr: false,
  loading: () => <div className="h-64 bg-gray-100 animate-pulse rounded" />
});

interface ProofViewerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  proofData?: {
    id: string;
    name: string;
    content: string;
    status: 'verified' | 'pending' | 'failed';
    hash: string;
    createdAt: string;
    verifiedAt?: string;
    theorem: string;
    dependencies: string[];
  };
}

export function ProofViewer({ open, onOpenChange, proofData }: ProofViewerProps) {
  const [copied, setCopied] = useState(false);

  const copyReference = () => {
    if (proofData) {
      const reference = `${proofData.hash} ${proofData.name}`;
      navigator.clipboard.writeText(reference);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const downloadProof = () => {
    if (proofData) {
      const blob = new Blob([proofData.content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${proofData.name}.lean`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'verified':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'pending':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'verified':
        return 'bg-green-100 text-green-800';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (!proofData) {
    return null;
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle className="text-xl">{proofData.name}</DialogTitle>
              <CardDescription>
                Formal verification proof for {proofData.theorem}
              </CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <Badge className={getStatusColor(proofData.status)}>
                {getStatusIcon(proofData.status)}
                {proofData.status}
              </Badge>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-6">
          {/* Proof Metadata */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Proof Information</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">Proof Hash</p>
                  <p className="text-sm font-mono bg-gray-100 p-2 rounded">
                    {proofData.hash}
                  </p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Created</p>
                  <p className="text-sm">
                    {new Date(proofData.createdAt).toLocaleString()}
                  </p>
                </div>
                {proofData.verifiedAt && (
                  <div>
                    <p className="text-sm font-medium text-gray-500">Verified</p>
                    <p className="text-sm">
                      {new Date(proofData.verifiedAt).toLocaleString()}
                    </p>
                  </div>
                )}
                <div>
                  <p className="text-sm font-medium text-gray-500">Dependencies</p>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {proofData.dependencies.map((dep, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {dep}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Proof Content */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Proof Content</CardTitle>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={copyReference}
                  >
                    <Copy className="h-4 w-4 mr-2" />
                    {copied ? 'Copied!' : 'Copy Reference'}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={downloadProof}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="border rounded-lg overflow-hidden">
                <MonacoEditor
                  height="400px"
                  language="lean"
                  value={proofData.content}
                  options={{
                    readOnly: true,
                    minimap: { enabled: false },
                    scrollBeyondLastLine: false,
                    fontSize: 14,
                    lineNumbers: 'on',
                    wordWrap: 'on',
                    theme: 'vs-dark'
                  }}
                />
              </div>
            </CardContent>
          </Card>

          {/* Verification Details */}
          {proofData.status === 'verified' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Verification Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span className="text-sm font-medium">Proof verified successfully</span>
                  </div>
                  <div className="text-sm text-gray-600">
                    <p>• All theorems and lemmas have been formally verified</p>
                    <p>• No counterexamples found</p>
                    <p>• Memory bounds and latency constraints satisfied</p>
                    <p>• Reward ordering properties maintained</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {proofData.status === 'failed' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Verification Failed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <AlertCircle className="h-5 w-5 text-red-500" />
                    <span className="text-sm font-medium">Proof verification failed</span>
                  </div>
                  <div className="text-sm text-gray-600">
                    <p>• Counterexample found in theorem verification</p>
                    <p>• Memory bounds exceeded in some cases</p>
                    <p>• Latency constraints not satisfied</p>
                    <p>• Review proof logic and model constraints</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
} 