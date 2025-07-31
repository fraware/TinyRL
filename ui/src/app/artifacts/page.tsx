'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { 
  Download, 
  Search, 
  Filter, 
  FileBinary, 
  Shield, 
  Copy,
  Eye,
  CheckCircle,
  AlertCircle,
  Clock
} from 'lucide-react';

// Mock data for artifacts
const mockArtifacts = [
  {
    id: '1',
    name: 'cartpole-ppo-v1.0.0.bin',
    projectId: '1',
    projectName: 'CartPole PPO Agent',
    version: '1.0.0',
    size: 2048,
    sha256: 'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456',
    status: 'verified',
    createdAt: '2024-01-20T14:45:00Z',
    algorithm: 'PPO',
    environment: 'CartPole-v1',
    reward: 98.5,
    binaryType: 'cortex-m55',
    flashSize: 2048,
    ramSize: 512,
    latency: 5.2,
    powerConsumption: 12.5,
    proofHash: 'proof_abc123def456',
    sbomHash: 'sbom_xyz789uvw012'
  },
  {
    id: '2',
    name: 'lunarlander-a2c-v0.9.1.bin',
    projectId: '2',
    projectName: 'LunarLander A2C',
    version: '0.9.1',
    size: 4096,
    sha256: 'b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567890',
    status: 'pending',
    createdAt: '2024-01-18T16:20:00Z',
    algorithm: 'A2C',
    environment: 'LunarLander-v2',
    reward: 85.3,
    binaryType: 'cortex-m55',
    flashSize: 4096,
    ramSize: 1024,
    latency: 8.7,
    powerConsumption: 18.2,
    proofHash: null,
    sbomHash: null
  },
  {
    id: '3',
    name: 'acrobot-dqn-v1.2.0.bin',
    projectId: '3',
    projectName: 'Acrobot DQN',
    version: '1.2.0',
    size: 3072,
    sha256: 'c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567890ab',
    status: 'verified',
    createdAt: '2024-01-15T13:30:00Z',
    algorithm: 'DQN',
    environment: 'Acrobot-v1',
    reward: 92.7,
    binaryType: 'cortex-m55',
    flashSize: 3072,
    ramSize: 768,
    latency: 6.1,
    powerConsumption: 15.8,
    proofHash: 'proof_def456ghi789',
    sbomHash: 'sbom_abc123def456'
  }
];

export default function ArtifactsPage() {
  const [artifacts, setArtifacts] = useState(mockArtifacts);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const filteredArtifacts = artifacts.filter(artifact => {
    const matchesSearch = artifact.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         artifact.projectName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         artifact.algorithm.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || artifact.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'verified':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />;
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

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // You could add a toast notification here
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const truncateHash = (hash: string, length: number = 8) => {
    return hash.substring(0, length) + '...' + hash.substring(hash.length - length);
  };

  return (
    <div className="container-responsive py-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Artifacts</h1>
            <p className="text-gray-600 mt-1">
              Manage and download your compiled model binaries
            </p>
          </div>
          <Button>
            <Download className="mr-2 h-4 w-4" />
            Download All
          </Button>
        </div>

        {/* Search and Filters */}
        <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <Input
              placeholder="Search artifacts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <div className="flex items-center gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Status</option>
              <option value="verified">Verified</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
            </select>
            <Button variant="outline" size="sm">
              <Filter className="mr-2 h-4 w-4" />
              Filter
            </Button>
          </div>
        </div>
      </div>

      {/* Artifacts Grid */}
      <div className="space-y-6">
        {filteredArtifacts.map((artifact) => (
          <Card key={artifact.id} className="hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <FileBinary className="h-6 w-6 text-blue-500" />
                  <div>
                    <CardTitle className="text-lg">{artifact.name}</CardTitle>
                    <CardDescription>
                      {artifact.projectName} • {artifact.algorithm} • {artifact.environment}
                    </CardDescription>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge className={getStatusColor(artifact.status)}>
                    {artifact.status}
                  </Badge>
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">Size</p>
                  <p className="text-lg font-semibold">{formatBytes(artifact.size)}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Reward</p>
                  <p className="text-lg font-semibold">{artifact.reward}%</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Latency</p>
                  <p className="text-lg font-semibold">{artifact.latency}ms</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Power</p>
                  <p className="text-lg font-semibold">{artifact.powerConsumption}mW</p>
                </div>
              </div>

              {/* SHA256 Hash */}
              <div className="mb-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-gray-500">SHA256 Hash</p>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(artifact.sha256)}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
                <p className="text-sm font-mono bg-gray-100 p-2 rounded">
                  {truncateHash(artifact.sha256, 12)}
                </p>
              </div>

              {/* Hardware Requirements */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">Target MCU</p>
                  <p className="text-sm font-semibold">{artifact.binaryType}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Flash Size</p>
                  <p className="text-sm font-semibold">{formatBytes(artifact.flashSize)}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">RAM Size</p>
                  <p className="text-sm font-semibold">{formatBytes(artifact.ramSize)}</p>
                </div>
              </div>

              {/* Verification and SBOM */}
              <div className="flex items-center space-x-4">
                {artifact.proofHash && (
                  <div className="flex items-center space-x-2">
                    <Shield className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-gray-600">Formal Proof</span>
                    <Button variant="ghost" size="sm">
                      <Eye className="h-4 w-4" />
                    </Button>
                  </div>
                )}
                {artifact.sbomHash && (
                  <div className="flex items-center space-x-2">
                    <FileBinary className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-600">SBOM</span>
                    <Button variant="ghost" size="sm">
                      <Eye className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>

              {/* Created Date */}
              <div className="mt-4 pt-4 border-t">
                <p className="text-sm text-gray-500">
                  Created: {new Date(artifact.createdAt).toLocaleDateString()} at{' '}
                  {new Date(artifact.createdAt).toLocaleTimeString()}
                </p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredArtifacts.length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-400 mb-4">
            <FileBinary className="mx-auto h-12 w-12" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No artifacts found</h3>
          <p className="text-gray-600 mb-4">
            {searchTerm ? 'Try adjusting your search terms.' : 'Artifacts will appear here after successful pipeline runs.'}
          </p>
        </div>
      )}
    </div>
  );
} 