'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { 
  Play, 
  Search, 
  Filter, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  ArrowRight,
  Plus
} from 'lucide-react';

// Mock data for pipelines
const mockPipelines = [
  {
    id: '1',
    name: 'CartPole PPO Training',
    projectId: '1',
    status: 'completed',
    startedAt: '2024-01-20T10:30:00Z',
    completedAt: '2024-01-20T14:45:00Z',
    duration: '4h 15m',
    steps: [
      { name: 'Data Collection', status: 'completed', duration: '45m' },
      { name: 'Model Training', status: 'completed', duration: '2h 30m' },
      { name: 'Quantization', status: 'completed', duration: '30m' },
      { name: 'Verification', status: 'completed', duration: '30m' }
    ],
    reward: 98.5,
    binarySize: 2048,
    algorithm: 'PPO',
    environment: 'CartPole-v1'
  },
  {
    id: '2',
    name: 'LunarLander A2C Optimization',
    projectId: '2',
    status: 'running',
    startedAt: '2024-01-20T16:00:00Z',
    completedAt: null,
    duration: '2h 30m',
    steps: [
      { name: 'Data Collection', status: 'completed', duration: '30m' },
      { name: 'Model Training', status: 'running', duration: '1h 45m' },
      { name: 'Quantization', status: 'pending', duration: null },
      { name: 'Verification', status: 'pending', duration: null }
    ],
    reward: null,
    binarySize: null,
    algorithm: 'A2C',
    environment: 'LunarLander-v2'
  },
  {
    id: '3',
    name: 'Acrobot DQN Verification',
    projectId: '3',
    status: 'failed',
    startedAt: '2024-01-19T09:00:00Z',
    completedAt: '2024-01-19T11:30:00Z',
    duration: '2h 30m',
    steps: [
      { name: 'Data Collection', status: 'completed', duration: '30m' },
      { name: 'Model Training', status: 'completed', duration: '1h 30m' },
      { name: 'Quantization', status: 'completed', duration: '15m' },
      { name: 'Verification', status: 'failed', duration: '15m' }
    ],
    reward: 92.7,
    binarySize: 3072,
    algorithm: 'DQN',
    environment: 'Acrobot-v1'
  }
];

export default function PipelinesPage() {
  const [pipelines, setPipelines] = useState(mockPipelines);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const filteredPipelines = pipelines.filter(pipeline => {
    const matchesSearch = pipeline.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         pipeline.algorithm.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         pipeline.environment.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || pipeline.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'running':
        return <Clock className="h-4 w-4 text-blue-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  return (
    <div className="container-responsive py-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Pipelines</h1>
            <p className="text-gray-600 mt-1">
              Monitor and manage your model training pipelines
            </p>
          </div>
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            New Pipeline
          </Button>
        </div>

        {/* Search and Filters */}
        <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <Input
              placeholder="Search pipelines..."
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
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
            <Button variant="outline" size="sm">
              <Filter className="mr-2 h-4 w-4" />
              Filter
            </Button>
          </div>
        </div>
      </div>

      {/* Pipelines Grid */}
      <div className="space-y-6">
        {filteredPipelines.map((pipeline) => (
          <Card key={pipeline.id} className="hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(pipeline.status)}
                  <div>
                    <CardTitle className="text-lg">{pipeline.name}</CardTitle>
                    <CardDescription>
                      {pipeline.algorithm} • {pipeline.environment}
                    </CardDescription>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge className={getStatusColor(pipeline.status)}>
                    {pipeline.status}
                  </Badge>
                  <Button variant="outline" size="sm">
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">Duration</p>
                  <p className="text-lg font-semibold">{pipeline.duration}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Started</p>
                  <p className="text-sm">{new Date(pipeline.startedAt).toLocaleDateString()}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Reward</p>
                  <p className="text-lg font-semibold">
                    {pipeline.reward ? `${pipeline.reward}%` : 'N/A'}
                  </p>
                </div>
              </div>

              {/* Pipeline Steps */}
              <div className="mt-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Pipeline Steps</p>
                <div className="space-y-2">
                  {pipeline.steps.map((step, index) => (
                    <div key={index} className="flex items-center justify-between text-sm">
                      <div className="flex items-center space-x-2">
                        {step.status === 'completed' && <CheckCircle className="h-4 w-4 text-green-500" />}
                        {step.status === 'running' && <Clock className="h-4 w-4 text-blue-500" />}
                        {step.status === 'failed' && <XCircle className="h-4 w-4 text-red-500" />}
                        {step.status === 'pending' && <AlertCircle className="h-4 w-4 text-gray-400" />}
                        <span className={step.status === 'completed' ? 'text-gray-900' : 'text-gray-500'}>
                          {step.name}
                        </span>
                      </div>
                      {step.duration && (
                        <span className="text-gray-500">{step.duration}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Results */}
              {pipeline.status === 'completed' && (
                <div className="mt-4 pt-4 border-t">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium text-gray-500">Binary Size</p>
                      <p className="text-sm font-semibold">{pipeline.binarySize} bytes</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-500">Final Reward</p>
                      <p className="text-sm font-semibold">{pipeline.reward}%</p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredPipelines.length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-400 mb-4">
            <Play className="mx-auto h-12 w-12" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No pipelines found</h3>
          <p className="text-gray-600 mb-4">
            {searchTerm ? 'Try adjusting your search terms.' : 'Get started by creating your first pipeline.'}
          </p>
          {!searchTerm && (
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Create Pipeline
            </Button>
          )}
        </div>
      )}
    </div>
  );
} 