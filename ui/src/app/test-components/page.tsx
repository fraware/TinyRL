'use client';

import React, { useState } from 'react';
import { ModelSnapshotSelector, ModelSnapshotCard } from '@/components/projects/model-snapshot-selector';
import { RewardEpisodeChart } from '@/components/charts/reward-episode-chart';
import { RadialGauge } from '@/components/charts/radial-gauge';
import { VerificationCard, VerificationSummary } from '@/components/projects/verification-card';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

// Mock data for testing
const mockSnapshots = [
  {
    id: '1',
    name: 'CartPole PPO v1.2',
    tag: 'v1.2.0',
    commit: 'a1b2c3d4e5f6',
    createdAt: '2024-01-20T10:30:00Z',
    reward: 98.5,
    binarySize: 2048,
    status: 'verified' as const,
    description: 'Latest stable version with improved performance'
  },
  {
    id: '2',
    name: 'CartPole PPO v1.1',
    tag: 'v1.1.0',
    commit: 'b2c3d4e5f6g7',
    createdAt: '2024-01-15T14:20:00Z',
    reward: 95.2,
    binarySize: 2156,
    status: 'verified' as const,
    description: 'Previous stable version'
  },
  {
    id: '3',
    name: 'CartPole PPO v1.0',
    tag: 'v1.0.0',
    commit: 'c3d4e5f6g7h8',
    createdAt: '2024-01-10T09:15:00Z',
    reward: 92.1,
    binarySize: 2304,
    status: 'verified' as const,
    description: 'Initial release'
  }
];

const mockEpisodeData = [
  { episode: 1, reward: 45.2, latency: 12.3, timestamp: '2024-01-20T10:00:00Z' },
  { episode: 2, reward: 52.8, latency: 11.8, timestamp: '2024-01-20T10:05:00Z' },
  { episode: 3, reward: 58.4, latency: 11.2, timestamp: '2024-01-20T10:10:00Z' },
  { episode: 4, reward: 64.7, latency: 10.9, timestamp: '2024-01-20T10:15:00Z' },
  { episode: 5, reward: 71.3, latency: 10.5, timestamp: '2024-01-20T10:20:00Z' },
  { episode: 6, reward: 78.9, latency: 10.1, timestamp: '2024-01-20T10:25:00Z' },
  { episode: 7, reward: 85.2, latency: 9.8, timestamp: '2024-01-20T10:30:00Z' },
  { episode: 8, reward: 91.7, latency: 9.5, timestamp: '2024-01-20T10:35:00Z' },
  { episode: 9, reward: 95.4, latency: 9.2, timestamp: '2024-01-20T10:40:00Z' },
  { episode: 10, reward: 98.5, latency: 9.0, timestamp: '2024-01-20T10:45:00Z' }
];

const mockVerification = {
  id: '1',
  status: 'verified' as const,
  createdAt: '2024-01-20T10:00:00Z',
  completedAt: '2024-01-20T10:05:00Z',
  duration: 300,
  properties: [
    {
      id: '1',
      name: 'Safety Constraint',
      description: 'Model never outputs unsafe actions',
      status: 'verified' as const,
      proof: 'Mathematical proof of safety...'
    },
    {
      id: '2',
      name: 'Performance Guarantee',
      description: 'Model maintains minimum reward threshold',
      status: 'verified' as const,
      proof: 'Performance analysis proof...'
    },
    {
      id: '3',
      name: 'Memory Bounds',
      description: 'Model respects memory constraints',
      status: 'verified' as const,
      proof: 'Memory usage analysis...'
    }
  ],
  proof: 'Complete formal verification proof...'
};

export default function TestComponentsPage() {
  const [selectedSnapshot, setSelectedSnapshot] = useState(mockSnapshots[0]);

  return (
    <div className="container-responsive py-6 space-y-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Component Test Page</h1>
        <p className="text-gray-600 mt-1">
          Testing all new UI components for TinyRL
        </p>
      </div>

      {/* Model Snapshot Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Model Snapshot Selector</CardTitle>
          <CardDescription>
            Test the model snapshot selector with Git tag search
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <ModelSnapshotSelector
            snapshots={mockSnapshots}
            selectedSnapshot={selectedSnapshot}
            onSnapshotSelect={setSelectedSnapshot}
            placeholder="Select a model snapshot..."
          />
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {mockSnapshots.map((snapshot) => (
              <ModelSnapshotCard
                key={snapshot.id}
                snapshot={snapshot}
                selected={selectedSnapshot?.id === snapshot.id}
                onSelect={setSelectedSnapshot}
              />
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Radial Gauge */}
      <Card>
        <CardHeader>
          <CardTitle>Radial Gauge Components</CardTitle>
          <CardDescription>
            Test radial gauges for memory usage and performance metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <RadialGauge
              value={2048}
              max={32768}
              label="Memory Usage"
              unit="KB"
              color="#3B82F6"
            />
            <RadialGauge
              value={98.5}
              max={100}
              label="Reward"
              unit="%"
              color="#10B981"
            />
            <RadialGauge
              value={9.0}
              max={50}
              label="Latency"
              unit="ms"
              color="#F59E0B"
            />
            <RadialGauge
              value={75}
              max={100}
              label="CPU Usage"
              unit="%"
              color="#EF4444"
            />
          </div>
        </CardContent>
      </Card>

      {/* Reward Episode Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Reward Episode Chart</CardTitle>
          <CardDescription>
            Test the reward progress chart with latency metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RewardEpisodeChart
            data={mockEpisodeData}
            title="Training Progress"
            description="Reward and latency over episodes"
            showLatency={true}
            showTrend={true}
            height={300}
          />
        </CardContent>
      </Card>

      {/* Verification Card */}
      <Card>
        <CardHeader>
          <CardTitle>Verification Components</CardTitle>
          <CardDescription>
            Test formal verification status and results
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <VerificationCard
            verification={mockVerification}
            onViewProof={(proof) => {
              console.log('View proof:', proof);
              alert('Proof viewer would open here');
            }}
            onDownloadProof={(proof) => {
              console.log('Download proof:', proof);
              alert('Proof download would start here');
            }}
          />
          
          <VerificationSummary
            verifications={[mockVerification]}
          />
        </CardContent>
      </Card>

      {/* Status Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Component Status</CardTitle>
          <CardDescription>
            Overview of all component tests
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">✅</div>
              <div className="text-sm text-green-800">Model Snapshot Selector</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">✅</div>
              <div className="text-sm text-green-800">Radial Gauge</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">✅</div>
              <div className="text-sm text-green-800">Reward Chart</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">✅</div>
              <div className="text-sm text-green-800">Verification Card</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 