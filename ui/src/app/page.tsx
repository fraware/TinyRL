'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { SparklineChart } from '@/components/charts/sparkline-chart';
import { 
  TrendingUp, 
  Cpu, 
  Zap, 
  Shield, 
  Activity, 
  Clock,
  ArrowRight,
  Play,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

// Mock data for dashboard
const dashboardStats = {
  totalProjects: 24,
  activePipelines: 3,
  totalReward: 2847.5,
  avgLatency: 12.3,
  verifiedModels: 18,
  fleetDevices: 156
};

const recentActivity = [
  {
    id: '1',
    type: 'pipeline_completed',
    title: 'CartPole PPO Training',
    description: 'Pipeline completed successfully',
    timestamp: '2 minutes ago',
    status: 'success',
    reward: 98.5
  },
  {
    id: '2',
    type: 'model_deployed',
    title: 'LunarLander A2C',
    description: 'Model deployed to fleet',
    timestamp: '15 minutes ago',
    status: 'success',
    devices: 12
  },
  {
    id: '3',
    type: 'verification_failed',
    title: 'Acrobot DQN',
    description: 'Formal verification failed',
    timestamp: '1 hour ago',
    status: 'error'
  }
];

const rewardData = {
  data: [85, 87, 89, 92, 90, 94, 96, 98, 95, 97, 99, 98.5],
  labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
};

export default function DashboardPage() {
  return (
    <div className="container-responsive py-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">
          Overview of your TinyRL projects and fleet performance
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Projects</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardStats.totalProjects}</div>
            <p className="text-xs text-muted-foreground">
              +12% from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Pipelines</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardStats.activePipelines}</div>
            <p className="text-xs text-muted-foreground">
              Currently running
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Reward</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardStats.totalReward}</div>
            <p className="text-xs text-muted-foreground">
              Across all models
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardStats.avgLatency}ms</div>
            <p className="text-xs text-muted-foreground">
              Fleet average
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Verified Models</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardStats.verifiedModels}</div>
            <p className="text-xs text-muted-foreground">
              Formally verified
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Fleet Devices</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dashboardStats.fleetDevices}</div>
            <p className="text-xs text-muted-foreground">
              Active devices
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Reward Performance Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Reward Performance</CardTitle>
            <CardDescription>
              Monthly reward progression across all projects
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-center justify-center">
              <SparklineChart 
                data={rewardData} 
                width={400} 
                height={200} 
                showArea={true}
              />
            </div>
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>
              Latest pipeline and deployment updates
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentActivity.map((activity) => (
                <div key={activity.id} className="flex items-start space-x-3">
                  <div className="flex-shrink-0">
                    {activity.status === 'success' ? (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-red-500" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">
                      {activity.title}
                    </p>
                    <p className="text-sm text-gray-500">
                      {activity.description}
                    </p>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className="text-xs text-gray-400">
                        {activity.timestamp}
                      </span>
                      {activity.reward && (
                        <Badge variant="secondary" className="text-xs">
                          {activity.reward} reward
                        </Badge>
                      )}
                      {activity.devices && (
                        <Badge variant="secondary" className="text-xs">
                          {activity.devices} devices
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4">
              <Button variant="outline" size="sm" className="w-full">
                View All Activity
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <div className="mt-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Button className="h-20 flex flex-col items-center justify-center space-y-2">
            <Play className="h-6 w-6" />
            <span>Start New Pipeline</span>
          </Button>
          <Button variant="outline" className="h-20 flex flex-col items-center justify-center space-y-2">
            <Cpu className="h-6 w-6" />
            <span>Deploy to Fleet</span>
          </Button>
          <Button variant="outline" className="h-20 flex flex-col items-center justify-center space-y-2">
            <Shield className="h-6 w-6" />
            <span>Verify Model</span>
          </Button>
        </div>
      </div>
    </div>
  );
} 