'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { SparklineChart } from '@/components/charts/sparkline-chart';
import { 
  Cpu, 
  Search, 
  Filter, 
  Wifi, 
  WifiOff,
  CheckCircle,
  AlertCircle,
  Clock,
  XCircle,
  Upload,
  Download,
  Activity,
  Battery,
  Signal
} from 'lucide-react';

// Mock data for fleet devices
const mockDevices = [
  {
    id: '1',
    name: 'STM32-Nucleo-144',
    deviceId: 'nucleo-144-001',
    status: 'online',
    model: 'CartPole PPO v1.0.0',
    reward: 98.5,
    latency: 5.2,
    powerConsumption: 12.5,
    uptime: '7d 12h 34m',
    lastSeen: '2024-01-20T15:30:00Z',
    location: 'Lab A',
    firmware: 'v1.2.3',
    signalStrength: 85,
    batteryLevel: 92,
    rewardHistory: {
      data: [95, 96, 97, 98, 97, 98, 98.5],
      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    }
  },
  {
    id: '2',
    name: 'Arduino Nano 33 BLE',
    deviceId: 'nano33-ble-002',
    status: 'online',
    model: 'LunarLander A2C v0.9.1',
    reward: 85.3,
    latency: 8.7,
    powerConsumption: 18.2,
    uptime: '3d 8h 15m',
    lastSeen: '2024-01-20T15:28:00Z',
    location: 'Lab B',
    firmware: 'v1.1.8',
    signalStrength: 72,
    batteryLevel: 78,
    rewardHistory: {
      data: [80, 82, 84, 83, 85, 84, 85.3],
      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    }
  },
  {
    id: '3',
    name: 'ESP32 DevKit',
    deviceId: 'esp32-devkit-003',
    status: 'offline',
    model: 'Acrobot DQN v1.2.0',
    reward: 92.7,
    latency: 6.1,
    powerConsumption: 15.8,
    uptime: '1d 2h 45m',
    lastSeen: '2024-01-19T10:15:00Z',
    location: 'Field Test',
    firmware: 'v1.0.5',
    signalStrength: 0,
    batteryLevel: 45,
    rewardHistory: {
      data: [90, 91, 92, 91, 92, 92.5, 92.7],
      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    }
  },
  {
    id: '4',
    name: 'Raspberry Pi Pico',
    deviceId: 'pico-004',
    status: 'warning',
    model: 'Pendulum SAC v0.8.2',
    reward: 45.2,
    latency: 12.3,
    powerConsumption: 25.1,
    uptime: '5d 16h 22m',
    lastSeen: '2024-01-20T15:25:00Z',
    location: 'Office',
    firmware: 'v1.3.1',
    signalStrength: 65,
    batteryLevel: 23,
    rewardHistory: {
      data: [50, 48, 45, 47, 44, 46, 45.2],
      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    }
  }
];

export default function FleetPage() {
  const [devices, setDevices] = useState(mockDevices);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const filteredDevices = devices.filter(device => {
    const matchesSearch = device.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         device.deviceId.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         device.model.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || device.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'offline':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'bg-green-100 text-green-800';
      case 'offline':
        return 'bg-red-100 text-red-800';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getSignalIcon = (strength: number) => {
    if (strength === 0) return <WifiOff className="h-4 w-4 text-red-500" />;
    if (strength >= 80) return <Wifi className="h-4 w-4 text-green-500" />;
    if (strength >= 60) return <Signal className="h-4 w-4 text-yellow-500" />;
    return <Signal className="h-4 w-4 text-red-500" />;
  };

  const getBatteryColor = (level: number) => {
    if (level >= 80) return 'text-green-600';
    if (level >= 60) return 'text-yellow-600';
    if (level >= 40) return 'text-orange-600';
    return 'text-red-600';
  };

  const fleetStats = {
    total: devices.length,
    online: devices.filter(d => d.status === 'online').length,
    offline: devices.filter(d => d.status === 'offline').length,
    warning: devices.filter(d => d.status === 'warning').length,
    avgReward: devices.reduce((sum, d) => sum + d.reward, 0) / devices.length,
    avgLatency: devices.reduce((sum, d) => sum + d.latency, 0) / devices.length
  };

  return (
    <div className="container-responsive py-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Device Fleet</h1>
            <p className="text-gray-600 mt-1">
              Monitor and manage your deployed devices
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button variant="outline">
              <Upload className="mr-2 h-4 w-4" />
              Deploy Update
            </Button>
            <Button>
              <Download className="mr-2 h-4 w-4" />
              Add Device
            </Button>
          </div>
        </div>

        {/* Fleet Stats */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-gray-900">{fleetStats.total}</div>
            <div className="text-sm text-gray-600">Total Devices</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-green-600">{fleetStats.online}</div>
            <div className="text-sm text-gray-600">Online</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-red-600">{fleetStats.offline}</div>
            <div className="text-sm text-gray-600">Offline</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-yellow-600">{fleetStats.warning}</div>
            <div className="text-sm text-gray-600">Warning</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-gray-900">{fleetStats.avgReward.toFixed(1)}%</div>
            <div className="text-sm text-gray-600">Avg Reward</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-gray-900">{fleetStats.avgLatency.toFixed(1)}ms</div>
            <div className="text-sm text-gray-600">Avg Latency</div>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <Input
              placeholder="Search devices..."
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
              <option value="online">Online</option>
              <option value="offline">Offline</option>
              <option value="warning">Warning</option>
            </select>
            <Button variant="outline" size="sm">
              <Filter className="mr-2 h-4 w-4" />
              Filter
            </Button>
          </div>
        </div>
      </div>

      {/* Devices Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredDevices.map((device) => (
          <Card key={device.id} className="hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(device.status)}
                  <div>
                    <CardTitle className="text-lg">{device.name}</CardTitle>
                    <CardDescription>
                      {device.deviceId} • {device.location}
                    </CardDescription>
                  </div>
                </div>
                <Badge className={getStatusColor(device.status)}>
                  {device.status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              {/* Model Info */}
              <div className="mb-4">
                <p className="text-sm font-medium text-gray-500">Current Model</p>
                <p className="text-sm font-semibold">{device.model}</p>
              </div>

              {/* Performance Metrics */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-sm font-medium text-gray-500">Reward</p>
                  <p className="text-lg font-semibold">{device.reward}%</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Latency</p>
                  <p className="text-lg font-semibold">{device.latency}ms</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Power</p>
                  <p className="text-sm font-semibold">{device.powerConsumption}mW</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-500">Uptime</p>
                  <p className="text-sm font-semibold">{device.uptime}</p>
                </div>
              </div>

              {/* Reward History Chart */}
              <div className="mb-4">
                <p className="text-sm font-medium text-gray-500 mb-2">Reward Trend</p>
                <div className="h-16">
                  <SparklineChart 
                    data={device.rewardHistory} 
                    width={200} 
                    height={60} 
                    showArea={false}
                  />
                </div>
              </div>

              {/* Device Status */}
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-1">
                    {getSignalIcon(device.signalStrength)}
                    <span className="text-gray-600">{device.signalStrength}%</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Battery className={`h-4 w-4 ${getBatteryColor(device.batteryLevel)}`} />
                    <span className={`${getBatteryColor(device.batteryLevel)}`}>
                      {device.batteryLevel}%
                    </span>
                  </div>
                </div>
                <div className="flex items-center space-x-1">
                  <Activity className="h-4 w-4 text-gray-400" />
                  <span className="text-gray-600 text-xs">
                    {new Date(device.lastSeen).toLocaleTimeString()}
                  </span>
                </div>
              </div>

              {/* Actions */}
              <div className="mt-4 pt-4 border-t">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-gray-500">
                    Firmware: {device.firmware}
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">
                      <Upload className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" size="sm">
                      <Activity className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {filteredDevices.length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-400 mb-4">
            <Cpu className="mx-auto h-12 w-12" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No devices found</h3>
          <p className="text-gray-600 mb-4">
            {searchTerm ? 'Try adjusting your search terms.' : 'Add your first device to get started.'}
          </p>
          {!searchTerm && (
            <Button>
              <Download className="mr-2 h-4 w-4" />
              Add Device
            </Button>
          )}
        </div>
      )}
    </div>
  );
} 