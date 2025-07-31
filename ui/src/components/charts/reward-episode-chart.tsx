'use client';

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export interface EpisodeData {
  episode: number;
  reward: number;
  latency: number;
  p50Latency?: number;
  p95Latency?: number;
  timestamp: string;
}

interface RewardEpisodeChartProps {
  data: EpisodeData[];
  title?: string;
  description?: string;
  showLatency?: boolean;
  showTrend?: boolean;
  className?: string;
  height?: number;
}

export function RewardEpisodeChart({
  data,
  title = "Reward Progress",
  description = "Training progress over episodes",
  showLatency = true,
  showTrend = true,
  className,
  height = 300
}: RewardEpisodeChartProps) {
  const latestData = data[data.length - 1];
  const previousData = data[data.length - 2];
  
  const rewardChange = previousData 
    ? ((latestData.reward - previousData.reward) / previousData.reward) * 100 
    : 0;
  
  const avgLatency = data.reduce((sum, d) => sum + d.latency, 0) / data.length;
  const p50Latency = data.sort((a, b) => a.latency - b.latency)[Math.floor(data.length * 0.5)]?.latency;
  const p95Latency = data.sort((a, b) => a.latency - b.latency)[Math.floor(data.length * 0.95)]?.latency;

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg">
          <p className="font-medium">Episode {label}</p>
          <p className="text-sm text-gray-600">
            Reward: <span className="font-medium">{data.reward.toFixed(2)}</span>
          </p>
          {showLatency && (
            <p className="text-sm text-gray-600">
              Latency: <span className="font-medium">{data.latency.toFixed(2)}ms</span>
            </p>
          )}
          {data.p50Latency && (
            <p className="text-sm text-gray-600">
              P50: <span className="font-medium">{data.p50Latency.toFixed(2)}ms</span>
            </p>
          )}
          {data.p95Latency && (
            <p className="text-sm text-gray-600">
              P95: <span className="font-medium">{data.p95Latency.toFixed(2)}ms</span>
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="h-4 w-4 text-green-500" />;
    if (change < 0) return <TrendingDown className="h-4 w-4 text-red-500" />;
    return <Minus className="h-4 w-4 text-gray-500" />;
  };

  const getTrendColor = (change: number) => {
    if (change > 0) return 'text-green-600';
    if (change < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          {showTrend && (
            <div className="flex items-center gap-2">
              {getTrendIcon(rewardChange)}
              <span className={`text-sm font-medium ${getTrendColor(rewardChange)}`}>
                {rewardChange > 0 ? '+' : ''}{rewardChange.toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                {latestData?.reward.toFixed(1)}
              </div>
              <div className="text-sm text-gray-600">Current Reward</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                {data.length}
              </div>
              <div className="text-sm text-gray-600">Episodes</div>
            </div>
            {showLatency && (
              <>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {avgLatency.toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-600">Avg Latency</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {p95Latency?.toFixed(1) || 'N/A'}
                  </div>
                  <div className="text-sm text-gray-600">P95 Latency</div>
                </div>
              </>
            )}
          </div>

          {/* Chart */}
          <div style={{ height }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="episode" 
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `E${value}`}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  domain={[0, 'dataMax + 10']}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="reward"
                  stroke="#3B82F6"
                  fill="#3B82F6"
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
                {showLatency && (
                  <Line
                    type="monotone"
                    dataKey="latency"
                    stroke="#F59E0B"
                    strokeWidth={1}
                    dot={false}
                    strokeDasharray="3 3"
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Latency Metrics */}
          {showLatency && (p50Latency || p95Latency) && (
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span>Reward</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full border-2 border-dashed"></div>
                <span>Latency</span>
              </div>
              {p50Latency && (
                <Badge variant="secondary" className="text-xs">
                  P50: {p50Latency.toFixed(1)}ms
                </Badge>
              )}
              {p95Latency && (
                <Badge variant="secondary" className="text-xs">
                  P95: {p95Latency.toFixed(1)}ms
                </Badge>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

interface MultiRewardChartProps {
  datasets: Array<{
    name: string;
    data: EpisodeData[];
    color: string;
  }>;
  title?: string;
  description?: string;
  height?: number;
  className?: string;
}

export function MultiRewardChart({
  datasets,
  title = "Comparison Chart",
  description = "Compare multiple training runs",
  height = 300,
  className
}: MultiRewardChartProps) {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg">
          <p className="font-medium">Episode {label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm text-gray-600">
              <span style={{ color: entry.color }}>●</span> {entry.name}: {entry.value.toFixed(2)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div style={{ height }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="episode" 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `E${value}`}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                domain={[0, 'dataMax + 10']}
              />
              <Tooltip content={<CustomTooltip />} />
              {datasets.map((dataset, index) => (
                <Line
                  key={index}
                  type="monotone"
                  data={dataset.data}
                  dataKey="reward"
                  name={dataset.name}
                  stroke={dataset.color}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
} 