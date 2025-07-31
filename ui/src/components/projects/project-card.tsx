'use client';

import React from 'react';
import Link from 'next/link';
import { formatDistanceToNow } from 'date-fns';
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  PlayCircle, 
  AlertCircle,
  HardDrive,
  Shield,
  Tag
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn, formatBytes, getStatusColor, getStatusBgColor } from '@/lib/utils';
import { type Project } from '@/types';
import { SparklineChart } from '@/components/charts/sparkline-chart';

interface ProjectCardProps {
  project: Project;
  viewMode: 'grid' | 'list';
}

export function ProjectCard({ project, viewMode }: ProjectCardProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <PlayCircle className="h-4 w-4 text-status-running" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-status-success" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-status-error" />;
      case 'idle':
        return <Clock className="h-4 w-4 text-status-idle" />;
      default:
        return <AlertCircle className="h-4 w-4 text-status-warning" />;
    }
  };

  const getProofStatusIcon = (status: string) => {
    switch (status) {
      case 'verified':
        return <Shield className="h-4 w-4 text-status-success" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-status-error" />;
      default:
        return <Clock className="h-4 w-4 text-status-warning" />;
    }
  };

  const getProofStatusText = (status: string) => {
    switch (status) {
      case 'verified':
        return 'Verified';
      case 'failed':
        return 'Failed';
      default:
        return 'Pending';
    }
  };

  // Mock sparkline data - in real app, this would come from API
  const sparklineData = {
    values: [85, 87, 89, 92, 90, 93, 95, 94, 96, 98, 97, 98.5],
    timestamps: Array.from({ length: 12 }, (_, i) => 
      new Date(Date.now() - (11 - i) * 24 * 60 * 60 * 1000).toISOString()
    ),
    color: project.status === 'completed' ? '#22c55e' : '#0ea5e9',
  };

  if (viewMode === 'list') {
    return (
      <Link href={`/projects/${project.id}`}>
        <div className="bg-white rounded-lg border shadow-sm hover:shadow-md transition-shadow p-6">
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3 mb-2">
                <h3 className="text-lg font-semibold text-gray-900 truncate">
                  {project.name}
                </h3>
                <div className="flex items-center gap-1">
                  {getStatusIcon(project.status)}
                  <span className={cn("text-sm font-medium", getStatusColor(project.status))}>
                    {project.status}
                  </span>
                </div>
              </div>
              
              <p className="text-gray-600 text-sm mb-3 line-clamp-2">
                {project.description}
              </p>

              <div className="flex items-center gap-4 text-sm text-gray-500 mb-3">
                <span>{project.environment}</span>
                <span>•</span>
                <span>{project.algorithm}</span>
                <span>•</span>
                <span>{formatDistanceToNow(new Date(project.updatedAt), { addSuffix: true })}</span>
              </div>

              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4 text-gray-400" />
                  <span className="text-sm text-gray-600">{formatBytes(project.binarySize)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-900">{project.reward}%</span>
                  <span className="text-sm text-gray-600">reward</span>
                </div>
                <div className="flex items-center gap-2">
                  {getProofStatusIcon(project.proofStatus)}
                  <span className="text-sm text-gray-600">{getProofStatusText(project.proofStatus)}</span>
                </div>
              </div>
            </div>

            <div className="flex flex-col items-end gap-2 ml-4">
              <div className="w-32 h-16">
                <SparklineChart data={sparklineData} />
              </div>
              <div className="flex flex-wrap gap-1">
                {project.tags.slice(0, 3).map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    {tag}
                  </Badge>
                ))}
                {project.tags.length > 3 && (
                  <Badge variant="secondary" className="text-xs">
                    +{project.tags.length - 3}
                  </Badge>
                )}
              </div>
            </div>
          </div>
        </div>
      </Link>
    );
  }

  return (
    <Link href={`/projects/${project.id}`}>
      <div className="bg-white rounded-lg border shadow-sm hover:shadow-md transition-all duration-200 hover:-translate-y-1 p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-semibold text-gray-900 truncate mb-1">
              {project.name}
            </h3>
            <div className="flex items-center gap-2">
              {getStatusIcon(project.status)}
              <span className={cn("text-sm font-medium", getStatusColor(project.status))}>
                {project.status}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-1">
            {getProofStatusIcon(project.proofStatus)}
            <span className="text-xs text-gray-600">{getProofStatusText(project.proofStatus)}</span>
          </div>
        </div>

        {/* Description */}
        <p className="text-gray-600 text-sm mb-4 line-clamp-2">
          {project.description}
        </p>

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{project.reward}%</div>
            <div className="text-xs text-gray-600">Reward</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-900">{formatBytes(project.binarySize)}</div>
            <div className="text-xs text-gray-600">Size</div>
          </div>
        </div>

        {/* Sparkline */}
        <div className="mb-4">
          <SparklineChart data={sparklineData} />
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span>{project.environment}</span>
            <span>•</span>
            <span>{project.algorithm}</span>
          </div>
          <div className="text-xs text-gray-500">
            {formatDistanceToNow(new Date(project.updatedAt), { addSuffix: true })}
          </div>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-1 mt-3">
          {project.tags.slice(0, 2).map((tag) => (
            <Badge key={tag} variant="secondary" className="text-xs">
              {tag}
            </Badge>
          ))}
          {project.tags.length > 2 && (
            <Badge variant="secondary" className="text-xs">
              +{project.tags.length - 2}
            </Badge>
          )}
        </div>
      </div>
    </Link>
  );
} 