'use client';

import React, { useState, useEffect } from 'react';
import { Plus, Search, Filter, Grid, List } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { ProjectCard } from '@/components/projects/project-card';
import { ProjectCardSkeleton } from '@/components/projects/project-card-skeleton';
import { CreateProjectDialog } from '@/components/projects/create-project-dialog';
import { type Project } from '@/types';

// Mock data for demonstration
const mockProjects: Project[] = [
  {
    id: '1',
    name: 'CartPole PPO Agent',
    description: 'A PPO agent trained on CartPole-v1 environment with 98.5% reward performance.',
    environment: 'CartPole-v1',
    algorithm: 'PPO',
    createdAt: '2024-01-15T10:30:00Z',
    updatedAt: '2024-01-20T14:45:00Z',
    status: 'completed',
    reward: 98.5,
    binarySize: 2048,
    proofStatus: 'verified',
    lastRunAt: '2024-01-20T14:45:00Z',
    totalRuns: 15,
    averageReward: 97.2,
    bestReward: 98.5,
    tags: ['ppo', 'cartpole', 'verified'],
  },
  {
    id: '2',
    name: 'LunarLander A2C',
    description: 'A2C agent for LunarLander-v2 with optimized memory usage and latency.',
    environment: 'LunarLander-v2',
    algorithm: 'A2C',
    createdAt: '2024-01-10T09:15:00Z',
    updatedAt: '2024-01-18T16:20:00Z',
    status: 'running',
    reward: 85.3,
    binarySize: 4096,
    proofStatus: 'pending',
    lastRunAt: '2024-01-18T16:20:00Z',
    totalRuns: 8,
    averageReward: 82.1,
    bestReward: 85.3,
    tags: ['a2c', 'lunarlander', 'optimized'],
  },
  {
    id: '3',
    name: 'Acrobot DQN',
    description: 'Deep Q-Network implementation for Acrobot-v1 with quantized weights.',
    environment: 'Acrobot-v1',
    algorithm: 'DQN',
    createdAt: '2024-01-05T11:00:00Z',
    updatedAt: '2024-01-15T13:30:00Z',
    status: 'completed',
    reward: 92.7,
    binarySize: 3072,
    proofStatus: 'verified',
    lastRunAt: '2024-01-15T13:30:00Z',
    totalRuns: 12,
    averageReward: 90.5,
    bestReward: 92.7,
    tags: ['dqn', 'acrobot', 'quantized'],
  },
  {
    id: '4',
    name: 'Pendulum SAC',
    description: 'Soft Actor-Critic agent for continuous control in Pendulum-v1.',
    environment: 'Pendulum-v1',
    algorithm: 'SAC',
    createdAt: '2024-01-12T08:45:00Z',
    updatedAt: '2024-01-19T10:15:00Z',
    status: 'failed',
    reward: 45.2,
    binarySize: 5120,
    proofStatus: 'failed',
    lastRunAt: '2024-01-19T10:15:00Z',
    totalRuns: 5,
    averageReward: 42.8,
    bestReward: 45.2,
    tags: ['sac', 'pendulum', 'continuous'],
  },
];

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [showCreateDialog, setShowCreateDialog] = useState(false);

  useEffect(() => {
    // Simulate API call
    const fetchProjects = async () => {
      setLoading(true);
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      setProjects(mockProjects);
      setLoading(false);
    };

    fetchProjects();
  }, []);

  const filteredProjects = projects.filter(project =>
    project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    project.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    project.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const stats = {
    total: projects.length,
    running: projects.filter(p => p.status === 'running').length,
    completed: projects.filter(p => p.status === 'completed').length,
    failed: projects.filter(p => p.status === 'failed').length,
  };

  return (
    <div className="container-responsive py-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Projects</h1>
            <p className="text-gray-600 mt-1">
              Manage your reinforcement learning projects and models
            </p>
          </div>
          <Button onClick={() => setShowCreateDialog(true)}>
            <Plus className="mr-2 h-4 w-4" />
            New Project
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
            <div className="text-sm text-gray-600">Total Projects</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-status-running">{stats.running}</div>
            <div className="text-sm text-gray-600">Running</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-status-success">{stats.completed}</div>
            <div className="text-sm text-gray-600">Completed</div>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border">
            <div className="text-2xl font-bold text-status-error">{stats.failed}</div>
            <div className="text-sm text-gray-600">Failed</div>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <Input
              placeholder="Search projects..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Filter className="mr-2 h-4 w-4" />
              Filter
            </Button>
            <div className="flex border rounded-md">
              <Button
                variant={viewMode === 'grid' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('grid')}
                className="rounded-r-none"
              >
                <Grid className="h-4 w-4" />
              </Button>
              <Button
                variant={viewMode === 'list' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('list')}
                className="rounded-l-none"
              >
                <List className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Projects Grid/List */}
      {loading ? (
        <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}>
          {Array.from({ length: 6 }).map((_, i) => (
            <ProjectCardSkeleton key={i} viewMode={viewMode} />
          ))}
        </div>
      ) : filteredProjects.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-400 mb-4">
            <Search className="mx-auto h-12 w-12" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No projects found</h3>
          <p className="text-gray-600 mb-4">
            {searchTerm ? 'Try adjusting your search terms.' : 'Get started by creating your first project.'}
          </p>
          {!searchTerm && (
            <Button onClick={() => setShowCreateDialog(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Create Project
            </Button>
          )}
        </div>
      ) : (
        <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}>
          {filteredProjects.map((project) => (
            <ProjectCard key={project.id} project={project} viewMode={viewMode} />
          ))}
        </div>
      )}

      {/* Create Project Dialog */}
      <CreateProjectDialog
        open={showCreateDialog}
        onOpenChange={setShowCreateDialog}
        onProjectCreated={(newProject) => {
          setProjects(prev => [newProject, ...prev]);
          setShowCreateDialog(false);
        }}
      />
    </div>
  );
} 