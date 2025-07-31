'use client';

import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { 
  Search, 
  Command, 
  Settings, 
  Plus, 
  Download, 
  Upload,
  Cpu,
  FileBinary,
  Activity,
  Users,
  BarChart3,
  HelpCircle,
  X
} from 'lucide-react';

interface CommandItem {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  action: () => void;
  category: string;
  keywords: string[];
}

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);

  const commands: CommandItem[] = [
    {
      id: 'new-project',
      title: 'Create New Project',
      description: 'Start a new reinforcement learning project',
      icon: Plus,
      action: () => {
        // Navigate to create project
        onOpenChange(false);
      },
      category: 'Projects',
      keywords: ['create', 'new', 'project', 'start']
    },
    {
      id: 'new-pipeline',
      title: 'Start Pipeline',
      description: 'Run a new training pipeline',
      icon: Activity,
      action: () => {
        // Navigate to pipelines
        onOpenChange(false);
      },
      category: 'Pipelines',
      keywords: ['pipeline', 'train', 'run', 'start']
    },
    {
      id: 'deploy-model',
      title: 'Deploy Model',
      description: 'Deploy a model to the fleet',
      icon: Upload,
      action: () => {
        // Navigate to fleet
        onOpenChange(false);
      },
      category: 'Fleet',
      keywords: ['deploy', 'fleet', 'device', 'upload']
    },
    {
      id: 'download-artifact',
      title: 'Download Artifact',
      description: 'Download a compiled binary',
      icon: Download,
      action: () => {
        // Navigate to artifacts
        onOpenChange(false);
      },
      category: 'Artifacts',
      keywords: ['download', 'artifact', 'binary', 'file']
    },
    {
      id: 'view-dashboard',
      title: 'View Dashboard',
      description: 'Go to the main dashboard',
      icon: BarChart3,
      action: () => {
        // Navigate to dashboard
        onOpenChange(false);
      },
      category: 'Navigation',
      keywords: ['dashboard', 'overview', 'home', 'main']
    },
    {
      id: 'manage-fleet',
      title: 'Manage Fleet',
      description: 'View and manage devices',
      icon: Cpu,
      action: () => {
        // Navigate to fleet
        onOpenChange(false);
      },
      category: 'Fleet',
      keywords: ['fleet', 'devices', 'manage', 'monitor']
    },
    {
      id: 'view-artifacts',
      title: 'View Artifacts',
      description: 'Browse compiled models',
      icon: FileBinary,
      action: () => {
        // Navigate to artifacts
        onOpenChange(false);
      },
      category: 'Artifacts',
      keywords: ['artifacts', 'models', 'binaries', 'files']
    },
    {
      id: 'view-pipelines',
      title: 'View Pipelines',
      description: 'Monitor training pipelines',
      icon: Activity,
      action: () => {
        // Navigate to pipelines
        onOpenChange(false);
      },
      category: 'Pipelines',
      keywords: ['pipelines', 'training', 'monitor', 'status']
    },
    {
      id: 'settings',
      title: 'Settings',
      description: 'Configure application settings',
      icon: Settings,
      action: () => {
        // Navigate to settings
        onOpenChange(false);
      },
      category: 'System',
      keywords: ['settings', 'configure', 'preferences', 'options']
    },
    {
      id: 'help',
      title: 'Help & Documentation',
      description: 'View help and documentation',
      icon: HelpCircle,
      action: () => {
        // Navigate to help
        onOpenChange(false);
      },
      category: 'System',
      keywords: ['help', 'docs', 'documentation', 'support']
    }
  ];

  const filteredCommands = commands.filter(command =>
    command.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    command.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    command.keywords.some(keyword => keyword.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const groupedCommands = filteredCommands.reduce((groups, command) => {
    if (!groups[command.category]) {
      groups[command.category] = [];
    }
    groups[command.category].push(command);
    return groups;
  }, {} as Record<string, CommandItem[]>);

  useEffect(() => {
    setSelectedIndex(0);
  }, [searchTerm]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!open) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => 
            prev < filteredCommands.length - 1 ? prev + 1 : 0
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => 
            prev > 0 ? prev - 1 : filteredCommands.length - 1
          );
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
          }
          break;
        case 'Escape':
          e.preventDefault();
          onOpenChange(false);
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [open, selectedIndex, filteredCommands, onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl p-0">
        <div className="p-4 border-b">
          <div className="flex items-center space-x-2">
            <Search className="h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search commands..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="border-0 shadow-none focus:ring-0 text-lg"
              autoFocus
            />
            <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="max-h-96 overflow-y-auto">
          {Object.entries(groupedCommands).map(([category, categoryCommands]) => (
            <div key={category}>
              <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider bg-gray-50">
                {category}
              </div>
              {categoryCommands.map((command, index) => {
                const globalIndex = filteredCommands.indexOf(command);
                const isSelected = globalIndex === selectedIndex;
                
                return (
                  <button
                    key={command.id}
                    className={`w-full px-4 py-3 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none ${
                      isSelected ? 'bg-gray-50' : ''
                    }`}
                    onClick={command.action}
                  >
                    <div className="flex items-center space-x-3">
                      <command.icon className="h-5 w-5 text-gray-400" />
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">{command.title}</div>
                        <div className="text-sm text-gray-500">{command.description}</div>
                      </div>
                      <div className="flex items-center space-x-1 text-xs text-gray-400">
                        <Command className="h-3 w-3" />
                        <span>⌘</span>
                        <span>K</span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        {filteredCommands.length === 0 && (
          <div className="p-8 text-center text-gray-500">
            <Search className="h-8 w-8 mx-auto mb-2" />
            <p>No commands found</p>
            <p className="text-sm">Try a different search term</p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
} 