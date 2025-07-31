'use client';

import React, { useState, useEffect } from 'react';
import { Check, ChevronsUpDown, GitBranch, Tag } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Badge } from '@/components/ui/badge';

export interface ModelSnapshot {
  id: string;
  name: string;
  tag: string;
  commit: string;
  createdAt: string;
  reward: number;
  binarySize: number;
  status: 'verified' | 'pending' | 'failed';
  description?: string;
}

interface ModelSnapshotSelectorProps {
  snapshots: ModelSnapshot[];
  selectedSnapshot?: ModelSnapshot;
  onSnapshotSelect: (snapshot: ModelSnapshot) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

export function ModelSnapshotSelector({
  snapshots,
  selectedSnapshot,
  onSnapshotSelect,
  placeholder = "Select model snapshot...",
  className,
  disabled = false
}: ModelSnapshotSelectorProps) {
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState('');

  const filteredSnapshots = snapshots.filter(snapshot =>
    snapshot.name.toLowerCase().includes(searchValue.toLowerCase()) ||
    snapshot.tag.toLowerCase().includes(searchValue.toLowerCase()) ||
    snapshot.commit.toLowerCase().includes(searchValue.toLowerCase())
  );

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

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'verified':
        return '✅';
      case 'pending':
        return '⏳';
      case 'failed':
        return '❌';
      default:
        return '⏳';
    }
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className={cn(
            "w-full justify-between",
            !selectedSnapshot && "text-muted-foreground",
            className
          )}
          disabled={disabled}
        >
          {selectedSnapshot ? (
            <div className="flex items-center gap-2">
              <GitBranch className="h-4 w-4" />
              <span className="truncate">{selectedSnapshot.name}</span>
              <Badge variant="secondary" className="text-xs">
                {selectedSnapshot.tag}
              </Badge>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <GitBranch className="h-4 w-4" />
              {placeholder}
            </div>
          )}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-full p-0" align="start">
        <Command>
          <CommandInput
            placeholder="Search snapshots..."
            value={searchValue}
            onValueChange={setSearchValue}
          />
          <CommandList>
            <CommandEmpty>No snapshots found.</CommandEmpty>
            <CommandGroup>
              {filteredSnapshots.map((snapshot) => (
                <CommandItem
                  key={snapshot.id}
                  value={snapshot.id}
                  onSelect={() => {
                    onSnapshotSelect(snapshot);
                    setOpen(false);
                    setSearchValue('');
                  }}
                  className="flex items-center justify-between"
                >
                  <div className="flex items-center gap-2">
                    <GitBranch className="h-4 w-4" />
                    <div className="flex flex-col">
                      <span className="font-medium">{snapshot.name}</span>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Tag className="h-3 w-3" />
                        <span>{snapshot.tag}</span>
                        <span>•</span>
                        <span>{snapshot.commit.substring(0, 7)}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge 
                      variant="secondary" 
                      className={cn("text-xs", getStatusColor(snapshot.status))}
                    >
                      {getStatusIcon(snapshot.status)} {snapshot.status}
                    </Badge>
                    {selectedSnapshot?.id === snapshot.id && (
                      <Check className="h-4 w-4" />
                    )}
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

interface ModelSnapshotCardProps {
  snapshot: ModelSnapshot;
  onSelect?: (snapshot: ModelSnapshot) => void;
  selected?: boolean;
  className?: string;
}

export function ModelSnapshotCard({
  snapshot,
  onSelect,
  selected = false,
  className
}: ModelSnapshotCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'verified':
        return 'border-green-200 bg-green-50';
      case 'pending':
        return 'border-yellow-200 bg-yellow-50';
      case 'failed':
        return 'border-red-200 bg-red-50';
      default:
        return 'border-gray-200 bg-gray-50';
    }
  };

  return (
    <div
      className={cn(
        "border rounded-lg p-4 cursor-pointer transition-all hover:shadow-md",
        selected && "ring-2 ring-blue-500",
        getStatusColor(snapshot.status),
        className
      )}
      onClick={() => onSelect?.(snapshot)}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-gray-500" />
          <h3 className="font-medium text-gray-900">{snapshot.name}</h3>
        </div>
        <Badge variant="secondary" className="text-xs">
          {snapshot.tag}
        </Badge>
      </div>
      
      <div className="space-y-2 text-sm text-gray-600">
        <div className="flex items-center gap-2">
          <Tag className="h-3 w-3" />
          <span>Commit: {snapshot.commit}</span>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1">
            <span>Reward:</span>
            <span className="font-medium">{snapshot.reward}%</span>
          </div>
          <div className="flex items-center gap-1">
            <span>Size:</span>
            <span className="font-medium">{snapshot.binarySize} KB</span>
          </div>
        </div>
        
        {snapshot.description && (
          <p className="text-xs text-gray-500 mt-2">
            {snapshot.description}
          </p>
        )}
      </div>
      
      <div className="mt-3 flex items-center justify-between">
        <Badge 
          variant="secondary" 
          className={cn(
            "text-xs",
            snapshot.status === 'verified' && "bg-green-100 text-green-800",
            snapshot.status === 'pending' && "bg-yellow-100 text-yellow-800",
            snapshot.status === 'failed' && "bg-red-100 text-red-800"
          )}
        >
          {snapshot.status === 'verified' && '✅'}
          {snapshot.status === 'pending' && '⏳'}
          {snapshot.status === 'failed' && '❌'}
          {snapshot.status}
        </Badge>
        
        <span className="text-xs text-gray-500">
          {new Date(snapshot.createdAt).toLocaleDateString()}
        </span>
      </div>
    </div>
  );
} 