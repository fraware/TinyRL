import React from 'react';

interface ProjectCardSkeletonProps {
  viewMode: 'grid' | 'list';
}

export function ProjectCardSkeleton({ viewMode }: ProjectCardSkeletonProps) {
  if (viewMode === 'list') {
    return (
      <div className="bg-white rounded-lg border shadow-sm p-6">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-2">
              <div className="skeleton h-6 w-48" />
              <div className="skeleton h-4 w-16" />
            </div>
            
            <div className="skeleton h-4 w-full mb-2" />
            <div className="skeleton h-4 w-3/4 mb-3" />

            <div className="flex items-center gap-4 mb-3">
              <div className="skeleton h-3 w-20" />
              <div className="skeleton h-3 w-3" />
              <div className="skeleton h-3 w-16" />
              <div className="skeleton h-3 w-3" />
              <div className="skeleton h-3 w-24" />
            </div>

            <div className="flex items-center gap-4">
              <div className="skeleton h-4 w-16" />
              <div className="skeleton h-4 w-12" />
              <div className="skeleton h-4 w-16" />
            </div>
          </div>

          <div className="flex flex-col items-end gap-2 ml-4">
            <div className="skeleton w-32 h-16" />
            <div className="flex gap-1">
              <div className="skeleton h-5 w-12" />
              <div className="skeleton h-5 w-10" />
              <div className="skeleton h-5 w-8" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border shadow-sm p-6">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <div className="skeleton h-6 w-32 mb-2" />
          <div className="skeleton h-4 w-20" />
        </div>
        <div className="skeleton h-4 w-16" />
      </div>

      {/* Description */}
      <div className="skeleton h-4 w-full mb-2" />
      <div className="skeleton h-4 w-3/4 mb-4" />

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="text-center">
          <div className="skeleton h-8 w-16 mx-auto mb-1" />
          <div className="skeleton h-3 w-12 mx-auto" />
        </div>
        <div className="text-center">
          <div className="skeleton h-6 w-14 mx-auto mb-1" />
          <div className="skeleton h-3 w-8 mx-auto" />
        </div>
      </div>

      {/* Sparkline */}
      <div className="skeleton h-16 w-full mb-4" />

      {/* Footer */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="skeleton h-3 w-16" />
          <div className="skeleton h-3 w-3" />
          <div className="skeleton h-3 w-12" />
        </div>
        <div className="skeleton h-3 w-20" />
      </div>

      {/* Tags */}
      <div className="flex gap-1 mt-3">
        <div className="skeleton h-5 w-12" />
        <div className="skeleton h-5 w-10" />
        <div className="skeleton h-5 w-8" />
      </div>
    </div>
  );
} 