'use client';

import React from 'react';
import { cn } from '@/lib/utils';

interface RadialGaugeProps {
  value: number;
  max: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  showValue?: boolean;
  showPercentage?: boolean;
  className?: string;
  label?: string;
  unit?: string;
}

export function RadialGauge({
  value,
  max,
  size = 120,
  strokeWidth = 8,
  color = '#3B82F6',
  showValue = true,
  showPercentage = true,
  className,
  label,
  unit
}: RadialGaugeProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const percentage = Math.min(Math.max(value / max, 0), 1);
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (percentage * circumference);

  const getColor = (value: number, max: number) => {
    const percentage = (value / max) * 100;
    if (percentage >= 80) return '#10B981'; // green
    if (percentage >= 60) return '#F59E0B'; // yellow
    if (percentage >= 40) return '#F97316'; // orange
    return '#EF4444'; // red
  };

  const gaugeColor = color === '#3B82F6' ? getColor(value, max) : color;

  return (
    <div className={cn('flex flex-col items-center', className)}>
      <div className="relative" style={{ width: size, height: size }}>
        {/* Background circle */}
        <svg
          width={size}
          height={size}
          className="transform -rotate-90"
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#E5E7EB"
            strokeWidth={strokeWidth}
            fill="none"
          />
          {/* Progress circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={gaugeColor}
            strokeWidth={strokeWidth}
            strokeDasharray={strokeDasharray}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="none"
            className="transition-all duration-300 ease-out"
          />
        </svg>
        
        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {showValue && (
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                {value}
                {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
              </div>
              {showPercentage && (
                <div className="text-sm text-gray-500">
                  {Math.round(percentage * 100)}%
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {label && (
        <div className="mt-2 text-sm font-medium text-gray-700 text-center">
          {label}
        </div>
      )}
    </div>
  );
}

interface MultiRadialGaugeProps {
  gauges: Array<{
    value: number;
    max: number;
    label: string;
    color?: string;
    unit?: string;
  }>;
  size?: number;
  className?: string;
}

export function MultiRadialGauge({ gauges, size = 80, className }: MultiRadialGaugeProps) {
  return (
    <div className={cn('grid grid-cols-2 gap-4', className)}>
      {gauges.map((gauge, index) => (
        <RadialGauge
          key={index}
          value={gauge.value}
          max={gauge.max}
          size={size}
          color={gauge.color}
          label={gauge.label}
          unit={gauge.unit}
          showValue={true}
          showPercentage={true}
        />
      ))}
    </div>
  );
} 