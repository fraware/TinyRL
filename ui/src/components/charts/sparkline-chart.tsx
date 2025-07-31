'use client';

import React from 'react';
import { SparklineData } from '@/types';

interface SparklineChartProps {
  data: SparklineData;
  width?: number;
  height?: number;
  showArea?: boolean;
}

export function SparklineChart({ 
  data, 
  width = 120, 
  height = 40, 
  showArea = true 
}: SparklineChartProps) {
  if (!data.values || data.values.length === 0) {
    return (
      <div 
        className="bg-gray-100 rounded"
        style={{ width, height }}
      />
    );
  }

  const padding = 4;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  const minValue = Math.min(...data.values);
  const maxValue = Math.max(...data.values);
  const range = maxValue - minValue || 1;

  const points = data.values.map((value, index) => {
    const x = (index / (data.values.length - 1)) * chartWidth + padding;
    const y = height - padding - ((value - minValue) / range) * chartHeight;
    return `${x},${y}`;
  });

  const path = points.join(' ');

  // Create area path for fill
  const areaPath = points.length > 0 
    ? `${points[0].split(',')[0]},${height - padding} ${path} ${points[points.length - 1].split(',')[0]},${height - padding}`
    : '';

  return (
    <svg 
      width={width} 
      height={height} 
      className="overflow-visible"
      aria-label="Reward trend sparkline"
    >
      {showArea && (
        <path
          d={`M ${areaPath}`}
          fill={data.color}
          fillOpacity="0.1"
          stroke="none"
        />
      )}
      <path
        d={`M ${path}`}
        stroke={data.color}
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Add subtle dots at data points */}
      {data.values.map((value, index) => {
        const x = (index / (data.values.length - 1)) * chartWidth + padding;
        const y = height - padding - ((value - minValue) / range) * chartHeight;
        return (
          <circle
            key={index}
            cx={x}
            cy={y}
            r="1.5"
            fill={data.color}
            opacity="0.6"
          />
        );
      })}
    </svg>
  );
} 