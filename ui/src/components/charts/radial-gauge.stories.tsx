import type { Meta, StoryObj } from '@storybook/react';
import { RadialGauge, MultiRadialGauge } from './radial-gauge';

const meta: Meta<typeof RadialGauge> = {
  title: 'Components/Charts/RadialGauge',
  component: RadialGauge,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    value: {
      control: { type: 'range', min: 0, max: 100, step: 1 },
    },
    max: {
      control: { type: 'number', min: 1, max: 1000 },
    },
    size: {
      control: { type: 'range', min: 60, max: 200, step: 10 },
    },
    strokeWidth: {
      control: { type: 'range', min: 2, max: 20, step: 1 },
    },
    color: {
      control: { type: 'color' },
    },
    showValue: {
      control: { type: 'boolean' },
    },
    showPercentage: {
      control: { type: 'boolean' },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    value: 75,
    max: 100,
    size: 120,
    strokeWidth: 8,
    color: '#3B82F6',
    showValue: true,
    showPercentage: true,
    label: 'Memory Usage',
    unit: 'MB',
  },
};

export const Small: Story = {
  args: {
    value: 45,
    max: 100,
    size: 80,
    strokeWidth: 6,
    label: 'CPU Usage',
    unit: '%',
  },
};

export const Large: Story = {
  args: {
    value: 92,
    max: 100,
    size: 160,
    strokeWidth: 12,
    label: 'Reward Performance',
    unit: '%',
  },
};

export const CustomColor: Story = {
  args: {
    value: 85,
    max: 100,
    size: 120,
    color: '#10B981',
    label: 'Success Rate',
    unit: '%',
  },
};

export const NoLabels: Story = {
  args: {
    value: 60,
    max: 100,
    size: 120,
    showValue: false,
    showPercentage: false,
  },
};

export const HighValue: Story = {
  args: {
    value: 95,
    max: 100,
    size: 120,
    label: 'Accuracy',
    unit: '%',
  },
};

export const LowValue: Story = {
  args: {
    value: 25,
    max: 100,
    size: 120,
    label: 'Error Rate',
    unit: '%',
  },
};

// MultiRadialGauge stories
const multiMeta: Meta<typeof MultiRadialGauge> = {
  title: 'Components/Charts/MultiRadialGauge',
  component: MultiRadialGauge,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
};

export const MultiGauge: StoryObj<typeof multiMeta> = {
  args: {
    gauges: [
      {
        value: 75,
        max: 100,
        label: 'Memory',
        unit: 'MB',
        color: '#3B82F6',
      },
      {
        value: 45,
        max: 100,
        label: 'CPU',
        unit: '%',
        color: '#10B981',
      },
      {
        value: 90,
        max: 100,
        label: 'Reward',
        unit: '%',
        color: '#F59E0B',
      },
      {
        value: 30,
        max: 100,
        label: 'Latency',
        unit: 'ms',
        color: '#EF4444',
      },
    ],
    size: 80,
  },
};

export const FleetMetrics: StoryObj<typeof multiMeta> = {
  args: {
    gauges: [
      {
        value: 156,
        max: 200,
        label: 'Devices',
        unit: '',
        color: '#3B82F6',
      },
      {
        value: 92,
        max: 100,
        label: 'Online',
        unit: '%',
        color: '#10B981',
      },
      {
        value: 12.3,
        max: 50,
        label: 'Avg Latency',
        unit: 'ms',
        color: '#F59E0B',
      },
      {
        value: 2847,
        max: 5000,
        label: 'Total Reward',
        unit: '',
        color: '#8B5CF6',
      },
    ],
    size: 80,
  },
}; 