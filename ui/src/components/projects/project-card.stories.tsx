import type { Meta, StoryObj } from '@storybook/react';
import { ProjectCard } from './project-card';
import { type Project } from '@/types';

const meta: Meta<typeof ProjectCard> = {
  title: 'Components/Projects/ProjectCard',
  component: ProjectCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    viewMode: {
      control: { type: 'select' },
      options: ['grid', 'list'],
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

const mockProject: Project = {
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
};

export const GridView: Story = {
  args: {
    project: mockProject,
    viewMode: 'grid',
  },
};

export const ListView: Story = {
  args: {
    project: mockProject,
    viewMode: 'list',
  },
};

export const RunningProject: Story = {
  args: {
    project: {
      ...mockProject,
      id: '2',
      name: 'LunarLander A2C',
      status: 'running',
      proofStatus: 'pending',
      reward: 85.3,
      algorithm: 'A2C',
      environment: 'LunarLander-v2',
      tags: ['a2c', 'lunarlander', 'running'],
    },
    viewMode: 'grid',
  },
};

export const FailedProject: Story = {
  args: {
    project: {
      ...mockProject,
      id: '3',
      name: 'Pendulum SAC',
      status: 'failed',
      proofStatus: 'failed',
      reward: 45.2,
      algorithm: 'SAC',
      environment: 'Pendulum-v1',
      tags: ['sac', 'pendulum', 'failed'],
    },
    viewMode: 'grid',
  },
};

export const HighRewardProject: Story = {
  args: {
    project: {
      ...mockProject,
      id: '4',
      name: 'Acrobot DQN',
      reward: 99.8,
      algorithm: 'DQN',
      environment: 'Acrobot-v1',
      tags: ['dqn', 'acrobot', 'high-reward'],
    },
    viewMode: 'grid',
  },
};

export const LargeBinaryProject: Story = {
  args: {
    project: {
      ...mockProject,
      id: '5',
      name: 'Complex Environment Model',
      binarySize: 8192,
      reward: 92.1,
      algorithm: 'PPO',
      environment: 'ComplexEnv-v1',
      tags: ['ppo', 'complex', 'large-binary'],
    },
    viewMode: 'grid',
  },
}; 