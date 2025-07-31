// Project Types
export interface Project {
  id: string;
  name: string;
  description: string;
  environment: string;
  algorithm: string;
  createdAt: string;
  updatedAt: string;
  status: 'idle' | 'running' | 'completed' | 'failed';
  reward: number;
  binarySize: number;
  proofStatus: 'pending' | 'verified' | 'failed';
  lastRunAt?: string;
  totalRuns: number;
  averageReward: number;
  bestReward: number;
  tags: string[];
}

// Pipeline Types
export interface Pipeline {
  id: string;
  projectId: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  steps: PipelineStep[];
  artifacts: Artifact[];
  metrics: PipelineMetrics;
}

export interface PipelineStep {
  id: string;
  name: string;
  type: 'training' | 'quantization' | 'pruning' | 'distillation' | 'verification' | 'codegen';
  status: 'pending' | 'running' | 'completed' | 'failed';
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  logs: string[];
  metrics: Record<string, number>;
}

export interface PipelineMetrics {
  reward: number;
  binarySize: number;
  latency: number;
  powerConsumption: number;
  memoryUsage: number;
  accuracy: number;
}

// Artifact Types
export interface Artifact {
  id: string;
  pipelineId: string;
  name: string;
  type: 'model' | 'binary' | 'proof' | 'sbom' | 'report';
  size: number;
  hash: string;
  createdAt: string;
  status: 'pending' | 'completed' | 'failed';
  metadata: Record<string, any>;
  downloadUrl?: string;
  verificationStatus: 'pending' | 'verified' | 'failed';
  signature?: string;
}

// Device Types
export interface Device {
  id: string;
  name: string;
  type: 'stm32' | 'esp32' | 'arduino' | 'raspberry-pi' | 'custom';
  status: 'online' | 'offline' | 'error';
  lastSeen: string;
  firmware: string;
  reward: number;
  latency: number;
  powerConsumption: number;
  memoryUsage: number;
  location?: string;
  tags: string[];
}

// Fleet Types
export interface Fleet {
  id: string;
  name: string;
  description: string;
  devices: Device[];
  totalDevices: number;
  onlineDevices: number;
  averageReward: number;
  averageLatency: number;
  lastDeployment?: string;
  deploymentStatus: 'idle' | 'deploying' | 'completed' | 'failed';
}

// Verification Types
export interface Proof {
  id: string;
  artifactId: string;
  type: 'lean' | 'smt' | 'formal';
  status: 'pending' | 'verified' | 'failed';
  createdAt: string;
  verifiedAt?: string;
  content: string;
  metadata: Record<string, any>;
}

// User and Authentication Types
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'developer' | 'viewer';
  createdAt: string;
  lastLoginAt?: string;
  preferences: UserPreferences;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  notifications: NotificationSettings;
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  slack: boolean;
  events: string[];
}

// API Response Types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  error?: string;
  pagination?: PaginationInfo;
}

export interface PaginationInfo {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
}

// Form Types
export interface CreateProjectForm {
  name: string;
  description: string;
  environment: string;
  algorithm: string;
  hyperparameters: Record<string, any>;
}

export interface CreatePipelineForm {
  projectId: string;
  name: string;
  steps: PipelineStepConfig[];
}

export interface PipelineStepConfig {
  type: string;
  config: Record<string, any>;
}

// Chart and Visualization Types
export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

export interface ChartDataset {
  label: string;
  data: number[];
  backgroundColor?: string;
  borderColor?: string;
  borderWidth?: number;
  fill?: boolean;
}

export interface SparklineData {
  values: number[];
  timestamps: string[];
  color: string;
  showArea?: boolean;
}

// Notification Types
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  createdAt: string;
  read: boolean;
  actionUrl?: string;
}

// Search and Filter Types
export interface SearchFilters {
  status?: string[];
  environment?: string[];
  algorithm?: string[];
  dateRange?: {
    start: string;
    end: string;
  };
  tags?: string[];
}

export interface SortConfig {
  field: string;
  direction: 'asc' | 'desc';
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'pipeline_update' | 'device_status' | 'notification' | 'error';
  data: any;
  timestamp: string;
}

// Error Types
export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
}

// Configuration Types
export interface AppConfig {
  api: {
    baseUrl: string;
    timeout: number;
  };
  websocket: {
    url: string;
    reconnectInterval: number;
  };
  features: {
    realTimeUpdates: boolean;
    advancedAnalytics: boolean;
    fleetManagement: boolean;
  };
} 