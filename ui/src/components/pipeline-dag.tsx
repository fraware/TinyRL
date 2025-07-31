'use client';

import React, { useState, useCallback, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  NodeTypes,
  Handle,
  Position,
} from 'react-flow-renderer';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Play, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Cpu, 
  Zap, 
  Shield,
  Activity,
  Settings,
  FileBinary
} from 'lucide-react';

interface PipelineStep {
  id: string;
  name: string;
  type: 'data_collection' | 'training' | 'quantization' | 'verification' | 'deployment';
  status: 'pending' | 'running' | 'completed' | 'failed';
  duration?: string;
  metrics?: {
    reward?: number;
    latency?: number;
    powerConsumption?: number;
    binarySize?: number;
  };
}

interface PipelineDAGProps {
  pipelineId: string;
  steps: PipelineStep[];
  onStepClick?: (stepId: string) => void;
}

// Custom node types
const CircleNode = ({ data }: { data: any }) => (
  <div className="relative">
    <Handle type="target" position={Position.Top} />
    <div className="w-16 h-16 rounded-full border-2 flex items-center justify-center bg-white shadow-md">
      {data.status === 'completed' && <CheckCircle className="h-6 w-6 text-green-500" />}
      {data.status === 'running' && <Activity className="h-6 w-6 text-blue-500 animate-pulse" />}
      {data.status === 'failed' && <AlertCircle className="h-6 w-6 text-red-500" />}
      {data.status === 'pending' && <Clock className="h-6 w-6 text-gray-400" />}
    </div>
    <Handle type="source" position={Position.Bottom} />
  </div>
);

const HexagonNode = ({ data }: { data: any }) => (
  <div className="relative">
    <Handle type="target" position={Position.Top} />
    <div className="w-20 h-16 bg-white border-2 shadow-md transform rotate-45 flex items-center justify-center">
      <div className="transform -rotate-45">
        {data.type === 'training' && <Cpu className="h-5 w-5 text-blue-500" />}
        {data.type === 'quantization' && <Zap className="h-5 w-5 text-yellow-500" />}
        {data.type === 'verification' && <Shield className="h-5 w-5 text-green-500" />}
        {data.type === 'deployment' && <FileBinary className="h-5 w-5 text-purple-500" />}
      </div>
    </div>
    <Handle type="source" position={Position.Bottom} />
  </div>
);

const RectangleNode = ({ data }: { data: any }) => (
  <div className="relative">
    <Handle type="target" position={Position.Top} />
    <div className="w-24 h-12 bg-white border-2 shadow-md rounded flex items-center justify-center">
      <Settings className="h-5 w-5 text-gray-500" />
    </div>
    <Handle type="source" position={Position.Bottom} />
  </div>
);

const nodeTypes: NodeTypes = {
  circle: CircleNode,
  hexagon: HexagonNode,
  rectangle: RectangleNode,
};

export function PipelineDAG({ pipelineId, steps, onStepClick }: PipelineDAGProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // Convert steps to nodes
  const initialNodes: Node[] = useMemo(() => {
    return steps.map((step, index) => {
      const nodeType = step.type === 'training' ? 'hexagon' : 
                      step.type === 'quantization' ? 'hexagon' : 
                      step.type === 'verification' ? 'hexagon' : 'circle';
      
      return {
        id: step.id,
        type: nodeType,
        position: { x: index * 200, y: 100 },
        data: {
          label: step.name,
          status: step.status,
          type: step.type,
          duration: step.duration,
          metrics: step.metrics,
        },
      };
    });
  }, [steps]);

  // Create edges between nodes
  const initialEdges: Edge[] = useMemo(() => {
    return steps.slice(0, -1).map((step, index) => ({
      id: `e${step.id}-${steps[index + 1].id}`,
      source: step.id,
      target: steps[index + 1].id,
      type: 'smoothstep',
      animated: steps[index + 1].status === 'running',
      style: {
        stroke: steps[index + 1].status === 'running' ? '#3B82F6' : '#9CA3AF',
        strokeWidth: 2,
      },
    }));
  }, [steps]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback((event: any, node: Node) => {
    setSelectedNode(node.id);
    onStepClick?.(node.id);
  }, [onStepClick]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const selectedStep = steps.find(step => step.id === selectedNode);

  return (
    <div className="h-[600px] border rounded-lg bg-gray-50">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
      </ReactFlow>

      {/* Node Details Panel */}
      {selectedStep && (
        <div className="absolute top-4 right-4 w-80">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{selectedStep.name}</CardTitle>
                <Badge className={getStatusColor(selectedStep.status)}>
                  {selectedStep.status}
                </Badge>
              </div>
              <CardDescription>
                {selectedStep.type.replace('_', ' ').toUpperCase()}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {selectedStep.duration && (
                  <div>
                    <p className="text-sm font-medium text-gray-500">Duration</p>
                    <p className="text-sm">{selectedStep.duration}</p>
                  </div>
                )}
                
                {selectedStep.metrics && (
                  <div className="grid grid-cols-2 gap-4">
                    {selectedStep.metrics.reward && (
                      <div>
                        <p className="text-sm font-medium text-gray-500">Reward</p>
                        <p className="text-sm font-semibold">{selectedStep.metrics.reward}%</p>
                      </div>
                    )}
                    {selectedStep.metrics.latency && (
                      <div>
                        <p className="text-sm font-medium text-gray-500">Latency</p>
                        <p className="text-sm font-semibold">{selectedStep.metrics.latency}ms</p>
                      </div>
                    )}
                    {selectedStep.metrics.powerConsumption && (
                      <div>
                        <p className="text-sm font-medium text-gray-500">Power</p>
                        <p className="text-sm font-semibold">{selectedStep.metrics.powerConsumption}mW</p>
                      </div>
                    )}
                    {selectedStep.metrics.binarySize && (
                      <div>
                        <p className="text-sm font-medium text-gray-500">Size</p>
                        <p className="text-sm font-semibold">{selectedStep.metrics.binarySize} bytes</p>
                      </div>
                    )}
                  </div>
                )}

                <div className="flex space-x-2">
                  <Button variant="outline" size="sm" className="flex-1">
                    <Activity className="h-4 w-4 mr-2" />
                    View Logs
                  </Button>
                  <Button variant="outline" size="sm" className="flex-1">
                    <Settings className="h-4 w-4 mr-2" />
                    Configure
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Pipeline Controls */}
      <div className="absolute bottom-4 left-4">
        <div className="flex space-x-2">
          <Button size="sm" variant="outline">
            <Play className="h-4 w-4 mr-2" />
            Start Pipeline
          </Button>
          <Button size="sm" variant="outline">
            <Activity className="h-4 w-4 mr-2" />
            View Logs
          </Button>
        </div>
      </div>
    </div>
  );
} 