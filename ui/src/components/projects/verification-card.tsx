'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  CheckCircle, 
  XCircle, 
  Clock, 
  AlertTriangle, 
  Eye, 
  Download,
  FileText,
  Shield,
  Zap
} from 'lucide-react';
import { cn } from '@/lib/utils';

export interface VerificationResult {
  id: string;
  status: 'verified' | 'pending' | 'failed' | 'timeout';
  createdAt: string;
  completedAt?: string;
  duration?: number;
  properties: VerificationProperty[];
  proof?: string;
  counterexample?: string;
  error?: string;
}

export interface VerificationProperty {
  id: string;
  name: string;
  description: string;
  status: 'verified' | 'failed' | 'timeout';
  proof?: string;
  counterexample?: string;
}

interface VerificationCardProps {
  verification: VerificationResult;
  onViewProof?: (proof: string) => void;
  onDownloadProof?: (proof: string) => void;
  className?: string;
}

export function VerificationCard({
  verification,
  onViewProof,
  onDownloadProof,
  className
}: VerificationCardProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'verified':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />;
      case 'timeout':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'pending':
        return <Clock className="h-5 w-5 text-blue-500" />;
      default:
        return <AlertTriangle className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'verified':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'timeout':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'pending':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'verified':
        return 'Verified';
      case 'failed':
        return 'Failed';
      case 'timeout':
        return 'Timeout';
      case 'pending':
        return 'Pending';
      default:
        return 'Unknown';
    }
  };

  const verifiedProperties = verification.properties.filter(p => p.status === 'verified').length;
  const totalProperties = verification.properties.length;
  const successRate = totalProperties > 0 ? (verifiedProperties / totalProperties) * 100 : 0;

  return (
    <Card className={cn("border-l-4", className)}>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            {getStatusIcon(verification.status)}
            <div>
              <CardTitle className="text-lg">Formal Verification</CardTitle>
              <CardDescription>
                Mathematical proof of model safety and correctness
              </CardDescription>
            </div>
          </div>
          <Badge 
            variant="secondary" 
            className={cn("text-sm font-medium", getStatusColor(verification.status))}
          >
            {getStatusText(verification.status)}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Progress Summary */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {verifiedProperties}
            </div>
            <div className="text-sm text-gray-600">Verified</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {totalProperties - verifiedProperties}
            </div>
            <div className="text-sm text-gray-600">Failed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {successRate.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Success Rate</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {verification.duration ? `${verification.duration}s` : 'N/A'}
            </div>
            <div className="text-sm text-gray-600">Duration</div>
          </div>
        </div>

        {/* Properties List */}
        <div className="space-y-2">
          <h4 className="font-medium text-gray-900">Properties</h4>
          <div className="space-y-2">
            {verification.properties.map((property) => (
              <div
                key={property.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-2">
                  {getStatusIcon(property.status)}
                  <div>
                    <div className="font-medium text-sm">{property.name}</div>
                    <div className="text-xs text-gray-600">{property.description}</div>
                  </div>
                </div>
                <Badge 
                  variant="secondary" 
                  className={cn(
                    "text-xs",
                    property.status === 'verified' && "bg-green-100 text-green-800",
                    property.status === 'failed' && "bg-red-100 text-red-800",
                    property.status === 'timeout' && "bg-yellow-100 text-yellow-800"
                  )}
                >
                  {property.status}
                </Badge>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 pt-2">
          {verification.proof && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={() => onViewProof?.(verification.proof!)}
                className="flex items-center gap-2"
              >
                <Eye className="h-4 w-4" />
                View Proof
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => onDownloadProof?.(verification.proof!)}
                className="flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Download
              </Button>
            </>
          )}
          
          {verification.counterexample && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onViewProof?.(verification.counterexample!)}
              className="flex items-center gap-2"
            >
              <AlertTriangle className="h-4 w-4" />
              View Counterexample
            </Button>
          )}
        </div>

        {/* Error Message */}
        {verification.error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center gap-2 text-red-800">
              <AlertTriangle className="h-4 w-4" />
              <span className="font-medium">Verification Error</span>
            </div>
            <p className="text-sm text-red-700 mt-1">{verification.error}</p>
          </div>
        )}

        {/* Timestamps */}
        <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t">
          <span>Created: {new Date(verification.createdAt).toLocaleString()}</span>
          {verification.completedAt && (
            <span>Completed: {new Date(verification.completedAt).toLocaleString()}</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

interface VerificationSummaryProps {
  verifications: VerificationResult[];
  className?: string;
}

export function VerificationSummary({ verifications, className }: VerificationSummaryProps) {
  const totalVerifications = verifications.length;
  const verifiedCount = verifications.filter(v => v.status === 'verified').length;
  const failedCount = verifications.filter(v => v.status === 'failed').length;
  const pendingCount = verifications.filter(v => v.status === 'pending').length;

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-blue-500" />
          <CardTitle className="text-lg">Verification Summary</CardTitle>
        </div>
        <CardDescription>
          Overview of formal verification results
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{verifiedCount}</div>
            <div className="text-sm text-gray-600">Verified</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{failedCount}</div>
            <div className="text-sm text-gray-600">Failed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{pendingCount}</div>
            <div className="text-sm text-gray-600">Pending</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{totalVerifications}</div>
            <div className="text-sm text-gray-600">Total</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 