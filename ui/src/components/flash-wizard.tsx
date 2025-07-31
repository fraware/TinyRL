'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { 
  Cpu, 
  Download, 
  Upload, 
  Wifi, 
  CheckCircle, 
  AlertCircle, 
  Clock,
  Terminal,
  Settings,
  HelpCircle,
  ExternalLink
} from 'lucide-react';

interface Device {
  id: string;
  name: string;
  type: 'stm32' | 'arduino' | 'esp32' | 'pico';
  port?: string;
  connected: boolean;
  firmware?: string;
}

interface Binary {
  id: string;
  name: string;
  size: number;
  target: string;
  reward: number;
  latency: number;
  powerConsumption: number;
}

interface FlashWizardProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function FlashWizard({ open, onOpenChange }: FlashWizardProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedDevice, setSelectedDevice] = useState<Device | null>(null);
  const [selectedBinary, setSelectedBinary] = useState<Binary | null>(null);
  const [powerBudget, setPowerBudget] = useState(50);
  const [isFlashing, setIsFlashing] = useState(false);
  const [flashProgress, setFlashProgress] = useState(0);
  const [consoleOutput, setConsoleOutput] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Mock data
  const availableDevices: Device[] = [
    { id: '1', name: 'STM32 Nucleo-144', type: 'stm32', port: 'COM3', connected: true, firmware: 'v1.2.3' },
    { id: '2', name: 'Arduino Nano 33 BLE', type: 'arduino', port: 'COM5', connected: true, firmware: 'v1.1.8' },
    { id: '3', name: 'ESP32 DevKit', type: 'esp32', port: 'COM7', connected: false, firmware: 'v1.0.5' },
    { id: '4', name: 'Raspberry Pi Pico', type: 'pico', port: 'COM9', connected: true, firmware: 'v1.3.1' },
  ];

  const availableBinaries: Binary[] = [
    { id: '1', name: 'CartPole PPO v1.0.0', size: 2048, target: 'cortex-m55', reward: 98.5, latency: 5.2, powerConsumption: 12.5 },
    { id: '2', name: 'LunarLander A2C v0.9.1', size: 4096, target: 'cortex-m55', reward: 85.3, latency: 8.7, powerConsumption: 18.2 },
    { id: '3', name: 'Acrobot DQN v1.2.0', size: 3072, target: 'cortex-m55', reward: 92.7, latency: 6.1, powerConsumption: 15.8 },
  ];

  const steps = [
    { id: 1, title: 'Device Detection', description: 'Connect and detect your device' },
    { id: 2, title: 'Binary Selection', description: 'Choose the model to flash' },
    { id: 3, title: 'Power Budget', description: 'Set power consumption limits' },
    { id: 4, title: 'Flash Device', description: 'Upload the binary to your device' },
  ];

  useEffect(() => {
    if (isFlashing) {
      const interval = setInterval(() => {
        setFlashProgress(prev => {
          if (prev >= 100) {
            setIsFlashing(false);
            return 100;
          }
          return prev + 10;
        });
      }, 500);

      return () => clearInterval(interval);
    }
  }, [isFlashing]);

  const detectDevices = () => {
    // Simulate device detection
    setConsoleOutput(prev => [...prev, '🔍 Scanning for connected devices...']);
    setTimeout(() => {
      setConsoleOutput(prev => [...prev, '✅ Found 3 connected devices']);
      setCurrentStep(2);
    }, 2000);
  };

  const startFlashing = () => {
    setIsFlashing(true);
    setFlashProgress(0);
    setConsoleOutput(prev => [...prev, '🚀 Starting flash process...']);
    
    // Simulate flash process
    setTimeout(() => {
      setConsoleOutput(prev => [...prev, '📡 Connecting to device...']);
    }, 1000);
    
    setTimeout(() => {
      setConsoleOutput(prev => [...prev, '📦 Uploading binary...']);
    }, 3000);
    
    setTimeout(() => {
      setConsoleOutput(prev => [...prev, '✅ Flash completed successfully!']);
      setIsFlashing(false);
    }, 8000);
  };

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case 'stm32':
        return <Cpu className="h-5 w-5 text-blue-500" />;
      case 'arduino':
        return <Cpu className="h-5 w-5 text-green-500" />;
      case 'esp32':
        return <Wifi className="h-5 w-5 text-orange-500" />;
      case 'pico':
        return <Cpu className="h-5 w-5 text-purple-500" />;
      default:
        return <Cpu className="h-5 w-5 text-gray-500" />;
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl">Flash Device Wizard</DialogTitle>
          <CardDescription>
            One-click deployment for your TinyRL models
          </CardDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Progress Steps */}
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center">
                <div className={`flex items-center justify-center w-8 h-8 rounded-full ${
                  currentStep >= step.id ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-500'
                }`}>
                  {currentStep > step.id ? <CheckCircle className="h-4 w-4" /> : step.id}
                </div>
                <div className="ml-2">
                  <p className="text-sm font-medium">{step.title}</p>
                  <p className="text-xs text-gray-500">{step.description}</p>
                </div>
                {index < steps.length - 1 && (
                  <div className={`w-16 h-0.5 mx-4 ${
                    currentStep > step.id ? 'bg-blue-500' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>

          {/* Step 1: Device Detection */}
          {currentStep === 1 && (
            <Card>
              <CardHeader>
                <CardTitle>Device Detection</CardTitle>
                <CardDescription>
                  Connect your device and we'll automatically detect it
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {availableDevices.map((device) => (
                      <div
                        key={device.id}
                        className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                          selectedDevice?.id === device.id
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                        onClick={() => setSelectedDevice(device)}
                      >
                        <div className="flex items-center space-x-3">
                          {getDeviceIcon(device.type)}
                          <div className="flex-1">
                            <p className="font-medium">{device.name}</p>
                            <p className="text-sm text-gray-500">
                              {device.connected ? 'Connected' : 'Not connected'} • {device.port}
                            </p>
                          </div>
                          {device.connected ? (
                            <CheckCircle className="h-5 w-5 text-green-500" />
                          ) : (
                            <AlertCircle className="h-5 w-5 text-red-500" />
                          )}
                        </div>
                        {device.firmware && (
                          <p className="text-xs text-gray-500 mt-2">
                            Firmware: {device.firmware}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button onClick={detectDevices}>
                      <Cpu className="h-4 w-4 mr-2" />
                      Detect Devices
                    </Button>
                    <Button variant="outline">
                      <HelpCircle className="h-4 w-4 mr-2" />
                      Troubleshoot
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Step 2: Binary Selection */}
          {currentStep === 2 && (
            <Card>
              <CardHeader>
                <CardTitle>Binary Selection</CardTitle>
                <CardDescription>
                  Choose the model binary to flash to your device
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-4">
                    {availableBinaries.map((binary) => (
                      <div
                        key={binary.id}
                        className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                          selectedBinary?.id === binary.id
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                        onClick={() => setSelectedBinary(binary)}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-medium">{binary.name}</p>
                            <p className="text-sm text-gray-500">
                              Target: {binary.target} • Size: {formatBytes(binary.size)}
                            </p>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium">{binary.reward}% reward</p>
                            <p className="text-xs text-gray-500">
                              {binary.latency}ms • {binary.powerConsumption}mW
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button onClick={() => setCurrentStep(3)} disabled={!selectedBinary}>
                      Continue
                    </Button>
                    <Button variant="outline" onClick={() => setCurrentStep(1)}>
                      Back
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Step 3: Power Budget */}
          {currentStep === 3 && (
            <Card>
              <CardHeader>
                <CardTitle>Power Budget</CardTitle>
                <CardDescription>
                  Set power consumption limits for your device
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Power Budget (mW)</label>
                    <div className="flex items-center space-x-4 mt-2">
                      <input
                        type="range"
                        min="10"
                        max="100"
                        value={powerBudget}
                        onChange={(e) => setPowerBudget(Number(e.target.value))}
                        className="flex-1"
                      />
                      <span className="text-sm font-medium">{powerBudget}mW</span>
                    </div>
                  </div>
                  
                  {selectedBinary && (
                    <div className="p-4 bg-gray-50 rounded-lg">
                      <p className="text-sm font-medium mb-2">Power Analysis</p>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-gray-500">Model Power</p>
                          <p className="font-medium">{selectedBinary.powerConsumption}mW</p>
                        </div>
                        <div>
                          <p className="text-gray-500">Available Budget</p>
                          <p className="font-medium">{powerBudget}mW</p>
                        </div>
                        <div>
                          <p className="text-gray-500">Margin</p>
                          <p className={`font-medium ${
                            powerBudget - selectedBinary.powerConsumption > 10 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {powerBudget - selectedBinary.powerConsumption}mW
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div className="flex space-x-2">
                    <Button onClick={() => setCurrentStep(4)}>
                      Continue
                    </Button>
                    <Button variant="outline" onClick={() => setCurrentStep(2)}>
                      Back
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Step 4: Flash Device */}
          {currentStep === 4 && (
            <Card>
              <CardHeader>
                <CardTitle>Flash Device</CardTitle>
                <CardDescription>
                  Upload the binary to your device
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Flash Progress */}
                  {isFlashing && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Flashing Progress</span>
                        <span className="text-sm text-gray-500">{flashProgress}%</span>
                      </div>
                      <Progress value={flashProgress} className="w-full" />
                    </div>
                  )}

                  {/* Console Output */}
                  <div className="bg-black text-green-400 p-4 rounded-lg font-mono text-sm h-48 overflow-y-auto">
                    {consoleOutput.map((line, index) => (
                      <div key={index} className="mb-1">
                        <span className="text-gray-500">$</span> {line}
                      </div>
                    ))}
                  </div>

                  {/* Error Display */}
                  {error && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <AlertCircle className="h-5 w-5 text-red-500" />
                        <span className="text-sm font-medium text-red-800">Error</span>
                      </div>
                      <p className="text-sm text-red-700 mt-1">{error}</p>
                    </div>
                  )}

                  <div className="flex space-x-2">
                    {!isFlashing ? (
                      <>
                        <Button onClick={startFlashing} disabled={!selectedDevice || !selectedBinary}>
                          <Upload className="h-4 w-4 mr-2" />
                          Start Flash
                        </Button>
                        <Button variant="outline" onClick={() => setCurrentStep(3)}>
                          Back
                        </Button>
                      </>
                    ) : (
                      <Button variant="outline" disabled>
                        <Clock className="h-4 w-4 mr-2" />
                        Flashing...
                      </Button>
                    )}
                  </div>

                  {/* Troubleshooting */}
                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-sm font-medium text-blue-800 mb-2">Troubleshooting Tips</p>
                    <ul className="text-sm text-blue-700 space-y-1">
                      <li>• Ensure your device is connected and recognized</li>
                      <li>• Check that the correct port is selected</li>
                      <li>• Make sure your device is in bootloader mode if needed</li>
                      <li>• Try pressing the reset button on your device</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
} 