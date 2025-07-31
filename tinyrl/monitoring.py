#!/usr/bin/env python3
"""
Post-Release Monitoring & Update Path Module

This module implements monitoring, telemetry, and OTA update mechanisms
for TinyRL deployments in production environments.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for post-release monitoring."""

    # Telemetry settings
    telemetry_enabled: bool = True
    telemetry_endpoint: str = "https://api.tinyrl.dev/telemetry"
    telemetry_interval_seconds: int = 3600  # 1 hour
    telemetry_batch_size: int = 100

    # Reward drift detection
    reward_drift_threshold: float = 0.05  # 5% drift
    reward_drift_window_days: int = 7
    anomaly_detection_enabled: bool = True

    # OTA update settings
    ota_enabled: bool = True
    ota_endpoint: str = "https://api.tinyrl.dev/ota"
    ota_check_interval_hours: int = 24
    ota_rollback_threshold: float = 0.1  # 10% performance drop

    # Alerting
    alert_webhook_url: Optional[str] = None
    alert_email: Optional[str] = None
    alert_slack_channel: Optional[str] = None

    # Storage
    telemetry_db_path: str = "./data/telemetry.db"
    model_cache_path: str = "./cache/models"


@dataclass
class TelemetryData:
    """Telemetry data structure."""

    device_id: str
    model_version: str
    timestamp: datetime
    reward: float
    latency_ms: float
    power_mw: float
    memory_usage_kb: int
    inference_count: int
    error_count: int = 0
    environment_data: Dict[str, Any] = field(default_factory=dict)


class RewardDriftDetector:
    """Detect reward drift in deployed models."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.baseline_reward = None
        self.reward_history = []
        self.drift_alerts = []

    def set_baseline(self, baseline_reward: float) -> None:
        """Set baseline reward for drift detection."""
        self.baseline_reward = baseline_reward
        logger.info(f"Baseline reward set to: {baseline_reward}")

    def add_reward_measurement(self, reward: float, timestamp: datetime) -> None:
        """Add a new reward measurement."""
        self.reward_history.append({"reward": reward, "timestamp": timestamp})

        # Keep only recent history
        cutoff = timestamp - timedelta(days=self.config.reward_drift_window_days)
        self.reward_history = [
            entry for entry in self.reward_history if entry["timestamp"] > cutoff
        ]

    def detect_drift(self) -> Optional[Dict[str, Any]]:
        """Detect reward drift and return alert if detected."""
        if not self.baseline_reward or len(self.reward_history) < 10:
            return None

        recent_rewards = [entry["reward"] for entry in self.reward_history[-10:]]
        current_avg = np.mean(recent_rewards)

        drift_ratio = abs(current_avg - self.baseline_reward) / self.baseline_reward

        if drift_ratio > self.config.reward_drift_threshold:
            device_ids = set(
                entry.get("device_id", "unknown") for entry in self.reward_history
            )
            alert = {
                "type": "reward_drift",
                "severity": "high" if drift_ratio > 0.1 else "medium",
                "baseline_reward": self.baseline_reward,
                "current_avg_reward": current_avg,
                "drift_ratio": drift_ratio,
                "timestamp": datetime.now(),
                "device_count": len(device_ids),
            }

            self.drift_alerts.append(alert)
            return alert

        return None


class AnomalyDetector:
    """Detect anomalies in model performance."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.latency_history = []
        self.power_history = []
        self.error_rate_history = []

    def add_measurement(
        self, latency_ms: float, power_mw: float, error_count: int, total_count: int
    ) -> None:
        """Add performance measurement."""
        error_rate = error_count / max(total_count, 1)

        self.latency_history.append(latency_ms)
        self.power_history.append(power_mw)
        self.error_rate_history.append(error_rate)

        # Keep only recent history
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-500:]
            self.power_history = self.power_history[-500:]
            self.error_rate_history = self.error_rate_history[-500:]

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        anomalies = []

        if len(self.latency_history) < 10:
            return anomalies

        # Latency anomaly detection
        recent_latency = np.array(self.latency_history[-10:])
        latency_mean = np.mean(recent_latency)
        latency_std = np.std(recent_latency)

        if latency_std > 0:
            latency_z_scores = np.abs((recent_latency - latency_mean) / latency_std)
            if np.any(latency_z_scores > 3):  # 3-sigma rule
                anomalies.append(
                    {
                        "type": "high_latency",
                        "severity": "medium",
                        "current_latency": float(recent_latency[-1]),
                        "expected_range": f"{latency_mean - 2*latency_std:.2f}-{latency_mean + 2*latency_std:.2f}ms",
                        "timestamp": datetime.now(),
                    }
                )

        # Power anomaly detection
        recent_power = np.array(self.power_history[-10:])
        power_mean = np.mean(recent_power)
        power_std = np.std(recent_power)

        if power_std > 0:
            power_z_scores = np.abs((recent_power - power_mean) / power_std)
            if np.any(power_z_scores > 3):
                anomalies.append(
                    {
                        "type": "high_power_consumption",
                        "severity": "medium",
                        "current_power": float(recent_power[-1]),
                        "expected_range": f"{power_mean - 2*power_std:.2f}-{power_mean + 2*power_std:.2f}mW",
                        "timestamp": datetime.now(),
                    }
                )

        # Error rate anomaly detection
        recent_error_rate = np.array(self.error_rate_history[-10:])
        error_rate_mean = np.mean(recent_error_rate)

        if error_rate_mean > 0.05:  # 5% error rate threshold
            anomalies.append(
                {
                    "type": "high_error_rate",
                    "severity": "high",
                    "current_error_rate": float(error_rate_mean),
                    "threshold": 0.05,
                    "timestamp": datetime.now(),
                }
            )

        return anomalies


class OTAManager:
    """Manage Over-The-Air updates for TinyRL models."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.current_model_version = None
        self.update_history = []
        self.rollback_count = 0

    def check_for_updates(self, current_version: str) -> Optional[Dict[str, Any]]:
        """Check for available model updates."""
        try:
            response = requests.get(
                f"{self.config.ota_endpoint}/updates",
                params={
                    "current_version": current_version,
                    "device_type": "cortex-m55",
                },
                timeout=30,
            )

            if response.status_code == 200:
                update_info = response.json()
                if update_info.get("available"):
                    return update_info

        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")

        return None

    def download_model(self, model_url: str, version: str) -> Optional[str]:
        """Download new model version."""
        try:
            cache_path = Path(self.config.model_cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)

            model_path = cache_path / f"model_{version}.bin"

            response = requests.get(model_url, stream=True, timeout=300)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify model integrity
            if self._verify_model(model_path):
                return str(model_path)
            else:
                logger.error("Model verification failed")
                model_path.unlink(missing_ok=True)
                return None

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None

    def _verify_model(self, model_path: Path) -> bool:
        """Verify downloaded model integrity."""
        try:
            # Check file size
            if model_path.stat().st_size < 1024:  # At least 1KB
                return False

            # Check file header (simplified)
            with open(model_path, "rb") as f:
                header = f.read(16)
                if not header.startswith(b"TINYRL"):
                    return False

            return True

        except Exception:
            return False

    def deploy_model(self, model_path: str, version: str) -> bool:
        """Deploy new model version."""
        try:
            # Simulate model deployment
            logger.info(f"Deploying model version {version}")

            # Record deployment
            self.update_history.append(
                {
                    "version": version,
                    "deployment_time": datetime.now(),
                    "model_path": model_path,
                    "status": "deployed",
                }
            )

            self.current_model_version = version
            return True

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False

    def rollback_if_needed(self, performance_metrics: Dict[str, float]) -> bool:
        """Rollback if performance degradation detected."""
        if not self.update_history:
            return False

        current_reward = performance_metrics.get("reward", 0)
        baseline_reward = performance_metrics.get("baseline_reward", current_reward)

        if baseline_reward > 0:
            performance_drop = (baseline_reward - current_reward) / baseline_reward

            if performance_drop > self.config.ota_rollback_threshold:
                logger.warning(f"Performance drop detected: {performance_drop:.2%}")
                self.rollback_count += 1

                # Rollback to previous version
                if len(self.update_history) > 1:
                    previous_version = self.update_history[-2]["version"]
                    logger.info(f"Rolling back to version {previous_version}")

                    self.update_history.append(
                        {
                            "version": previous_version,
                            "deployment_time": datetime.now(),
                            "status": "rollback",
                            "reason": f"performance_drop_{performance_drop:.2%}",
                        }
                    )

                    return True

        return False


class TelemetryCollector:
    """Collect and transmit telemetry data."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.telemetry_buffer = []
        self.last_transmission = datetime.now()

    def add_telemetry(self, telemetry_data: TelemetryData) -> None:
        """Add telemetry data to buffer."""
        self.telemetry_buffer.append(telemetry_data)

        # Transmit if buffer is full or enough time has passed
        if (
            len(self.telemetry_buffer) >= self.config.telemetry_batch_size
            or (datetime.now() - self.last_transmission).seconds
            >= self.config.telemetry_interval_seconds
        ):
            self.transmit_telemetry()

    def transmit_telemetry(self) -> bool:
        """Transmit buffered telemetry data."""
        if not self.telemetry_buffer:
            return True

        try:
            # Convert to JSON-serializable format
            telemetry_json = []
            for data in self.telemetry_buffer:
                telemetry_json.append(
                    {
                        "device_id": data.device_id,
                        "model_version": data.model_version,
                        "timestamp": data.timestamp.isoformat(),
                        "reward": data.reward,
                        "latency_ms": data.latency_ms,
                        "power_mw": data.power_mw,
                        "memory_usage_kb": data.memory_usage_kb,
                        "inference_count": data.inference_count,
                        "error_count": data.error_count,
                        "environment_data": data.environment_data,
                    }
                )

            response = requests.post(
                self.config.telemetry_endpoint,
                json={
                    "telemetry": telemetry_json,
                    "batch_size": len(telemetry_json),
                    "transmission_time": datetime.now().isoformat(),
                },
                timeout=30,
            )

            if response.status_code == 200:
                logger.info(f"Transmitted {len(telemetry_json)} telemetry records")
                self.telemetry_buffer.clear()
                self.last_transmission = datetime.now()
                return True
            else:
                logger.error(f"Telemetry transmission failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Telemetry transmission error: {e}")
            return False


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = []

    def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert through configured channels."""
        self.alert_history.append({**alert_data, "timestamp": datetime.now()})

        success = True

        # Send webhook alert
        if self.config.alert_webhook_url:
            success &= self._send_webhook_alert(alert_data)

        # Send email alert
        if self.config.alert_email:
            success &= self._send_email_alert(alert_data)

        # Send Slack alert
        if self.config.alert_slack_channel:
            success &= self._send_slack_alert(alert_data)

        return success

    def _send_webhook_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert via webhook."""
        try:
            response = requests.post(
                self.config.alert_webhook_url, json=alert_data, timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
            return False

    def _send_email_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert via email."""
        # Simplified email implementation
        logger.info(f"Email alert: {alert_data}")
        return True

    def _send_slack_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert via Slack."""
        # Simplified Slack implementation
        logger.info(f"Slack alert: {alert_data}")
        return True


class MonitoringDashboard:
    """Dashboard for monitoring TinyRL deployments."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.telemetry_collector = TelemetryCollector(config)
        self.reward_drift_detector = RewardDriftDetector(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.ota_manager = OTAManager(config)
        self.alert_manager = AlertManager(config)

    def process_telemetry(self, telemetry_data: TelemetryData) -> None:
        """Process incoming telemetry data."""
        # Add to telemetry collector
        self.telemetry_collector.add_telemetry(telemetry_data)

        # Update drift detector
        self.reward_drift_detector.add_reward_measurement(
            telemetry_data.reward, telemetry_data.timestamp
        )

        # Update anomaly detector
        self.anomaly_detector.add_measurement(
            telemetry_data.latency_ms,
            telemetry_data.power_mw,
            telemetry_data.error_count,
            telemetry_data.inference_count,
        )

        # Check for drift
        drift_alert = self.reward_drift_detector.detect_drift()
        if drift_alert:
            self.alert_manager.send_alert(drift_alert)

        # Check for anomalies
        anomalies = self.anomaly_detector.detect_anomalies()
        for anomaly in anomalies:
            self.alert_manager.send_alert(anomaly)

    def check_for_updates(self, current_version: str) -> None:
        """Check for model updates."""
        update_info = self.ota_manager.check_for_updates(current_version)

        if update_info:
            logger.info(f"Update available: {update_info['version']}")

            # Download and deploy new model
            model_path = self.ota_manager.download_model(
                update_info["download_url"], update_info["version"]
            )

            if model_path:
                success = self.ota_manager.deploy_model(
                    model_path, update_info["version"]
                )
                if success:
                    logger.info(
                        f"Successfully deployed version {update_info['version']}"
                    )
                else:
                    logger.error("Model deployment failed")

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get current dashboard metrics."""
        return {
            "telemetry_buffer_size": len(self.telemetry_collector.telemetry_buffer),
            "reward_drift_alerts": len(self.reward_drift_detector.drift_alerts),
            "anomaly_count": len(self.anomaly_detector.latency_history),
            "ota_updates": len(self.ota_manager.update_history),
            "rollback_count": self.ota_manager.rollback_count,
            "alert_count": len(self.alert_manager.alert_history),
            "current_model_version": self.ota_manager.current_model_version,
        }


def create_monitoring_report(
    dashboard: MonitoringDashboard, config: MonitoringConfig
) -> Dict[str, Any]:
    """Create monitoring report."""
    metrics = dashboard.get_dashboard_metrics()

    return {
        "status": "ACTIVE",
        "telemetry_enabled": config.telemetry_enabled,
        "ota_enabled": config.ota_enabled,
        "metrics": metrics,
        "config": {
            "reward_drift_threshold": config.reward_drift_threshold,
            "telemetry_interval": config.telemetry_interval_seconds,
            "ota_check_interval": config.ota_check_interval_hours,
        },
        "timestamp": datetime.now().isoformat(),
    }


def run_monitoring_pipeline(
    config: MonitoringConfig,
    device_id: str,
    model_version: str,
    output_dir: str = "./outputs/monitoring",
) -> Dict[str, Any]:
    """Run complete monitoring pipeline."""
    logger.info("Starting monitoring pipeline")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize dashboard
    dashboard = MonitoringDashboard(config)

    # Set baseline for drift detection (example)
    dashboard.reward_drift_detector.set_baseline(100.0)

    # Simulate telemetry data
    telemetry_data = TelemetryData(
        device_id=device_id,
        model_version=model_version,
        timestamp=datetime.now(),
        reward=95.0,  # Slight drift from baseline
        latency_ms=4.2,
        power_mw=8.5,
        memory_usage_kb=28,
        inference_count=1000,
        error_count=2,
        environment_data={"temperature": 25.0, "humidity": 60.0},
    )

    # Process telemetry
    dashboard.process_telemetry(telemetry_data)

    # Check for updates
    dashboard.check_for_updates(model_version)

    # Create report
    report = create_monitoring_report(dashboard, config)

    # Save report
    with open(output_path / "monitoring_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Monitoring pipeline completed")
    return {"report": report, "dashboard_metrics": dashboard.get_dashboard_metrics()}
