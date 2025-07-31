#!/usr/bin/env python3
"""
Monitoring CLI

Command-line interface for TinyRL monitoring and OTA updates.
"""

import argparse
import logging
import sys

from tinyrl.monitoring import MonitoringConfig, run_monitoring_pipeline


def create_monitoring_config(args) -> MonitoringConfig:
    """Create monitoring configuration from CLI arguments."""
    return MonitoringConfig(
        telemetry_enabled=not args.disable_telemetry,
        telemetry_endpoint=args.telemetry_endpoint,
        telemetry_interval_seconds=args.telemetry_interval,
        telemetry_batch_size=args.telemetry_batch_size,
        reward_drift_threshold=args.reward_drift_threshold,
        reward_drift_window_days=args.reward_drift_window,
        anomaly_detection_enabled=not args.disable_anomaly_detection,
        ota_enabled=not args.disable_ota,
        ota_endpoint=args.ota_endpoint,
        ota_check_interval_hours=args.ota_check_interval,
        ota_rollback_threshold=args.ota_rollback_threshold,
        alert_webhook_url=args.alert_webhook,
        alert_email=args.alert_email,
        alert_slack_channel=args.alert_slack,
        telemetry_db_path=args.telemetry_db_path,
        model_cache_path=args.model_cache_path,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="TinyRL Monitoring and OTA Update CLI")

    # Telemetry settings
    parser.add_argument(
        "--disable-telemetry", action="store_true", help="Disable telemetry collection"
    )
    parser.add_argument(
        "--telemetry-endpoint",
        default="https://api.tinyrl.dev/telemetry",
        help="Telemetry endpoint URL",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=int,
        default=3600,
        help="Telemetry transmission interval in seconds",
    )
    parser.add_argument(
        "--telemetry-batch-size", type=int, default=100, help="Telemetry batch size"
    )

    # Drift detection
    parser.add_argument(
        "--reward-drift-threshold",
        type=float,
        default=0.05,
        help="Reward drift threshold (default: 0.05)",
    )
    parser.add_argument(
        "--reward-drift-window", type=int, default=7, help="Reward drift window in days"
    )
    parser.add_argument(
        "--disable-anomaly-detection",
        action="store_true",
        help="Disable anomaly detection",
    )

    # OTA settings
    parser.add_argument(
        "--disable-ota", action="store_true", help="Disable OTA updates"
    )
    parser.add_argument(
        "--ota-endpoint", default="https://api.tinyrl.dev/ota", help="OTA endpoint URL"
    )
    parser.add_argument(
        "--ota-check-interval", type=int, default=24, help="OTA check interval in hours"
    )
    parser.add_argument(
        "--ota-rollback-threshold",
        type=float,
        default=0.1,
        help="OTA rollback threshold (default: 0.1)",
    )

    # Alerting
    parser.add_argument("--alert-webhook", help="Alert webhook URL")
    parser.add_argument("--alert-email", help="Alert email address")
    parser.add_argument("--alert-slack", help="Alert Slack channel")

    # Storage
    parser.add_argument(
        "--telemetry-db-path",
        default="./data/telemetry.db",
        help="Telemetry database path",
    )
    parser.add_argument(
        "--model-cache-path", default="./cache/models", help="Model cache path"
    )

    # Device info
    parser.add_argument("--device-id", required=True, help="Device identifier")
    parser.add_argument("--model-version", required=True, help="Current model version")

    # Output
    parser.add_argument(
        "--output-dir", default="./outputs/monitoring", help="Output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Create configuration
        config = create_monitoring_config(args)

        if args.verbose:
            print("TinyRL Monitoring Configuration:")
            print(f"  Telemetry enabled: {config.telemetry_enabled}")
            print(f"  OTA enabled: {config.ota_enabled}")
            print(f"  Device ID: {args.device_id}")
            print(f"  Model version: {args.model_version}")

        # Run monitoring pipeline
        results = run_monitoring_pipeline(
            config=config,
            device_id=args.device_id,
            model_version=args.model_version,
            output_dir=args.output_dir,
        )

        # Print results
        if args.verbose:
            print("\nMonitoring Results:")
            print(f"  Status: {results['report']['status']}")
            metrics = results["dashboard_metrics"]
            print(f"  Telemetry buffer: {metrics['telemetry_buffer_size']}")
            print(f"  Drift alerts: {metrics['reward_drift_alerts']}")
            print(f"  OTA updates: {metrics['ota_updates']}")
            print(f"  Rollback count: {metrics['rollback_count']}")

        report_path = f"{args.output_dir}/monitoring_report.json"
        print(f"Monitoring report saved to: {report_path}")

    except Exception as e:
        print(f"Error running monitoring pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
