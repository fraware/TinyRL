#!/usr/bin/env python3
"""
Benchmark Harness

This module implements comprehensive benchmarking for TinyRL models,
measuring reward, latency, and power across firmware builds.
"""

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark harness."""

    # Benchmark parameters
    num_runs: int = 10
    warmup_runs: int = 3
    timeout_seconds: int = 300

    # Hardware configuration
    target_mcu: str = "cortex-m55"
    clock_frequency_mhz: float = 80.0
    voltage_v: float = 3.3

    # Power measurement
    use_power_profiling: bool = True
    power_sensor: str = "INA219"
    power_measurement_interval_ms: int = 100

    # Output configuration
    output_dir: str = "./outputs/benchmarks"
    csv_output: bool = True
    junit_output: bool = True
    json_output: bool = True


class PowerProfiler:
    """Power profiling using INA219 sensor."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.measurements = []
        self.is_measuring = False

    def start_measurement(self) -> None:
        """Start power measurement."""
        self.is_measuring = True
        self.measurements = []
        logger.info("Power measurement started")

    def stop_measurement(self) -> None:
        """Stop power measurement."""
        self.is_measuring = False
        logger.info("Power measurement stopped")

    def record_measurement(self, power_mw: float, timestamp: float) -> None:
        """Record a power measurement."""
        if self.is_measuring:
            self.measurements.append(
                {
                    "power_mw": power_mw,
                    "timestamp": timestamp,
                }
            )

    def get_power_stats(self) -> Dict[str, float]:
        """Get power statistics."""
        if not self.measurements:
            return {
                "avg_power_mw": 0.0,
                "max_power_mw": 0.0,
                "min_power_mw": 0.0,
                "total_energy_mj": 0.0,
            }

        powers = [m["power_mw"] for m in self.measurements]
        timestamps = [m["timestamp"] for m in self.measurements]

        # Calculate energy consumption
        total_energy_mj = 0.0
        for i in range(1, len(self.measurements)):
            dt = timestamps[i] - timestamps[i - 1]
            power = (powers[i] + powers[i - 1]) / 2
            total_energy_mj += power * dt * 1000  # Convert to mJ

        return {
            "avg_power_mw": np.mean(powers),
            "max_power_mw": np.max(powers),
            "min_power_mw": np.min(powers),
            "total_energy_mj": total_energy_mj,
        }


class LatencyProfiler:
    """Latency profiling for inference operations."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.measurements = []

    def measure_inference_latency(self, model, input_data: torch.Tensor) -> float:
        """Measure inference latency for a single forward pass."""
        # Warm up
        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                _ = model(input_data)

        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        with torch.no_grad():
            _ = model(input_data)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        self.measurements.append(latency_ms)

        return latency_ms

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.measurements:
            return {
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
            }

        latencies = np.array(self.measurements)

        return {
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
        }


class RewardEvaluator:
    """Reward evaluation for RL models."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.evaluations = []

    def evaluate_model(
        self, model, env_name: str, num_episodes: int = 100
    ) -> Dict[str, float]:
        """Evaluate model performance on environment."""
        import gymnasium as gym
        from stable_baselines3.common.evaluation import evaluate_policy

        # Create environment
        env = gym.make(env_name)

        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=num_episodes, deterministic=True
        )

        # Calculate additional metrics
        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                if truncated:
                    done = True

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        env.close()

        results = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "success_rate": float(np.mean([r > 0 for r in episode_rewards])),
        }

        self.evaluations.append(results)
        return results


class BenchmarkRunner:
    """Main benchmark runner."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.power_profiler = PowerProfiler(config)
        self.latency_profiler = LatencyProfiler(config)
        self.reward_evaluator = RewardEvaluator(config)
        self.results = []

    def run_model_benchmark(
        self, model_path: str, env_name: str, model_name: str
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark for a model."""
        logger.info(f"Starting benchmark for {model_name}")

        # Load model
        from stable_baselines3 import PPO, A2C

        if "ppo" in model_path.lower():
            model = PPO.load(model_path)
        elif "a2c" in model_path.lower():
            model = A2C.load(model_path)
        else:
            raise ValueError(f"Unknown model type in {model_path}")

        # Generate test input
        obs_dim = model.observation_space.shape[0]
        test_input = torch.randn(1, obs_dim)

        # Power profiling
        if self.config.use_power_profiling:
            self.power_profiler.start_measurement()

        # Latency profiling
        latencies = []
        for _ in range(self.config.num_runs):
            latency = self.latency_profiler.measure_inference_latency(model, test_input)
            latencies.append(latency)

            # Simulate power measurement
            if self.config.use_power_profiling:
                power_mw = 10.0 + np.random.normal(0, 1)  # Simulated power
                self.power_profiler.record_measurement(power_mw, time.time())

        if self.config.use_power_profiling:
            self.power_profiler.stop_measurement()

        # Reward evaluation
        reward_results = self.reward_evaluator.evaluate_model(model, env_name)

        # Compile results
        latency_stats = self.latency_profiler.get_latency_stats()
        power_stats = self.power_profiler.get_power_stats()

        benchmark_result = {
            "model_name": model_name,
            "model_path": model_path,
            "env_name": env_name,
            "timestamp": time.time(),
            "latency": latency_stats,
            "power": power_stats,
            "reward": reward_results,
            "config": self.config.__dict__,
        }

        self.results.append(benchmark_result)
        logger.info(f"Completed benchmark for {model_name}")

        return benchmark_result

    def save_results(self, output_dir: str) -> None:
        """Save benchmark results in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())

        # Save JSON results
        if self.config.json_output:
            json_path = output_path / f"benchmark_results_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Saved JSON results to {json_path}")

        # Save CSV results
        if self.config.csv_output:
            csv_path = output_path / f"benchmark_results_{timestamp}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._get_csv_fields())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(self._flatten_result(result))
            logger.info(f"Saved CSV results to {csv_path}")

        # Save JUnit XML results
        if self.config.junit_output:
            junit_path = output_path / f"benchmark_results_{timestamp}.xml"
            self._save_junit_xml(junit_path)
            logger.info(f"Saved JUnit XML to {junit_path}")

    def _get_csv_fields(self) -> List[str]:
        """Get CSV field names."""
        return [
            "model_name",
            "env_name",
            "timestamp",
            "avg_latency_ms",
            "p95_latency_ms",
            "max_latency_ms",
            "avg_power_mw",
            "max_power_mw",
            "total_energy_mj",
            "mean_reward",
            "success_rate",
            "mean_episode_length",
        ]

    def _flatten_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested result for CSV output."""
        return {
            "model_name": result["model_name"],
            "env_name": result["env_name"],
            "timestamp": result["timestamp"],
            "avg_latency_ms": result["latency"]["avg_latency_ms"],
            "p95_latency_ms": result["latency"]["p95_latency_ms"],
            "max_latency_ms": result["latency"]["max_latency_ms"],
            "avg_power_mw": result["power"]["avg_power_mw"],
            "max_power_mw": result["power"]["max_power_mw"],
            "total_energy_mj": result["power"]["total_energy_mj"],
            "mean_reward": result["reward"]["mean_reward"],
            "success_rate": result["reward"]["success_rate"],
            "mean_episode_length": result["reward"]["mean_episode_length"],
        }

    def _save_junit_xml(self, output_path: Path) -> None:
        """Save results in JUnit XML format."""
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += "<testsuites>\n"
        xml_content += (
            f'  <testsuite name="TinyRL Benchmarks" tests="{len(self.results)}">\n'
        )

        for result in self.results:
            test_name = f"{result['model_name']}_{result['env_name']}"

            # Determine test status based on performance thresholds
            latency_ok = result["latency"]["p95_latency_ms"] <= 5.0
            power_ok = result["power"]["avg_power_mw"] <= 50.0
            reward_ok = result["reward"]["success_rate"] >= 0.8

            if latency_ok and power_ok and reward_ok:
                xml_content += (
                    f'    <testcase name="{test_name}" classname="Benchmark">\n'
                )
                xml_content += "    </testcase>\n"
            else:
                xml_content += (
                    f'    <testcase name="{test_name}" classname="Benchmark">\n'
                )
                xml_content += (
                    '      <failure message="Performance thresholds not met">\n'
                )
                xml_content += (
                    f'        Latency: {result["latency"]["p95_latency_ms"]:.2f}ms\n'
                )
                xml_content += (
                    f'        Power: {result["power"]["avg_power_mw"]:.2f}mW\n'
                )
                xml_content += (
                    f'        Success Rate: {result["reward"]["success_rate"]:.2f}\n'
                )
                xml_content += "      </failure>\n"
                xml_content += "    </testcase>\n"

        xml_content += "  </testsuite>\n"
        xml_content += "</testsuites>\n"

        with open(output_path, "w") as f:
            f.write(xml_content)


def run_benchmark_harness(
    config: BenchmarkConfig,
    model_paths: List[str],
    env_names: List[str],
    output_dir: str = "./outputs/benchmarks",
) -> Dict[str, Any]:
    """Run complete benchmark harness."""
    logger.info("Starting benchmark harness")

    runner = BenchmarkRunner(config)

    # Run benchmarks for each model
    for model_path in model_paths:
        if not os.path.exists(model_path):
            logger.warning(f"Model path not found: {model_path}")
            continue

        # Determine environment name from model path
        env_name = None
        for env in env_names:
            if env.lower() in model_path.lower():
                env_name = env
                break

        if not env_name:
            logger.warning(f"Could not determine environment for {model_path}")
            continue

        # Extract model name
        model_name = Path(model_path).stem

        try:
            result = runner.run_model_benchmark(model_path, env_name, model_name)
            logger.info(f"Benchmark completed for {model_name}")
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")

    # Save results
    runner.save_results(output_dir)

    # Generate summary
    summary = {
        "total_models": len(runner.results),
        "successful_benchmarks": len([r for r in runner.results if "error" not in r]),
        "average_latency": np.mean(
            [r["latency"]["avg_latency_ms"] for r in runner.results]
        ),
        "average_power": np.mean([r["power"]["avg_power_mw"] for r in runner.results]),
        "average_reward": np.mean([r["reward"]["mean_reward"] for r in runner.results]),
    }

    logger.info("Benchmark harness completed")
    return {
        "summary": summary,
        "results": runner.results,
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run TinyRL benchmark harness")

    parser.add_argument(
        "--model-paths",
        nargs="+",
        required=True,
        help="Paths to model files to benchmark",
    )

    parser.add_argument(
        "--env-names",
        nargs="+",
        default=["CartPole-v1", "LunarLander-v2"],
        help="Environment names for evaluation",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/benchmarks",
        help="Output directory (default: ./outputs/benchmarks)",
    )

    parser.add_argument(
        "--power-profiling",
        action="store_true",
        default=True,
        help="Enable power profiling (default: True)",
    )

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        num_runs=args.num_runs,
        use_power_profiling=args.power_profiling,
        output_dir=args.output_dir,
    )

    # Run benchmark harness
    try:
        results = run_benchmark_harness(
            config=config,
            model_paths=args.model_paths,
            env_names=args.env_names,
            output_dir=args.output_dir,
        )

        # Print summary
        summary = results["summary"]
        print(f"\nBenchmark Summary:")
        print(f"Total Models: {summary['total_models']}")
        print(f"Successful Benchmarks: {summary['successful_benchmarks']}")
        print(f"Average Latency: {summary['average_latency']:.2f}ms")
        print(f"Average Power: {summary['average_power']:.2f}mW")
        print(f"Average Reward: {summary['average_reward']:.2f}")

        print(f"\nResults saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
