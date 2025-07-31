#!/usr/bin/env python3
"""
RAM-Aware Dispatcher CLI Script

Run RAM-aware dispatcher benchmark with command-line interface.
"""

import argparse
import json
import sys

from tinyrl.dispatcher import DispatcherConfig, run_dispatcher_benchmark


def create_dispatcher_config(args) -> DispatcherConfig:
    """Create dispatcher configuration from command line arguments."""
    return DispatcherConfig(
        max_stack_size=args.max_stack_size,
        max_heap_size=args.max_heap_size,
        buffer_size=args.buffer_size,
        dma_channel=args.dma_channel,
        dma_priority=args.dma_priority,
        dma_burst_size=args.dma_burst_size,
        max_isr_latency_us=args.max_isr_latency_us,
        target_throughput_mbps=args.target_throughput_mbps,
        flash_sector_size=args.flash_sector_size,
        flash_read_latency_ns=args.flash_read_latency_ns,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run RAM-aware dispatcher benchmark")

    # Memory constraints
    parser.add_argument(
        "--max-stack-size",
        type=int,
        default=4096,
        help="Maximum stack size in bytes (default: 4096)",
    )
    parser.add_argument(
        "--max-heap-size",
        type=int,
        default=28672,
        help="Maximum heap size in bytes (default: 28672)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=2048,
        help="Circular buffer size in bytes (default: 2048)",
    )

    # DMA configuration
    parser.add_argument(
        "--dma-channel",
        type=int,
        default=0,
        help="DMA channel number (default: 0)",
    )
    parser.add_argument(
        "--dma-priority",
        type=int,
        default=2,
        help="DMA priority (default: 2)",
    )
    parser.add_argument(
        "--dma-burst-size",
        type=int,
        default=4,
        help="DMA burst size in words (default: 4)",
    )

    # Performance targets
    parser.add_argument(
        "--max-isr-latency-us",
        type=float,
        default=50.0,
        help="Maximum interrupt latency in microseconds (default: 50.0)",
    )
    parser.add_argument(
        "--target-throughput-mbps",
        type=float,
        default=100.0,
        help="Target throughput in MB/s (default: 100.0)",
    )

    # Flash configuration
    parser.add_argument(
        "--flash-sector-size",
        type=int,
        default=4096,
        help="Flash sector size in bytes (default: 4096)",
    )
    parser.add_argument(
        "--flash-read-latency-ns",
        type=float,
        default=200.0,
        help="Flash read latency in nanoseconds (default: 200.0)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--model-size",
        type=int,
        default=32768,
        help="Model size in bytes (default: 32768)",
    )
    parser.add_argument(
        "--num-evals",
        type=int,
        default=1000,
        help="Number of evaluations (default: 1000)",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/dispatcher",
        help="Output directory (default: ./outputs/dispatcher)",
    )

    # Other options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configuration without running"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.max_stack_size <= 0 or args.max_stack_size > 8192:
        print("Error: max_stack_size must be between 1 and 8192")
        sys.exit(1)

    if args.max_heap_size <= 0 or args.max_heap_size > 65536:
        print("Error: max_heap_size must be between 1 and 65536")
        sys.exit(1)

    if args.buffer_size <= 0 or args.buffer_size > args.max_heap_size:
        print("Error: buffer_size must be between 1 and max_heap_size")
        sys.exit(1)

    # Create configuration
    config = create_dispatcher_config(args)

    if args.verbose:
        print("Dispatcher Configuration:")
        print(json.dumps(config.__dict__, indent=2))

    if args.dry_run:
        print("Dry run mode - configuration validated")
        return

    # Run dispatcher benchmark
    try:
        results = run_dispatcher_benchmark(
            config=config,
            model_size=args.model_size,
            num_evals=args.num_evals,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("DISPATCHER BENCHMARK COMPLETED")
        print("=" * 50)

        report = results["report"]
        performance = report["performance"]
        memory_usage = report["memory_usage"]

        print(f"Status: {report['status']}")
        print(f"Load Count: {performance['load_count']}")
        print(f"Eval Count: {performance['eval_count']}")
        print(f"Miss Rate: {performance['miss_rate']:.2%}")
        print(f"Memory Efficiency: {performance['memory_efficiency']:.2%}")
        print(f"Used RAM: {memory_usage['used_ram']} bytes")
        print(f"Available RAM: {memory_usage['available_ram']} bytes")

        print(f"\nResults saved to: {args.output_dir}")

        if report["status"] == "PASSED":
            print("✅ Dispatcher PASSED - all constraints met")
        else:
            print("❌ Dispatcher FAILED - some constraints violated")
            failed_constraints = [
                constraint
                for constraint, passed in report["constraints"].items()
                if not passed
            ]
            print(f"Failed constraints: {failed_constraints}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during dispatcher benchmark: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
