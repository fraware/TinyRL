#!/usr/bin/env python3
"""
Cross-Compiler and Code Generation CLI Script

Run cross-compilation and code generation pipeline with command-line interface.
"""

import argparse
import json
import os
import sys

import torch

from tinyrl.codegen import CodegenConfig, run_codegen_pipeline


def create_codegen_config(args) -> CodegenConfig:
    """Create code generation configuration from command line arguments."""
    return CodegenConfig(
        target_mcu=args.target_mcu,
        target_arch=args.target_arch,
        target_float_abi=args.target_float_abi,
        optimization_level=args.optimization_level,
        debug_info=args.debug_info,
        stack_protection=args.stack_protection,
        misra_compliance=args.misra_compliance,
        max_stack_size=args.max_stack_size,
        max_heap_size=args.max_heap_size,
        max_flash_size=args.max_flash_size,
        inline_threshold=args.inline_threshold,
        function_sections=args.function_sections,
        data_sections=args.data_sections,
        use_tvm_micro=args.use_tvm_micro,
        use_glow=args.use_glow,
        use_cmsis_nn=args.use_cmsis_nn,
        rust_target=args.rust_target,
        arduino_board=args.arduino_board,
        arduino_library_name=args.arduino_library_name,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run cross-compilation and code generation pipeline"
    )

    # Required arguments
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument(
        "env_name", help="Environment name (e.g., CartPole-v1, LunarLander-v2)"
    )

    # Target platform
    parser.add_argument(
        "--target-mcu",
        type=str,
        default="cortex-m55",
        help="Target MCU (default: cortex-m55)",
    )
    parser.add_argument(
        "--target-arch",
        type=str,
        default="armv8-m.main",
        help="Target architecture (default: armv8-m.main)",
    )
    parser.add_argument(
        "--target-float-abi",
        type=str,
        default="hard",
        choices=["soft", "softfp", "hard"],
        help="Target float ABI (default: hard)",
    )

    # Compilation options
    parser.add_argument(
        "--optimization-level",
        type=str,
        default="O2",
        choices=["O0", "O1", "O2", "O3", "Os"],
        help="Optimization level (default: O2)",
    )
    parser.add_argument(
        "--debug-info", action="store_true", help="Include debug information"
    )
    parser.add_argument(
        "--stack-protection",
        action="store_true",
        default=True,
        help="Enable stack protection",
    )
    parser.add_argument(
        "--misra-compliance",
        action="store_true",
        default=True,
        help="Enable MISRA-C compliance",
    )

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
        "--max-flash-size",
        type=int,
        default=131072,
        help="Maximum flash size in bytes (default: 131072)",
    )

    # Code generation
    parser.add_argument(
        "--inline-threshold",
        type=int,
        default=100,
        help="Inline threshold (default: 100)",
    )
    parser.add_argument(
        "--function-sections",
        action="store_true",
        default=True,
        help="Use function sections",
    )
    parser.add_argument(
        "--data-sections", action="store_true", default=True, help="Use data sections"
    )

    # Backend options
    parser.add_argument(
        "--use-tvm-micro",
        action="store_true",
        default=True,
        help="Use TVM-Micro backend",
    )
    parser.add_argument("--use-glow", action="store_true", help="Use Glow backend")
    parser.add_argument(
        "--use-cmsis-nn", action="store_true", default=True, help="Use CMSIS-NN"
    )

    # Rust options
    parser.add_argument(
        "--rust-target",
        type=str,
        default="thumbv8m.main-none-eabihf",
        help="Rust target triple (default: thumbv8m.main-none-eabihf)",
    )

    # Arduino options
    parser.add_argument(
        "--arduino-board",
        type=str,
        default="nano33ble",
        help="Arduino board (default: nano33ble)",
    )
    parser.add_argument(
        "--arduino-library-name",
        type=str,
        default="TinyRL",
        help="Arduino library name (default: TinyRL)",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/codegen",
        help="Output directory (default: ./outputs/codegen)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Other options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configuration without running"
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)

    if args.max_stack_size <= 0 or args.max_stack_size > 32768:
        print("Error: max_stack_size must be between 1 and 32768")
        sys.exit(1)

    if args.max_heap_size <= 0 or args.max_heap_size > 65536:
        print("Error: max_heap_size must be between 1 and 65536")
        sys.exit(1)

    if args.max_flash_size <= 0 or args.max_flash_size > 1048576:
        print("Error: max_flash_size must be between 1 and 1048576")
        sys.exit(1)

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create configuration
    config = create_codegen_config(args)

    if args.verbose:
        print("Code Generation Configuration:")
        print(json.dumps(config.__dict__, indent=2))

    if args.dry_run:
        print("Dry run mode - configuration validated")
        return

    # Load model
    try:
        model = torch.load(args.model_path, map_location="cpu")
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Run code generation pipeline
    try:
        results = run_codegen_pipeline(
            config=config,
            model=model,
            output_dir=args.output_dir,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("CODE GENERATION COMPLETED")
        print("=" * 50)

        report = results["report"]
        targets = report["targets"]

        print(f"Status: {report['status']}")
        print(f"Files Generated: {report['files_generated']}")
        print(f"Targets:")
        print(f"  CMake: {'✅' if targets['cmake'] else '❌'}")
        print(f"  Rust: {'✅' if targets['rust'] else '❌'}")
        print(f"  Arduino: {'✅' if targets['arduino'] else '❌'}")
        print(f"  TVM: {'✅' if targets['tvm'] else '❌'}")

        print(f"\nResults saved to: {args.output_dir}")

        if report["passed"]:
            print("✅ Code Generation PASSED - all targets generated")
        else:
            print("❌ Code Generation FAILED - some targets failed")
            failed_targets = [
                target for target, passed in targets.items() if not passed
            ]
            print(f"Failed targets: {failed_targets}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during code generation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
