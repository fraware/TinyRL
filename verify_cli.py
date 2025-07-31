#!/usr/bin/env python3
"""
Formal Verification CLI Script

Run formal verification pipeline with command-line interface.
"""

import argparse
import json
import sys

from tinyrl.verification import VerificationConfig, run_verification_pipeline


def create_verification_config(args) -> VerificationConfig:
    """Create verification configuration from command line arguments."""
    return VerificationConfig(
        epsilon=args.epsilon,
        max_states=args.max_states,
        timeout_seconds=args.timeout_seconds,
        lean_version=args.lean_version,
        lean_path=args.lean_path,
        lake_path=args.lake_path,
        smt_solver=args.smt_solver,
        smt_timeout=args.smt_timeout,
        output_dir=args.output_dir,
        save_proofs=args.save_proofs,
        generate_docs=args.generate_docs,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run formal verification pipeline")

    # Verification parameters
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Tolerance for reward preservation (default: 0.05)",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=1000,
        help="Maximum states to verify (default: 1000)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Verification timeout in seconds (default: 300)",
    )

    # Lean configuration
    parser.add_argument(
        "--lean-version",
        type=str,
        default="4.0.0",
        help="Lean version (default: 4.0.0)",
    )
    parser.add_argument(
        "--lean-path",
        type=str,
        default="lean",
        help="Path to lean executable (default: lean)",
    )
    parser.add_argument(
        "--lake-path",
        type=str,
        default="lake",
        help="Path to lake executable (default: lake)",
    )

    # SMT configuration
    parser.add_argument(
        "--smt-solver",
        type=str,
        default="z3",
        help="SMT solver to use (default: z3)",
    )
    parser.add_argument(
        "--smt-timeout",
        type=int,
        default=30,
        help="SMT query timeout in seconds (default: 30)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/verification",
        help="Output directory (default: ./outputs/verification)",
    )
    parser.add_argument(
        "--save-proofs",
        action="store_true",
        default=True,
        help="Save proof artifacts (default: True)",
    )
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        default=True,
        help="Generate documentation (default: True)",
    )

    # Other options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configuration without running"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.epsilon <= 0.0 or args.epsilon > 1.0:
        print("Error: epsilon must be between 0.0 and 1.0")
        sys.exit(1)

    if args.max_states <= 0 or args.max_states > 10000:
        print("Error: max_states must be between 1 and 10000")
        sys.exit(1)

    if args.timeout_seconds <= 0 or args.timeout_seconds > 3600:
        print("Error: timeout_seconds must be between 1 and 3600")
        sys.exit(1)

    # Create configuration
    config = create_verification_config(args)

    if args.verbose:
        print("Verification Configuration:")
        print(json.dumps(config.__dict__, indent=2))

    if args.dry_run:
        print("Dry run mode - configuration validated")
        return

    # Run verification pipeline
    try:
        results = run_verification_pipeline(
            config=config,
            output_dir=args.output_dir,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("FORMAL VERIFICATION COMPLETED")
        print("=" * 50)

        report = results["report"]
        theorem_results = report["theorem_results"]
        smt_results = report["smt_results"]

        print(f"Status: {report['status']}")
        print(f"Total Theorems: {report['total_theorems']}")
        print(f"Passed Theorems: {report['passed_theorems']}")
        print(f"Total SMT Queries: {report['total_smt_queries']}")
        print(f"Passed SMT Queries: {report['passed_smt_queries']}")

        print("\nTheorem Results:")
        for theorem, passed in theorem_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {theorem}: {status}")

        print("\nSMT Query Results:")
        for query, passed in smt_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {query}: {status}")

        print(f"\nResults saved to: {args.output_dir}")

        if report["status"] == "PASSED":
            print("✅ Verification PASSED - all proofs verified")
        else:
            print("❌ Verification FAILED - some proofs failed")
            failed_theorems = [
                theorem for theorem, passed in theorem_results.items() if not passed
            ]
            failed_smt = [query for query, passed in smt_results.items() if not passed]
            if failed_theorems:
                print(f"Failed theorems: {failed_theorems}")
            if failed_smt:
                print(f"Failed SMT queries: {failed_smt}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during verification: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
