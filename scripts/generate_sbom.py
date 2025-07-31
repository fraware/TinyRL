#!/usr/bin/env python3
"""
SBOM Generation Script

Generate Software Bill of Materials (SBOM) in CycloneDX format for TinyRL.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    from cyclonedx.model.bom import Bom
    from cyclonedx.model.component import Component, ComponentType
    from cyclonedx.model.vulnerability import Vulnerability, VulnerabilitySource
    from cyclonedx.output import OutputFormat, get_instance
    from cyclonedx.model import ExternalReference, ExternalReferenceType
    from cyclonedx.model.license import License, LicenseChoice
    from cyclonedx.model.tool import Tool, ToolVendor, ToolName
except ImportError:
    print("Error: cyclonedx-python-lib not installed")
    print("Install with: pip install cyclonedx-python-lib")
    sys.exit(1)


def get_python_dependencies() -> List[Dict[str, Any]]:
    """Get Python dependencies from requirements.txt."""
    dependencies = []

    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        return dependencies

    with open(requirements_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Parse package spec
                if "==" in line:
                    name, version = line.split("==", 1)
                elif ">=" in line:
                    name, version = line.split(">=", 1)
                elif "<=" in line:
                    name, version = line.split("<=", 1)
                else:
                    name = line
                    version = "unknown"

                dependencies.append(
                    {
                        "name": name.strip(),
                        "version": version.strip(),
                        "type": "library",
                    }
                )

    return dependencies


def get_git_info() -> Dict[str, str]:
    """Get Git repository information."""
    try:
        # Get commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()

        # Get repository URL
        repo_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        # Get tag if available
        try:
            tag = subprocess.check_output(
                ["git", "describe", "--tags", "--exact-match"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError:
            tag = None

        return {"commit_hash": commit_hash, "repo_url": repo_url, "tag": tag}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit_hash": "unknown", "repo_url": "unknown", "tag": None}


def get_vulnerabilities() -> List[Dict[str, Any]]:
    """Get known vulnerabilities from security scan."""
    # This would integrate with security scanning tools
    # For now, return empty list
    return []


def create_sbom(
    project_name: str = "TinyRL",
    project_version: str = "1.0.0",
    output_format: str = "json",
) -> Bom:
    """Create SBOM for TinyRL project."""

    # Get project information
    git_info = get_git_info()

    # Create main component
    main_component = Component(
        name=project_name,
        version=project_version,
        component_type=ComponentType.APPLICATION,
        description="Production-grade reinforcement learning library for microcontrollers",
        licenses=[
            LicenseChoice(license=License(id="Apache-2.0", name="Apache License 2.0"))
        ],
        external_references=[
            ExternalReference(
                reference_type=ExternalReferenceType.VCS, url=git_info["repo_url"]
            ),
            ExternalReference(
                reference_type=ExternalReferenceType.DISTRIBUTION,
                url=f"https://github.com/fraware/TinyRL/releases/tag/v{project_version}",
            ),
        ],
    )

    # Add metadata
    main_component.metadata = {
        "commit_hash": git_info["commit_hash"],
        "tag": git_info["tag"],
    }

    # Create SBOM
    bom = Bom()
    bom.metadata.component = main_component

    # Add tool information
    bom.metadata.tools = [
        Tool(vendor=ToolVendor.TINYRL, name=ToolName.CUSTOM_TOOL, version="1.0.0")
    ]

    # Add dependencies
    dependencies = get_python_dependencies()
    for dep in dependencies:
        component = Component(
            name=dep["name"],
            version=dep["version"],
            component_type=ComponentType.LIBRARY,
        )
        bom.components.add(component)

    # Add vulnerabilities
    vulnerabilities = get_vulnerabilities()
    for vuln in vulnerabilities:
        vulnerability = Vulnerability(
            id=vuln.get("id", "unknown"),
            source=VulnerabilitySource(
                name=vuln.get("source", "unknown"), url=vuln.get("url", "")
            ),
            description=vuln.get("description", ""),
            severity=vuln.get("severity", "unknown"),
        )
        bom.vulnerabilities.add(vulnerability)

    return bom


def save_sbom(bom: Bom, output_path: str, output_format: str = "json") -> None:
    """Save SBOM to file."""

    # Determine output format
    if output_format.lower() == "json":
        format_enum = OutputFormat.JSON
    elif output_format.lower() == "xml":
        format_enum = OutputFormat.XML
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    # Generate output
    output = get_instance(bom=bom, output_format=format_enum)

    # Save to file
    with open(output_path, "w") as f:
        f.write(output.output_as_string())

    print(f"SBOM saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate SBOM for TinyRL")

    parser.add_argument(
        "--project-name",
        type=str,
        default="TinyRL",
        help="Project name (default: TinyRL)",
    )

    parser.add_argument(
        "--project-version",
        type=str,
        default="1.0.0",
        help="Project version (default: 1.0.0)",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "xml"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="sbom.json",
        help="Output file path (default: sbom.json)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    try:
        # Create SBOM
        if args.verbose:
            print("Generating SBOM...")

        bom = create_sbom(
            project_name=args.project_name,
            project_version=args.project_version,
            output_format=args.output_format,
        )

        # Save SBOM
        save_sbom(bom, args.output_file, args.output_format)

        if args.verbose:
            print(f"SBOM generated successfully:")
            print(f"  - Project: {args.project_name} v{args.project_version}")
            print(f"  - Format: {args.output_format}")
            print(f"  - Output: {args.output_file}")
            print(f"  - Components: {len(bom.components)}")
            print(f"  - Vulnerabilities: {len(bom.vulnerabilities)}")

    except Exception as e:
        print(f"Error generating SBOM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
