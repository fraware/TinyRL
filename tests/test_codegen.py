#!/usr/bin/env python3
"""
Tests for cross-compilation and code generation functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tinyrl.codegen import (
    ArduinoGenerator,
    CMakeGenerator,
    CodegenConfig,
    CodegenTrainer,
    RustGenerator,
    TVMMicroBackend,
    create_codegen_report,
    run_codegen_pipeline,
)


class TestCodegenConfig:
    """Test code generation configuration."""

    def test_config_creation(self):
        """Test configuration creation with defaults."""
        config = CodegenConfig()

        assert config.target_mcu == "cortex-m55"
        assert config.target_arch == "armv8-m.main"
        assert config.target_float_abi == "hard"
        assert config.optimization_level == "O2"
        assert config.debug_info is False
        assert config.stack_protection is True
        assert config.misra_compliance is True
        assert config.max_stack_size == 4096
        assert config.max_heap_size == 28672
        assert config.max_flash_size == 131072
        assert config.inline_threshold == 100
        assert config.function_sections is True
        assert config.data_sections is True
        assert config.use_tvm_micro is True
        assert config.use_glow is False
        assert config.use_cmsis_nn is True
        assert config.rust_target == "thumbv8m.main-none-eabihf"
        assert config.arduino_board == "nano33ble"
        assert config.arduino_library_name == "TinyRL"

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = CodegenConfig(
            target_mcu="cortex-m4",
            optimization_level="O3",
            max_stack_size=2048,
            use_glow=True,
        )

        assert config.target_mcu == "cortex-m4"
        assert config.optimization_level == "O3"
        assert config.max_stack_size == 2048
        assert config.use_glow is True


class TestCMakeGenerator:
    """Test CMake generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CodegenConfig()
        self.generator = CMakeGenerator(self.config)

    def test_generate_cmake_lists(self):
        """Test CMakeLists.txt generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.generator.generate_cmake_lists(output_dir)

            cmake_file = output_dir / "CMakeLists.txt"
            assert cmake_file.exists()

            content = cmake_file.read_text()
            assert "cmake_minimum_required" in content
            assert "project(TinyRL MCU)" in content
            assert "CMAKE_C_STANDARD 11" in content

    def test_generate_makefile(self):
        """Test Makefile generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.generator.generate_makefile(output_dir)

            makefile = output_dir / "Makefile"
            assert makefile.exists()

            content = makefile.read_text()
            assert "CC = arm-none-eabi-gcc" in content
            assert "TARGET = tinyrl_mcu" in content
            assert "CFLAGS" in content


class TestRustGenerator:
    """Test Rust generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CodegenConfig()
        self.generator = RustGenerator(self.config)

    def test_generate_cargo_toml(self):
        """Test Cargo.toml generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.generator.generate_cargo_toml(output_dir)

            cargo_file = output_dir / "Cargo.toml"
            assert cargo_file.exists()

            content = cargo_file.read_text()
            assert 'name = "tinyrl-runtime"' in content
            assert 'version = "0.1.0"' in content
            assert 'edition = "2021"' in content
            assert "cortex-m = " in content

    def test_generate_lib_rs(self):
        """Test lib.rs generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.generator.generate_lib_rs(output_dir)

            lib_file = output_dir / "src" / "lib.rs"
            assert lib_file.exists()

            content = lib_file.read_text()
            assert "#![no_std]" in content
            assert "TinyRL Runtime for Microcontrollers" in content
            assert "pub struct TinyRLPolicy" in content
            assert 'pub extern "C" fn tinyrl_inference' in content


class TestArduinoGenerator:
    """Test Arduino generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CodegenConfig()
        self.generator = ArduinoGenerator(self.config)

    def test_generate_library_properties(self):
        """Test library.properties generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.generator.generate_library_properties(output_dir)

            properties_file = output_dir / "library.properties"
            assert properties_file.exists()

            content = properties_file.read_text()
            assert "name=TinyRL" in content
            assert "version=1.0.0" in content
            assert "author=TinyRL Team" in content

    def test_generate_header(self):
        """Test TinyRL.h generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.generator.generate_header(output_dir)

            header_file = output_dir / "src" / "TinyRL.h"
            assert header_file.exists()

            content = header_file.read_text()
            assert "#ifndef TINYRL_H" in content
            assert "typedef struct" in content
            assert "TinyRLPolicy_t" in content
            assert "class TinyRL" in content

    def test_generate_implementation(self):
        """Test TinyRL.cpp generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.generator.generate_implementation(output_dir)

            impl_file = output_dir / "src" / "TinyRL.cpp"
            assert impl_file.exists()

            content = impl_file.read_text()
            assert '#include "TinyRL.h"' in content
            assert "tinyrl_init_policy" in content
            assert "tinyrl_inference" in content
            assert "TinyRL::TinyRL()" in content


class TestTVMMicroBackend:
    """Test TVM-Micro backend."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CodegenConfig()
        self.backend = TVMMicroBackend(self.config)

        # Create a simple model
        self.model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))

    def test_generate_tvm_model(self):
        """Test TVM model generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            self.backend.generate_tvm_model(self.model, output_dir)

            tvm_file = output_dir / "tvm_model.py"
            assert tvm_file.exists()

            content = tvm_file.read_text()
            assert "TVM Model Definition" in content
            assert "import tvm" in content
            assert "def create_tinyrl_model()" in content


class TestCodegenTrainer:
    """Test code generation trainer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CodegenConfig()
        self.trainer = CodegenTrainer(self.config)

        # Create a simple model
        self.model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))

    def test_trainer_creation(self):
        """Test trainer creation."""
        assert self.trainer.config == self.config
        assert hasattr(self.trainer, "cmake_gen")
        assert hasattr(self.trainer, "rust_gen")
        assert hasattr(self.trainer, "arduino_gen")
        assert hasattr(self.trainer, "tvm_backend")

    def test_generate_all(self):
        """Test generation of all targets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = self.trainer.generate_all(self.model, temp_dir)

            assert "cmake" in results
            assert "rust" in results
            assert "arduino" in results
            assert "tvm" in results
            assert "config" in results

            # Check that files were generated
            output_path = Path(temp_dir)
            assert (output_path / "codegen_results.json").exists()

    def test_generate_cmake(self):
        """Test CMake generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            results = self.trainer._generate_cmake(output_path)

            assert "cmake_lists" in results
            assert "makefile" in results

            cmake_dir = output_path / "cmake"
            assert (cmake_dir / "CMakeLists.txt").exists()
            assert (cmake_dir / "Makefile").exists()

    def test_generate_rust(self):
        """Test Rust generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            results = self.trainer._generate_rust(output_path)

            assert "cargo_toml" in results
            assert "lib_rs" in results

            rust_dir = output_path / "rust"
            assert (rust_dir / "Cargo.toml").exists()
            assert (rust_dir / "src" / "lib.rs").exists()

    def test_generate_arduino(self):
        """Test Arduino generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            results = self.trainer._generate_arduino(output_path)

            assert "library_properties" in results
            assert "header" in results
            assert "implementation" in results

            arduino_dir = output_path / "arduino"
            assert (arduino_dir / "library.properties").exists()
            assert (arduino_dir / "src" / "TinyRL.h").exists()
            assert (arduino_dir / "src" / "TinyRL.cpp").exists()

    def test_generate_tvm(self):
        """Test TVM generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            results = self.trainer._generate_tvm(self.model, output_path)

            assert "tvm_model" in results

            tvm_dir = output_path / "tvm"
            assert (tvm_dir / "tvm_model.py").exists()


class TestCodegenReport:
    """Test code generation report."""

    def test_report_creation(self):
        """Test report creation."""
        config = CodegenConfig()

        results = {
            "cmake": {"cmake_lists": "test.cmake"},
            "rust": {"cargo_toml": "test.cargo"},
            "arduino": {"library_properties": "test.properties"},
            "tvm": {"tvm_model": "test.tvm"},
        }

        report = create_codegen_report(results, config)

        assert "status" in report
        assert "passed" in report
        assert "targets" in report
        assert "config" in report
        assert "files_generated" in report

        # Check that all targets are marked as successful
        assert report["status"] == "PASSED"
        assert report["passed"] is True
        assert all(report["targets"].values())

    def test_report_failure(self):
        """Test report with failed targets."""
        config = CodegenConfig()

        results = {
            "cmake": {},  # Missing required files
            "rust": {"cargo_toml": "test.cargo"},
            "arduino": {},  # Missing required files
            "tvm": {"tvm_model": "test.tvm"},
        }

        report = create_codegen_report(results, config)

        assert report["status"] == "FAILED"
        assert report["passed"] is False
        assert not report["targets"]["cmake"]
        assert report["targets"]["rust"]
        assert not report["targets"]["arduino"]
        assert report["targets"]["tvm"]


class TestCodegenPipeline:
    """Test complete code generation pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = CodegenConfig()

        # Create a simple model
        self.model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))

    def test_pipeline_integration(self):
        """Test complete pipeline integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_codegen_pipeline(
                config=self.config, model=self.model, output_dir=temp_dir
            )

            assert "cmake" in results
            assert "rust" in results
            assert "arduino" in results
            assert "tvm" in results
            assert "config" in results
            assert "report" in results

            # Check report
            report = results["report"]
            assert report["status"] == "PASSED"
            assert report["passed"] is True

            # Check that files were generated
            output_path = Path(temp_dir)
            assert (output_path / "codegen_results.json").exists()


class TestDeterministicBehavior:
    """Test deterministic behavior of code generation."""

    def test_deterministic_generation(self):
        """Test that code generation is deterministic."""
        config = CodegenConfig()
        model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))

        with tempfile.TemporaryDirectory() as temp_dir_1:
            with tempfile.TemporaryDirectory() as temp_dir_2:
                # Generate twice with same seed
                torch.manual_seed(42)
                results_1 = run_codegen_pipeline(
                    config=config, model=model, output_dir=temp_dir_1
                )

                torch.manual_seed(42)
                results_2 = run_codegen_pipeline(
                    config=config, model=model, output_dir=temp_dir_2
                )

                # Reports should be identical
                assert results_1["report"]["status"] == results_2["report"]["status"]
                assert results_1["report"]["passed"] == results_2["report"]["passed"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_model(self):
        """Test code generation with empty model."""
        config = CodegenConfig()
        empty_model = nn.Sequential()

        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_codegen_pipeline(
                config=config, model=empty_model, output_dir=temp_dir
            )

            # Should still generate all targets
            assert "cmake" in results
            assert "rust" in results
            assert "arduino" in results
            assert "tvm" in results

    def test_invalid_config(self):
        """Test code generation with invalid configuration."""
        config = CodegenConfig(max_stack_size=0, max_heap_size=0)  # Invalid  # Invalid

        model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle gracefully
            results = run_codegen_pipeline(
                config=config, model=model, output_dir=temp_dir
            )

            assert "report" in results
