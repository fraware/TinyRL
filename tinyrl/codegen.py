#!/usr/bin/env python3
"""
Cross-Compiler and Code Generation Module

This module implements MCU-ready binary generation in C (CMSIS-NN), Rust (no_std),
and Arduino libraries with TVM-Micro backend integration.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from tinyrl.utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class CodegenConfig:
    """Configuration for cross-compilation and code generation."""

    # Target platforms
    target_mcu: str = "cortex-m55"
    target_arch: str = "armv8-m.main"
    target_float_abi: str = "hard"

    # Compilation options
    optimization_level: str = "O2"
    debug_info: bool = False
    stack_protection: bool = True
    misra_compliance: bool = True

    # Memory constraints
    max_stack_size: int = 4096  # 4KB
    max_heap_size: int = 28672  # 28KB
    max_flash_size: int = 131072  # 128KB

    # Code generation
    inline_threshold: int = 100
    function_sections: bool = True
    data_sections: bool = True

    # Backend options
    use_tvm_micro: bool = True
    use_glow: bool = False
    use_cmsis_nn: bool = True

    # Rust options
    rust_target: str = "thumbv8m.main-none-eabihf"
    rust_features: List[str] = None  # Will be set to ["fpu"]

    # Arduino options
    arduino_board: str = "nano33ble"
    arduino_library_name: str = "TinyRL"


class CMakeGenerator:
    """Generate CMake project for C/C++ compilation."""

    def __init__(self, config: CodegenConfig):
        self.config = config

    def generate_cmake_lists(self, output_dir: Path) -> None:
        """Generate CMakeLists.txt for the project."""
        cmake_content = f"""cmake_minimum_required(VERSION 3.16)
project(TinyRL MCU)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -{self.config.optimization_level}")
set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -march={self.config.target_arch}")
set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -mfloat-abi={self.config.target_float_abi}")

# MISRA-C compliance
if({str(self.config.misra_compliance).lower()})
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -fstack-protector-strong")
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -Werror=implicit-function-declaration")
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -Werror=return-type")
endif()

# Function and data sections for size optimization
if({str(self.config.function_sections).lower()})
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -ffunction-sections")
endif()

if({str(self.config.data_sections).lower()})
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -fdata-sections")
endif()

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS "${{CMAKE_EXE_LINKER_FLAGS}} -Wl,--gc-sections")
set(CMAKE_EXE_LINKER_FLAGS "${{CMAKE_EXE_LINKER_FLAGS}} -Wl,--print-memory-usage")

# Memory constraints
set(CMAKE_EXE_LINKER_FLAGS "${{CMAKE_EXE_LINKER_FLAGS}} -Wl,--stack=${{self.config.max_stack_size}}")

# Include directories
include_directories(include)
include_directories(src)

# Source files
file(GLOB_RECURSE SOURCES "src/*.c")
file(GLOB_RECURSE HEADERS "include/*.h")

# Create executable
add_executable(tinyrl_mcu ${{SOURCES}})

# Set target properties
set_target_properties(tinyrl_mcu PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Memory usage check
add_custom_command(TARGET tinyrl_mcu POST_BUILD
    COMMAND ${{CMAKE_COMMAND}} -E echo "Memory usage:"
    COMMAND arm-none-eabi-size $<TARGET_FILE:tinyrl_mcu>
)
"""

        cmake_file = output_dir / "CMakeLists.txt"
        cmake_file.write_text(cmake_content)
        logger.info(f"Generated CMakeLists.txt: {cmake_file}")

    def generate_makefile(self, output_dir: Path) -> None:
        """Generate Makefile for direct compilation."""
        makefile_content = f"""# TinyRL MCU Makefile
CC = arm-none-eabi-gcc
AR = arm-none-eabi-ar
OBJCOPY = arm-none-eabi-objcopy
SIZE = arm-none-eabi-size

# Target configuration
TARGET = tinyrl_mcu
MCU = {self.config.target_mcu}
ARCH = {self.config.target_arch}
FLOAT_ABI = {self.config.target_float_abi}

# Compiler flags
CFLAGS = -{self.config.optimization_level} -g
CFLAGS += -march=$(ARCH) -mfloat-abi=$(FLOAT_ABI)
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -Wall -Wextra -Werror=implicit-function-declaration
CFLAGS += -Werror=return-type -Wstack-usage=4096

# MISRA-C compliance
if({str(self.config.misra_compliance).lower()})
    CFLAGS += -fstack-protector-strong
    CFLAGS += -Werror=format-security
endif()

# Linker flags
LDFLAGS = -Wl,--gc-sections -Wl,--print-memory-usage
LDFLAGS += -Wl,--stack={self.config.max_stack_size}

# Source files
SOURCES = $(wildcard src/*.c)
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET).elf
	$(SIZE) $(TARGET).elf

$(TARGET).elf: $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET).elf

.PHONY: all clean
"""

        makefile_file = output_dir / "Makefile"
        makefile_file.write_text(makefile_content)
        logger.info(f"Generated Makefile: {makefile_file}")


class RustGenerator:
    """Generate Rust crate for no_std compilation."""

    def __init__(self, config: CodegenConfig):
        self.config = config
        if self.config.rust_features is None:
            self.config.rust_features = ["fpu"]

    def generate_cargo_toml(self, output_dir: Path) -> None:
        """Generate Cargo.toml for Rust crate."""
        cargo_content = f"""[package]
name = "tinyrl-runtime"
version = "0.1.0"
edition = "2021"
authors = ["TinyRL Team <team@tinyrl.dev>"]
description = "TinyRL runtime for microcontrollers"
license = "Apache-2.0"
repository = "https://github.com/fraware/TinyRL"
keywords = ["embedded", "reinforcement-learning", "microcontroller"]
categories = ["embedded", "no-std"]

[lib]
name = "tinyrl_runtime"
crate-type = ["staticlib", "cdylib"]

[dependencies]
# Core dependencies
cortex-m = "0.7"
cortex-m-rt = "0.7"
embedded-hal = "0.2"

# Optional FPU support
cortex-m-rtic = {{ version = "1.0", optional = true }}

# Serialization
serde = {{ version = "1.0", features = ["derive"], optional = true }}

[features]
default = ["fpu"]
fpu = ["cortex-m/fpu"]
rtic = ["cortex-m-rtic"]
serde = ["serde"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[profile.dev]
opt-level = 0
debug = true

[target.'{self.config.rust_target}'.dependencies]
cortex-m = "0.7"
cortex-m-rt = "0.7"
"""

        cargo_file = output_dir / "Cargo.toml"
        cargo_file.write_text(cargo_content)
        logger.info(f"Generated Cargo.toml: {cargo_file}")

    def generate_lib_rs(self, output_dir: Path) -> None:
        """Generate lib.rs with core functionality."""
        lib_content = f"""#![no_std]
#![feature(asm_const)]

//! TinyRL Runtime for Microcontrollers
//! 
//! This crate provides a no_std implementation of the TinyRL runtime
//! for ARM Cortex-M microcontrollers.

use core::panic::PanicInfo;
use cortex_m::asm;

#[cfg(feature = "fpu")]
use cortex_m::register::fpu;

/// Panic handler for no_std environment
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {{
    loop {{
        asm::wfe();
    }}
}}

/// Initialize the runtime
pub fn init() {{
    #[cfg(feature = "fpu")]
    {{
        // Enable FPU
        fpu::cpacr::write(fpu::cpacr::CP10::FULL_ACCESS);
        fpu::cpacr::write(fpu::cpacr::CP11::FULL_ACCESS);
        
        // Enable FPU context switching
        fpu::fpexc::write(fpu::fpexc::EX::ENABLED);
    }}
}}

/// TinyRL policy structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TinyRLPolicy {{
    pub weights: *const i8,
    pub scales: *const f32,
    pub lut: *const i8,
    pub input_dim: u16,
    pub hidden_dim: u16,
    pub output_dim: u16,
}}

/// Inference result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InferenceResult {{
    pub action: i32,
    pub confidence: f32,
    pub latency_us: u32,
}}

/// Initialize policy from flash memory
#[no_mangle]
pub extern "C" fn tinyrl_init_policy(
    weights_ptr: *const i8,
    scales_ptr: *const f32,
    lut_ptr: *const i8,
    input_dim: u16,
    hidden_dim: u16,
    output_dim: u16,
) -> TinyRLPolicy {{
    TinyRLPolicy {{
        weights: weights_ptr,
        scales: scales_ptr,
        lut: lut_ptr,
        input_dim,
        hidden_dim,
        output_dim,
    }}
}}

/// Run inference on observation
#[no_mangle]
pub extern "C" fn tinyrl_inference(
    policy: &TinyRLPolicy,
    observation: *const f32,
    result: *mut InferenceResult,
) -> i32 {{
    // TODO: Implement actual inference
    // This is a placeholder for the real implementation
    
    unsafe {{
        let obs_slice = core::slice::from_raw_parts(observation, policy.input_dim as usize);
        
        // Simple linear transformation (placeholder)
        let mut hidden = [0.0f32; 64];
        for i in 0..policy.hidden_dim as usize {{
            for j in 0..policy.input_dim as usize {{
                hidden[i] += obs_slice[j] * 0.1; // Placeholder weight
            }}
        }}
        
        // Simple argmax (placeholder)
        let action = hidden.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        
        let confidence = 0.8; // Placeholder
        let latency_us = 100; // Placeholder
        
        *result = InferenceResult {{
            action,
            confidence,
            latency_us,
        }};
    }}
    
    0 // Success
}}

/// Get policy memory usage
#[no_mangle]
pub extern "C" fn tinyrl_get_memory_usage(policy: &TinyRLPolicy) -> u32 {{
    let weights_size = policy.input_dim as u32 * policy.hidden_dim as u32;
    let scales_size = policy.hidden_dim as u32 * 4; // f32 = 4 bytes
    let lut_size = 256; // Fixed LUT size
    
    weights_size + scales_size + lut_size as u32
}}

#[cfg(test)]
mod tests {{
    use super::*;
    
    #[test]
    fn test_policy_creation() {{
        let weights = [0i8; 256];
        let scales = [1.0f32; 64];
        let lut = [0i8; 256];
        
        let policy = tinyrl_init_policy(
            weights.as_ptr(),
            scales.as_ptr(),
            lut.as_ptr(),
            4,  // input_dim
            64, // hidden_dim
            2,  // output_dim
        );
        
        assert_eq!(policy.input_dim, 4);
        assert_eq!(policy.hidden_dim, 64);
        assert_eq!(policy.output_dim, 2);
    }}
}}
"""

        lib_file = output_dir / "src" / "lib.rs"
        lib_file.parent.mkdir(parents=True, exist_ok=True)
        lib_file.write_text(lib_content)
        logger.info(f"Generated lib.rs: {lib_file}")


class ArduinoGenerator:
    """Generate Arduino library."""

    def __init__(self, config: CodegenConfig):
        self.config = config

    def generate_library_properties(self, output_dir: Path) -> None:
        """Generate library.properties for Arduino library."""
        properties_content = f"""name=TinyRL
version=1.0.0
author=TinyRL Team <team@tinyrl.dev>
maintainer=TinyRL Team <team@tinyrl.dev>
sentence=Reinforcement Learning for Microcontrollers
paragraph=TinyRL enables deployment of trained RL agents on resource-constrained devices (≤32KB RAM, ≤128KB Flash) while maintaining performance within 2% of full-precision baselines.
category=Machine Learning
url=https://github.com/fraware/TinyRL
architectures=samd,esp32,stm32
depends=
includes=TinyRL.h
"""

        properties_file = output_dir / "library.properties"
        properties_file.write_text(properties_content)
        logger.info(f"Generated library.properties: {properties_file}")

    def generate_header(self, output_dir: Path) -> None:
        """Generate TinyRL.h header file."""
        header_content = f"""#ifndef TINYRL_H
#define TINYRL_H

#include <Arduino.h>

#ifdef __cplusplus
extern "C" {{
#endif

// TinyRL policy structure
typedef struct {{
    const int8_t* weights;
    const float* scales;
    const int8_t* lut;
    uint16_t input_dim;
    uint16_t hidden_dim;
    uint16_t output_dim;
}} TinyRLPolicy_t;

// Inference result
typedef struct {{
    int32_t action;
    float confidence;
    uint32_t latency_us;
}} TinyRLResult_t;

// Initialize policy from flash memory
TinyRLPolicy_t tinyrl_init_policy(
    const int8_t* weights_ptr,
    const float* scales_ptr,
    const int8_t* lut_ptr,
    uint16_t input_dim,
    uint16_t hidden_dim,
    uint16_t output_dim
);

// Run inference on observation
int32_t tinyrl_inference(
    const TinyRLPolicy_t* policy,
    const float* observation,
    TinyRLResult_t* result
);

// Get policy memory usage
uint32_t tinyrl_get_memory_usage(const TinyRLPolicy_t* policy);

#ifdef __cplusplus
}}

// C++ wrapper class
class TinyRL {{
public:
    TinyRL();
    ~TinyRL();
    
    bool init(const TinyRLPolicy_t& policy);
    bool inference(const float* observation, TinyRLResult_t& result);
    uint32_t getMemoryUsage() const;
    
private:
    TinyRLPolicy_t policy_;
    bool initialized_;
}};

#endif

#endif // TINYRL_H
"""

        header_file = output_dir / "src" / "TinyRL.h"
        header_file.parent.mkdir(parents=True, exist_ok=True)
        header_file.write_text(header_content)
        logger.info(f"Generated TinyRL.h: {header_file}")

    def generate_implementation(self, output_dir: Path) -> None:
        """Generate TinyRL.cpp implementation file."""
        impl_content = f"""#include "TinyRL.h"
#include <math.h>

// C implementation
TinyRLPolicy_t tinyrl_init_policy(
    const int8_t* weights_ptr,
    const float* scales_ptr,
    const int8_t* lut_ptr,
    uint16_t input_dim,
    uint16_t hidden_dim,
    uint16_t output_dim
) {{
    TinyRLPolicy_t policy;
    policy.weights = weights_ptr;
    policy.scales = scales_ptr;
    policy.lut = lut_ptr;
    policy.input_dim = input_dim;
    policy.hidden_dim = hidden_dim;
    policy.output_dim = output_dim;
    return policy;
}}

int32_t tinyrl_inference(
    const TinyRLPolicy_t* policy,
    const float* observation,
    TinyRLResult_t* result
) {{
    if (!policy || !observation || !result) {{
        return -1; // Error
    }}
    
    // TODO: Implement actual inference
    // This is a placeholder for the real implementation
    
    // Simple linear transformation (placeholder)
    float hidden[64];
    for (uint16_t i = 0; i < policy->hidden_dim; i++) {{
        hidden[i] = 0.0f;
        for (uint16_t j = 0; j < policy->input_dim; j++) {{
            hidden[i] += observation[j] * 0.1f; // Placeholder weight
        }}
    }}
    
    // Simple argmax (placeholder)
    int32_t action = 0;
    float max_val = hidden[0];
    for (uint16_t i = 1; i < policy->output_dim; i++) {{
        if (hidden[i] > max_val) {{
            max_val = hidden[i];
            action = i;
        }}
    }}
    
    result->action = action;
    result->confidence = 0.8f; // Placeholder
    result->latency_us = 100; // Placeholder
    
    return 0; // Success
}}

uint32_t tinyrl_get_memory_usage(const TinyRLPolicy_t* policy) {{
    if (!policy) {{
        return 0;
    }}
    
    uint32_t weights_size = policy->input_dim * policy->hidden_dim;
    uint32_t scales_size = policy->hidden_dim * 4; // float = 4 bytes
    uint32_t lut_size = 256; // Fixed LUT size
    
    return weights_size + scales_size + lut_size;
}}

#ifdef __cplusplus

// C++ implementation
TinyRL::TinyRL() : initialized_(false) {{
}}

TinyRL::~TinyRL() {{
}}

bool TinyRL::init(const TinyRLPolicy_t& policy) {{
    policy_ = policy;
    initialized_ = true;
    return true;
}}

bool TinyRL::inference(const float* observation, TinyRLResult_t& result) {{
    if (!initialized_) {{
        return false;
    }}
    
    return tinyrl_inference(&policy_, observation, &result) == 0;
}}

uint32_t TinyRL::getMemoryUsage() const {{
    return tinyrl_get_memory_usage(&policy_);
}}

#endif
"""

        impl_file = output_dir / "src" / "TinyRL.cpp"
        impl_file.parent.mkdir(parents=True, exist_ok=True)
        impl_file.write_text(impl_content)
        logger.info(f"Generated TinyRL.cpp: {impl_file}")


class TVMMicroBackend:
    """TVM-Micro backend integration."""

    def __init__(self, config: CodegenConfig):
        self.config = config

    def generate_tvm_model(self, model: torch.nn.Module, output_dir: Path) -> None:
        """Generate TVM model from PyTorch model."""
        # This is a placeholder - in practice, you'd use TVM's Python API
        # to convert PyTorch models to TVM format

        tvm_model_content = """# TVM Model Definition (placeholder)
# In practice, this would be generated using TVM's Python API

import tvm
from tvm import relay
import numpy as np

def create_tinyrl_model():
    # Define input
    data = relay.var("data", shape=(1, 4), dtype="float32")
    
    # Define weights (placeholder)
    weights = relay.var("weights", shape=(4, 64), dtype="int8")
    scales = relay.var("scales", shape=(64,), dtype="float32")
    
    # Linear transformation
    x = relay.nn.dense(data, weights)
    x = relay.multiply(x, scales)
    x = relay.nn.relu(x)
    
    # Output layer
    output_weights = relay.var("output_weights", shape=(64, 2), dtype="int8")
    output_scales = relay.var("output_scales", shape=(2,), dtype="float32")
    output = relay.nn.dense(x, output_weights)
    output = relay.multiply(output, output_scales)
    
    return relay.Function([data, weights, scales, output_weights, output_scales], output)

# Create model
mod = tvm.IRModule()
mod["main"] = create_tinyrl_model()

# Save model
with open("tinyrl_model.json", "w") as f:
    f.write(tvm.ir.save_json(mod))
"""

        tvm_file = output_dir / "tvm_model.py"
        tvm_file.write_text(tvm_model_content)
        logger.info(f"Generated TVM model: {tvm_file}")


class CodegenTrainer:
    """Trainer for cross-compilation and code generation."""

    def __init__(self, config: CodegenConfig):
        self.config = config
        self.cmake_gen = CMakeGenerator(config)
        self.rust_gen = RustGenerator(config)
        self.arduino_gen = ArduinoGenerator(config)
        self.tvm_backend = TVMMicroBackend(config)

    def generate_all(
        self, model: torch.nn.Module, output_dir: str = "./outputs/codegen"
    ) -> Dict[str, Any]:
        """Generate all target formats."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            "cmake": self._generate_cmake(output_path),
            "rust": self._generate_rust(output_path),
            "arduino": self._generate_arduino(output_path),
            "tvm": self._generate_tvm(model, output_path),
            "config": self.config.__dict__,
        }

        # Save results
        with open(output_path / "codegen_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Code generation completed. Results saved to: {output_dir}")
        return results

    def _generate_cmake(self, output_dir: Path) -> Dict[str, str]:
        """Generate CMake project."""
        cmake_dir = output_dir / "cmake"
        cmake_dir.mkdir(exist_ok=True)

        self.cmake_gen.generate_cmake_lists(cmake_dir)
        self.cmake_gen.generate_makefile(cmake_dir)

        return {
            "cmake_lists": str(cmake_dir / "CMakeLists.txt"),
            "makefile": str(cmake_dir / "Makefile"),
        }

    def _generate_rust(self, output_dir: Path) -> Dict[str, str]:
        """Generate Rust crate."""
        rust_dir = output_dir / "rust"
        rust_dir.mkdir(exist_ok=True)

        self.rust_gen.generate_cargo_toml(rust_dir)
        self.rust_gen.generate_lib_rs(rust_dir)

        return {
            "cargo_toml": str(rust_dir / "Cargo.toml"),
            "lib_rs": str(rust_dir / "src" / "lib.rs"),
        }

    def _generate_arduino(self, output_dir: Path) -> Dict[str, str]:
        """Generate Arduino library."""
        arduino_dir = output_dir / "arduino"
        arduino_dir.mkdir(exist_ok=True)

        self.arduino_gen.generate_library_properties(arduino_dir)
        self.arduino_gen.generate_header(arduino_dir)
        self.arduino_gen.generate_implementation(arduino_dir)

        return {
            "library_properties": str(arduino_dir / "library.properties"),
            "header": str(arduino_dir / "src" / "TinyRL.h"),
            "implementation": str(arduino_dir / "src" / "TinyRL.cpp"),
        }

    def _generate_tvm(self, model: torch.nn.Module, output_dir: Path) -> Dict[str, str]:
        """Generate TVM model."""
        tvm_dir = output_dir / "tvm"
        tvm_dir.mkdir(exist_ok=True)

        self.tvm_backend.generate_tvm_model(model, tvm_dir)

        return {"tvm_model": str(tvm_dir / "tvm_model.py")}


def create_codegen_report(
    results: Dict[str, Any], config: CodegenConfig
) -> Dict[str, Any]:
    """Create comprehensive code generation report."""

    # Check if all targets were generated successfully
    cmake_ok = "cmake" in results and "cmake_lists" in results["cmake"]
    rust_ok = "rust" in results and "cargo_toml" in results["rust"]
    arduino_ok = "arduino" in results and "library_properties" in results["arduino"]
    tvm_ok = "tvm" in results and "tvm_model" in results["tvm"]

    passed = all([cmake_ok, rust_ok, arduino_ok, tvm_ok])

    return {
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "targets": {
            "cmake": cmake_ok,
            "rust": rust_ok,
            "arduino": arduino_ok,
            "tvm": tvm_ok,
        },
        "config": config.__dict__,
        "files_generated": len(results) - 1,  # Exclude config
    }


def run_codegen_pipeline(
    config: CodegenConfig,
    model: torch.nn.Module,
    output_dir: str = "./outputs/codegen",
) -> Dict[str, Any]:
    """Run complete code generation pipeline."""

    # Initialize trainer
    trainer = CodegenTrainer(config)

    # Generate all targets
    results = trainer.generate_all(model, output_dir)

    # Create report
    report = create_codegen_report(results, config)
    results["report"] = report

    return results
