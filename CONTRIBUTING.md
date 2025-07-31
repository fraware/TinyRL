# Contributing to TinyRL

Thank you for your interest in contributing to TinyRL! This document provides guidelines for contributing to the project.

## Quick Start

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/fraware/TinyRL.git
   cd TinyRL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests**
   ```bash
   pytest tests/
   ```

## Development Workflow

### 1. Issue Tracking
- Create an issue for any bug reports or feature requests
- Use appropriate labels: `bug`, `enhancement`, `documentation`, etc.
- Reference related issues in your PR description

### 2. Branch Strategy
- Create feature branches from `main`
- Use descriptive branch names: `feature/quantization-engine`, `fix/memory-leak`
- Keep branches focused on single features/fixes

### 3. Code Standards

#### Python Code Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all function parameters and return values
- Maximum line length: 88 characters (Black formatter)
- Use f-strings for string formatting

#### Example:
```python
from typing import Optional, Tuple, List
import numpy as np
import torch


def quantize_weights(
    weights: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to specified bit precision.
    
    Args:
        weights: Input weights tensor
        bits: Target bit precision (default: 8)
        symmetric: Use symmetric quantization (default: True)
    
    Returns:
        Tuple of (quantized_weights, scales)
    """
    # Implementation here
    pass
```

#### C/C++ Code Style
- Follow [MISRA C:2023](https://www.misra.org.uk/) guidelines
- Use snake_case for variables and functions
- Use UPPER_CASE for constants and macros
- Maximum line length: 100 characters

#### Example:
```c
#include <stdint.h>
#include <stdbool.h>

#define MAX_WEIGHTS 1024
#define QUANTIZATION_SCALE 0.01f

typedef struct {
    int8_t weights[MAX_WEIGHTS];
    float scales[MAX_WEIGHTS];
    uint16_t num_weights;
} quantized_policy_t;

bool tinyrl_quantize_policy(
    const float* input_weights,
    uint16_t num_weights,
    quantized_policy_t* output
);
```

### 4. Testing Requirements

#### Unit Tests
- Write tests for all new functionality
- Aim for >95% code coverage
- Use descriptive test names
- Test both success and failure cases

#### Example:
```python
import pytest
import torch
from tinyrl.quantization import quantize_weights


def test_quantize_weights_symmetric():
    """Test symmetric quantization of weights."""
    weights = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    quantized, scales = quantize_weights(weights, bits=8, symmetric=True)
    
    assert quantized.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert torch.all(quantized >= -128)
    assert torch.all(quantized <= 127)


def test_quantize_weights_invalid_bits():
    """Test quantization with invalid bit precision."""
    weights = torch.randn(10)
    
    with pytest.raises(ValueError, match="bits must be between 1 and 8"):
        quantize_weights(weights, bits=16)
```

#### Integration Tests
- Test end-to-end workflows
- Test on target hardware (when possible)
- Test performance benchmarks

### 5. Documentation

#### Code Documentation
- Use docstrings for all public functions
- Follow [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints and examples

#### API Documentation
- Update API documentation for any changes
- Include usage examples
- Document breaking changes

### 6. Performance Requirements

#### Memory Budgets
- **RAM**: ≤32KB total usage
- **Flash**: ≤128KB total usage
- **Stack**: ≤4KB per function
- **Heap**: ≤28KB dynamic allocation

#### Latency Requirements
- **Inference Time**: ≤5ms (target: ≤1ms)
- **Interrupt Latency**: ≤50µs
- **Memory Access**: ≤100ns per word

### 7. Security Guidelines

#### Code Security
- Validate all inputs
- Use bounds checking
- Avoid buffer overflows
- Sanitize user data

#### Example:
```c
bool tinyrl_validate_input(
    const float* observations,
    uint16_t obs_dim,
    uint16_t max_dim
) {
    if (observations == NULL || obs_dim == 0 || obs_dim > max_dim) {
        return false;
    }
    
    // Check for NaN/Inf values
    for (uint16_t i = 0; i < obs_dim; i++) {
        if (!isfinite(observations[i])) {
            return false;
        }
    }
    
    return true;
}
```

## Development Tools

### Pre-commit Hooks
The following hooks are automatically run on commit:
- `black`: Code formatting
- `isort`: Import sorting
- `flake8`: Linting
- `mypy`: Type checking
- `pytest`: Unit tests

### Manual Checks
```bash
# Format code
black tinyrl/

# Sort imports
isort tinyrl/

# Run linter
flake8 tinyrl/

# Type checking
mypy tinyrl/

# Run tests
pytest tests/ -v --cov=tinyrl

# Security scan
trivy fs .
```

## Pull Request Process

### 1. Before Submitting
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance benchmarks pass
- [ ] Security scan clean

### 2. PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Performance benchmarks pass
- [ ] Hardware testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Closes #123
```

### 3. Review Process
- All PRs require at least 2 approvals
- CI/CD pipeline must pass all checks
- Performance regression tests must pass
- Security scan must be clean

## Architecture Guidelines

### Module Structure
```
tinyrl/
├── __init__.py
├── training/          # Training pipeline
├── quantization/      # Quantization engine
├── runtime/          # MCU runtime
├── verification/     # Formal verification
└── utils/           # Shared utilities
```

### Interface Design
- Use abstract base classes for interfaces
- Provide default implementations
- Support dependency injection
- Maintain backward compatibility

### Error Handling
- Use custom exception classes
- Provide meaningful error messages
- Include recovery suggestions
- Log errors appropriately

## Release Process

### Pre-release Checklist
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Performance benchmarks meet targets
- [ ] Security audit completed
- [ ] License compliance verified
- [ ] Release notes drafted

### Release Steps
1. Create release branch from `main`
2. Update version numbers
3. Run full test suite
4. Generate release artifacts
5. Create GitHub release
6. Deploy documentation

## Community Guidelines

### Communication
- Be respectful and inclusive
- Use clear, professional language
- Provide constructive feedback
- Help newcomers

### Recognition
- Contributors will be listed in CONTRIBUTORS.md
- Significant contributions will be acknowledged in release notes
- Maintainers will be listed in README.md

## Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the docs/ directory for detailed guides

Thank you for contributing to TinyRL!