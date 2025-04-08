# Contributing to Spin-Based Neural Computation Framework

Thank you for your interest in contributing to the Spin-Based Neural Computation Framework! This document provides guidelines and instructions for contributing to this project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct, which promotes a respectful and inclusive environment.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Set up the development environment** as described in the [Development Setup](#development-setup) section
4. **Create a branch** for your work

## How to Contribute

### Reporting Bugs

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with the following information:
   - Clear and descriptive title
   - Detailed steps to reproduce the bug
   - Expected vs. actual behavior
   - Screenshots or code snippets if applicable
   - Environment details (OS, compiler version, etc.)
   - Any relevant error messages or logs

### Suggesting Enhancements

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with the following information:
   - Clear and descriptive title
   - Detailed description of the proposed enhancement
   - Justification: why this enhancement would be valuable
   - Potential implementation approach (optional)
   - Any references or examples (optional)

### Pull Requests

1. **Update your fork** with the latest changes from the main repository
2. **Create a branch** with a descriptive name related to your change
3. **Make your changes**, following our [Coding Standards](#coding-standards)
4. **Add tests** for your changes to ensure functionality
5. **Run tests** to make sure your changes don't break existing functionality
6. **Update documentation** if needed
7. **Submit a pull request** with the following information:
   - Reference to the related issue(s)
   - Description of the changes made
   - Any notes on implementation choices
   - Checklist of completed items (tests, documentation, etc.)

## Development Setup

To set up your development environment:

1. **Install dependencies**:
   - C compiler (GCC or Clang)
   - Make build system
   - SDL2 (for visualization)

2. **Build from source**:
   ```bash
   make all_versions
   ```

3. **Run tests**:
   ```bash
   make test
   ```

## Coding Standards

- **Code Style**:
  - Use 4 spaces for indentation
  - Follow K&R style braces
  - Use descriptive variable and function names
  - Keep functions focused and reasonably sized

- **Comments**:
  - Add comments for complex logic and non-obvious behavior
  - Document functions with purpose, parameters, and return values
  - Use `//` for single-line comments and `/* */` for multi-line comments
  - Include references to academic papers or mathematical formulas where appropriate

- **Error Handling**:
  - Always check return values from functions that might fail
  - Use consistent error reporting mechanism
  - Free allocated resources in error paths

## Testing Guidelines

- **Test Coverage**:
  - Add tests for new features
  - Ensure tests cover both normal and edge cases
  - Include validation against known analytical solutions where possible

- **Test Types**:
  - Unit tests for individual functions
  - Integration tests for module interactions
  - Performance tests for critical paths
  - Numerical stability tests for computational functions

## Documentation Guidelines

- **Code Documentation**:
  - Document all public functions in header files
  - Explain parameters and return values
  - Describe any side effects or dependencies

- **User Documentation**:
  - Update README.md for significant changes
  - Provide examples for new features
  - Keep documentation in the `docs/` directory up-to-date
  - Include references to relevant research papers or educational materials

Thank you for contributing to the Spin-Based Neural Computation Framework!
