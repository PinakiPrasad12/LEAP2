# Contributing to LEAP

Thank you for your interest in contributing to LEAP (Learning Expert Adaptation & Pruning)! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Workflow](#development-workflow)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and professional in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Make your changes
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Git

### Installation

1. Clone your fork:
```bash
git clone https://github.com/yourusername/LEAP_Code.git
cd LEAP_Code
```

2. Create a virtual environment:
```bash
python -m venv leap_env
source leap_env/bin/activate  # On Windows: leap_env\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e .[dev]
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Contributing Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Use meaningful variable and function names

### Code Formatting

We use the following tools for code formatting and linting:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Run these before submitting:

```bash
# Format code
black leap/ examples/ tests/

# Check linting
flake8 leap/ examples/ tests/

# Type checking
mypy leap/
```

### Testing

- Write tests for new functionality
- Ensure all existing tests pass
- Aim for good test coverage

Run tests:

```bash
pytest tests/
```

### Documentation

- Update docstrings for new/modified functions
- Add examples for new features
- Update README.md if needed

## Pull Request Process

1. **Create a feature branch** from `main`:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the guidelines above

3. **Write or update tests** for your changes

4. **Run the test suite** to ensure everything works:
```bash
pytest tests/
black leap/ examples/ tests/
flake8 leap/ examples/ tests/
```

5. **Commit your changes** with a descriptive message:
```bash
git commit -m "Add feature: description of your changes"
```

6. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

7. **Create a pull request** on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - Screenshots/examples if applicable

### Pull Request Requirements

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] New functionality is tested
- [ ] Documentation is updated
- [ ] No merge conflicts

## Issue Reporting

When reporting issues, please include:

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected vs actual behavior**
4. **Environment details**:
   - Python version
   - PyTorch version
   - CUDA version (if applicable)
   - Operating system
5. **Error messages** (full traceback)
6. **Minimal code example** if possible

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

## Development Workflow

### Branch Naming

- `feature/description`: New features
- `bugfix/description`: Bug fixes
- `docs/description`: Documentation updates
- `refactor/description`: Code refactoring

### Commit Messages

Follow conventional commit format:

- `feat: add new pruning algorithm`
- `fix: resolve memory leak in routing agent`
- `docs: update installation instructions`
- `test: add unit tests for evaluation metrics`
- `refactor: simplify model loading logic`

### Release Process

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and publish to PyPI

## Areas for Contribution

We welcome contributions in the following areas:

### High Priority
- **New MoE architectures**: Implement support for additional models
- **Evaluation metrics**: Add task-specific evaluation methods
- **Optimization algorithms**: Improve pruning and routing strategies
- **Documentation**: Tutorials, examples, and API documentation

### Medium Priority
- **Performance optimizations**: Speed up training and inference
- **Distributed training**: Multi-GPU and multi-node support
- **Visualization tools**: Better analysis and debugging tools
- **Integration**: Support for popular ML frameworks

### Low Priority
- **Code cleanup**: Refactoring and optimization
- **Testing**: Increase test coverage
- **CI/CD**: Improve automation and testing

## Getting Help

If you need help or have questions:

1. Check existing [issues](https://github.com/yourusername/LEAP_Code/issues)
2. Create a new issue with the `question` label
3. Join our community discussions
4. Read the documentation

## Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Project documentation

Thank you for contributing to LEAP! 🚀
