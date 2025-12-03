# Contributing Guide

Thank you for considering contributing to the Barbados Traffic Analysis project! This guide will help you get started.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [How to Contribute](#how-to-contribute)
6. [Coding Standards](#coding-standards)
7. [Testing Guidelines](#testing-guidelines)
8. [Pull Request Process](#pull-request-process)
9. [Reporting Issues](#reporting-issues)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Maintain professional communication

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of machine learning
- Familiarity with OpenCV (for video features)

### Development Setup

1. **Fork and Clone**
   ```powershell
   git clone https://github.com/your-username/Barbados-Traffic-Analysis-Challenge.git
   cd Barbados-Traffic-Analysis-Challenge-main
   ```

2. **Create Virtual Environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install in Development Mode**
   ```powershell
   pip install -e .
   ```

## Project Structure

```
src/
├── config/          # Configuration modules
├── features/        # Feature extraction modules
├── models/          # Model training and inference
├── pipelines/       # End-to-end workflows
└── utils/           # Utility functions

tests/               # Test suite
benchmarks/          # Performance benchmarks
docs/                # Documentation
```

## How to Contribute

### Types of Contributions

1. **Bug Fixes**: Fix issues in existing code
2. **New Features**: Add new functionality
3. **Documentation**: Improve or add documentation
4. **Performance**: Optimize existing code
5. **Tests**: Add or improve test coverage

### Finding Issues

- Check the [Issues](https://github.com/project/issues) page
- Look for labels: `good first issue`, `help wanted`, `bug`
- Ask in discussions if you're unsure where to start

### Creating a Branch

```powershell
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Grouped (standard, third-party, local)
- **Naming**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_CASE`
  - Private: `_leading_underscore`

### Documentation Standards

All public classes and methods must have docstrings:

```python
def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from input data.
    
    Args:
        data: Input DataFrame with raw data
        
    Returns:
        DataFrame with extracted features
        
    Raises:
        ValueError: If data format is invalid
    """
    pass
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import Dict, List, Optional

def process_data(
    data: pd.DataFrame,
    config: FeatureConfig,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    pass
```

### Code Organization

1. **Imports** at the top (grouped)
2. **Module docstring** after imports
3. **Constants** after docstring
4. **Classes** and **functions** in logical order
5. **Main block** at the end (`if __name__ == "__main__"`)

## Testing Guidelines

### Writing Tests

Create tests in `tests/` directory:

```python
# tests/test_video_features.py

import pytest
from src.features import VideoFeatureExtractor

def test_video_extractor_initialization():
    """Test VideoFeatureExtractor initialization."""
    extractor = VideoFeatureExtractor()
    assert extractor is not None
    assert len(extractor.feature_names) > 0

def test_extract_features_from_video():
    """Test feature extraction from video file."""
    extractor = VideoFeatureExtractor()
    # Test implementation
    pass
```

### Running Tests

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/test_video_features.py

# Run with coverage
pytest --cov=src tests/
```

### Test Coverage

Aim for:
- **Minimum**: 70% code coverage
- **Target**: 85% code coverage
- **Critical paths**: 100% coverage

## Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Run tests**: `pytest`
4. **Check style**: `flake8 src/`
5. **Format code**: `black src/`
6. **Update CHANGELOG.md** if applicable

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] CHANGELOG updated
```

### Review Process

1. Submit PR to `main` branch
2. Automated tests run
3. Code review by maintainers
4. Address feedback
5. Approval and merge

## Reporting Issues

### Bug Reports

Include:
1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Exact steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Python version, package versions
6. **Screenshots/Logs**: If applicable

### Feature Requests

Include:
1. **Problem**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Additional Context**: Any other relevant info

## Development Workflow

### Standard Workflow

```powershell
# 1. Update your fork
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes and commit
git add .
git commit -m "Add feature: description"

# 4. Push to your fork
git push origin feature/my-feature

# 5. Create Pull Request on GitHub
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: Add new temporal feature extractor
fix: Correct video frame counting bug
docs: Update architecture documentation
test: Add tests for model trainer
perf: Optimize video processing pipeline
refactor: Restructure feature extraction module
```

## Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas
- **Pull Requests**: Code contributions

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in documentation

## Questions?

- Check existing documentation
- Search closed issues
- Ask in GitHub Discussions
- Contact maintainers

Thank you for contributing! 🎉
