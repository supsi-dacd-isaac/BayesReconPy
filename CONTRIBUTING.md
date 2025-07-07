# Contributing to BayesReconPy 🧠

First off, thanks for considering contributing! These guidelines will help you get started.

---

## 📚 Table of Contents

- [🧰 Getting Started](#-getting-started)
  - [Requirements](#requirements)
  - [Code Style & Formatting](#code-style--formatting)
- [🧪 Running Tests](#-running-tests)
- [🚀 Pull Request Workflow](#-pull-request-workflow)
  - [PR Guidelines](#pr-guidelines)
- [🔧 Reporting Bugs & Requesting Features](#-reporting-bugs--requesting-features)
- [📐 Code Guidelines](#-code-guidelines)
- [📚 Documentation](#-documentation)
- [✅ Review & Merge Process](#-review--merge-process)
- [🏷️ Versioning & Releases](#️-versioning--releases)
- [🤝 Code of Conduct](#-code-of-conduct)
- [🙏 Thank You!](#-thank-you)

---

## 🧰 Getting Started

### Requirements

Make sure you have:

- Python 3.10+ (preferably via `venv` or `conda`)
- Core dependencies installed via:

  ```bash
  pip install -e .[dev]
  ```

## 🎨 Code Style & Formatting

We enforce consistent and readable code using automatic tools.

### 📦 Required Tools

Make sure you have the development dependencies installed:

```bash
pip install -e .[dev]
```

Then install pre-commit hooks (run once):

```bash
pip install pre-commit
pre-commit install
```

These tools ensure that code is formatted and linted before each commit.

---

### ✨ Formatting with `black`

We use [`black`](https://black.readthedocs.io/) for code formatting:

```bash
black .
```

`black` will automatically reformat your code to follow a consistent style.

---

### 🧹 Linting with `flake8`

We use [`flake8`](https://flake8.pycqa.org/) for static analysis and code linting:

```bash
flake8 bayesreconpy
```

This helps catch syntax errors, unused variables, and style violations.

---

### ⚙️ Recommended Configuration

You can add the following configuration to your `pyproject.toml` or `.flake8` file to customize checks:

#### `.flake8`

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
```

#### `pyproject.toml`

```toml
[tool.black]
line-length = 88
target-version = ['py38']
```

---

### 💡 Tip

All of this is automatically run if you commit via `git` and have `pre-commit` installed. To manually run all hooks:

```bash
pre-commit run --all-files
```

---

Following these steps helps ensure code quality and consistency across contributions.

---

## 🧪 Running Tests

Use `pytest` to run the test suite:

```bash
pytest
```

For test coverage reports:

```bash
pytest --cov=bayesreconpy
```

---

## 🚀 Pull Request Workflow

1. Fork the repository and clone your fork.
2. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Implement your changes and add tests.
4. Ensure all checks pass (`black`, `flake8`, and `pytest`).
5. Commit your changes:

   ```bash
   git add .
   git commit -m "feat: Add your concise feature summary"
   ```

6. Push your branch and open a Pull Request against the `main` branch.

### PR Guidelines

- **Title** should begin with: `feat:`, `fix:`, `refactor:`, or `docs:`
- **Description** must explain:
  - Motivation
  - Summary of changes
  - Related issues (e.g., `Closes #123`)
- Include **tests** for new features or bug fixes.

---

## 🔧 Reporting Bugs & Requesting Features

- Search [issues](https://github.com/supsi-dacd-isaac/BayesReconPy/issues) to avoid duplicates.
- When filing a bug, include:
  - Minimal reproduction example
  - Expected vs actual behavior
  - Environment (OS, Python version, BayesReconPy version)
- For features, explain your use-case and benefits.

---

## 📐 Code Guidelines

- Use **type hints** and **docstrings** for functions and classes.
- Follow consistent naming and functional decomposition.
- Write **tests** for each feature.
- Add examples in the `examples/` or `tutorials/` folder when relevant.

---

## 📚 Documentation

- Update docstrings and the README when needed.
- Add structured Jupyter notebooks or markdown examples for complex workflows.

---

## ✅ Review & Merge Process

- CI will run checks on each PR.
- At least one approving review is required.
- You may be asked to rebase or sync with `main`.
- Maintainers will merge once everything is in order.

---

## 🏷️ Versioning & Releases

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** for breaking changes
- **MINOR** for new features
- **PATCH** for fixes

Each release is tagged on GitHub.

---

## 🤝 Code of Conduct

By contributing, you agree to follow the [Contributor Covenant](https://www.contributor-covenant.org/). Please be respectful and constructive.

---

## 🙏 Thank You!

We appreciate your time and effort to improve **BayesReconPy**! Your contributions help build a better tool for probabilistic forecast reconciliation.
