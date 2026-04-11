# Contributing to DrugLab

Thank you for your interest in contributing to DrugLab. We welcome contributions that help improve the project, whether through code, documentation, bug reports, or feature suggestions.

This document outlines the guidelines for contributing, reporting issues, and proposing new features.

## How to Contribute

We follow a standard GitHub workflow using feature branches and Pull Requests. Please do not push changes directly to the `main` branch.

### 1. Create a Branch

Start by creating a new branch from `main` for your work. Use a clear prefix (`feature/`, `bugfix/`, `docs/`, `refactor/`) followed by a descriptive name.

```bash
git checkout main
git pull
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

Implement your changes with the following in mind:

- Keep commits focused and atomic.
- Write clear, descriptive commit messages.
- When adding new pipeline blocks or table classes, follow the existing architecture (especially functional immutability and the `HistoryEntry` audit log).

### 3. Open a Pull Request

Push your branch and open a Pull Request targeting `main`.

```bash
git push origin feature/your-feature-name
```

**Tips for a good PR:**
- Use a clear, descriptive title.
- Explain the changes in the description, including the reasoning behind your approach and any related issues it addresses.
- Be open to feedback; maintainers or reviewers may request adjustments before merging.

## Reporting Bugs

If you discover a bug, please open an Issue on GitHub with the following details:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your operating system and Python environment information

## Code Style

I'm still figuring this part out, so yeah :D