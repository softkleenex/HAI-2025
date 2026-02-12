# Product Guidelines

## Tone and Style
- **Professional and Concise:** Documentation and comments should be clear, direct, and free of unnecessary jargon. Focus on "what" and "why" rather than "how" unless it's complex.
- **Language:** English (US) is the standard language for all code and documentation.

## Coding Standards
- **Python:** Adhere to **PEP 8** standards for code formatting.
- **Type Hinting:** Mandatory use of type hints for function signatures to ensure code clarity and enable static analysis.
- **Docstrings:** Use **Google Style** docstrings for all modules, classes, and functions.
- **Modularity:** Emphasize clean, modular code architecture to facilitate testing and maintenance.

## Version Control
- **Commit Messages:** Follow the Conventional Commits specification (e.g., `feat: add new model architecture`, `fix: resolve data loading issue`).
- **Branching Strategy:** Use feature branches for new development and pull requests for merging into the main branch.

## Reproducibility
- **Configuration:** All hyperparameters and file paths must be configurable via YAML files.
- **Random Seeds:** Ensure reproducibility by setting random seeds for all stochastic operations (NumPy, PyTorch, etc.).
