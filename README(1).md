# LITTLE — Project Documentation

This directory contains the initial project-definition documents:

- `prd.md` — product requirements and scope
- `architecture.md` — proposed system architecture
- `spec.md` — technical specification
- `coreidea.md` — central research hypothesis and design philosophy
- `implementationplan.md` — staged implementation plan
- `todo.md` — actionable task list

Stack decision:

- Python-first
- uv for project/dependency management
- Ruff for lint/format
- Pyrefly for static typing
- pytest for tests
- PyTorch for neural components when needed
- SQLite for initial persistent memory
- Rust + PyO3 + Maturin for profiled performance-critical components

The first prototype should be small, CPU-capable, inspectable, and measurable.
