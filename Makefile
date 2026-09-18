.PHONY: check format lint format-check format-markdown format-markdown-check typecheck test clean help

# Default target: show help when you just run `make`
help:
	@echo "Available targets:"
	@echo "  make check                  Run all CI checks (lint, format-check, markdown format-check, typecheck, test)"
	@echo "  make format                 Auto-fix formatting and lint issues"
	@echo "  make lint                   Run ruff linter"
	@echo "  make format-check           Verify Python files are formatted (does not modify files)"
	@echo "  make format-markdown        Auto-fix markdown formatting using preview mode"
	@echo "  make format-markdown-check  Verify markdown files are formatted using preview mode"
	@echo "  make typecheck              Run mypy"
	@echo "  make test                   Run pytest"
	@echo "  make clean                  Remove caches and build artifacts"

# run everytime I edit the codebase
check: lint format format-check format-markdown-check typecheck test

# auto-fix what can be auto-fixed; use before committing
format:
	# format Python files
	ruff format .

	# lint and clean up my files
	ruff check . --fix

# format markdown with preview mode only where needed
format-markdown:
	find . -name "*.md" -print0 | xargs -0 ruff format --preview

# verify markdown formatting without modifying files
format-markdown-check:
	find . -name "*.md" -print0 | xargs -0 ruff format --check --preview

# individual checks below

# lint and clean up my files
lint:
	ruff check .

# verify formatting without modifying files
format-check:
	ruff format --check .

# static type checker
typecheck:
	mypy edge_ir

# run my tests!
test:
	python -m pytest

# remove caches and build artifacts
clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf *.egg-info build dist
	find . -type d -name __pycache__ -exec rm -rf {} +