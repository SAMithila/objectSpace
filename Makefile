.PHONY: help install install-dev test lint format type-check clean build run

# Default target
help:
	@echo "Smart Desk Monitor - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install package in production mode"
	@echo "  make install-dev  Install package with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         Run tests with pytest"
	@echo "  make lint         Run linting (flake8)"
	@echo "  make format       Format code (black + isort)"
	@echo "  make type-check   Run type checking (mypy)"
	@echo "  make check        Run all checks (lint + type-check + test)"
	@echo ""
	@echo "Build:"
	@echo "  make build        Build distribution packages"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Run:"
	@echo "  make run INPUT=path/to/video.mp4  Process a video"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install || true

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=smart_desk_monitor --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/smart_desk_monitor tests/

format:
	black src/smart_desk_monitor tests/
	isort src/smart_desk_monitor tests/

type-check:
	mypy src/smart_desk_monitor

check: lint type-check test

# Build
build: clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Run the pipeline
run:
ifndef INPUT
	@echo "Usage: make run INPUT=path/to/video.mp4 [OUTPUT=output/]"
	@exit 1
endif
	smart-desk-monitor process $(INPUT) -o $(or $(OUTPUT),output/) -v
