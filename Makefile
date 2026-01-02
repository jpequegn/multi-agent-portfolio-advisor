.PHONY: install dev test lint format typecheck clean run docker-up docker-down help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install     - Install production dependencies"
	@echo "  make dev         - Install all dependencies (including dev)"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linter (ruff)"
	@echo "  make format      - Format code (ruff)"
	@echo "  make typecheck   - Run type checker (mypy)"
	@echo "  make check       - Run all checks (lint, typecheck, test)"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make run         - Run the API server"
	@echo "  make docker-up   - Start Docker services"
	@echo "  make docker-down - Stop Docker services"

# Dependencies
install:
	uv sync

dev:
	uv sync --all-extras

# Testing
test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=term-missing

# Code Quality
lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy src

check: lint typecheck test

# Development
run:
	uv run uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/
