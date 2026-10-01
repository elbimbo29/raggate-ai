.PHONY: help dev test lint fmt check clean

help:
	@echo "RAGGate AI - common tasks"
	@echo "  make dev    : boot the FastAPI server with reload"
	@echo "  make test   : run the test suite"
	@echo "  make lint   : ruff check"
	@echo "  make fmt    : ruff format + import sort"
	@echo "  make check  : lint + test"
	@echo "  make clean  : remove caches and local artifacts"

dev:
	uv run raggate

test:
	uv run pytest -v

lint:
	uv run ruff check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f data/*.sqlite reports/*.json