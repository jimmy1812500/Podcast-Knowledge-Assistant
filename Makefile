.PHONY: help install run lint format test precommit-install precommit-run \
        ingest chroma-up chroma-down chroma-logs clean

PORT ?= 8000
ARGS ?=

help:
	@echo "make install            install project + dev dependencies (uv sync)"
	@echo "make run                run the FastAPI backend with auto-reload"
	@echo "make lint               check code style with ruff"
	@echo "make format             auto-format code with ruff"
	@echo "make test               run the test suite with pytest"
	@echo "make precommit-install  install git pre-commit hooks"
	@echo "make precommit-run      run pre-commit hooks against all files"
	@echo "make ingest ARGS=...    run the batch podcast ingestion script"
	@echo "make chroma-up          start the ChromaDB server via docker compose"
	@echo "make chroma-down        stop the ChromaDB server"
	@echo "make chroma-logs        tail the ChromaDB server logs"
	@echo "make clean              remove caches and bytecode"

install:
	uv sync

run:
	uv run uvicorn backend.main:app --reload --port $(PORT)

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

precommit-install:
	uv run pre-commit install

precommit-run:
	uv run pre-commit run --all-files

# Example: make ingest ARGS="--since 2026-06-01 "
ingest:
	if [ -f backend/.env ]; then set -a; . backend/.env; set +a; fi; uv run python -m backend.ingest_batch $(ARGS)

# Example: make ingest_demo"
ingest_demo:
	if [ -f backend/.env ]; then set -a; . backend/.env; set +a; fi; uv run python -m backend.ingest_batch_demo

chroma-up:
	docker compose up -d chroma

chroma-down:
	docker compose down

chroma-logs:
	docker compose logs -f chroma

clean:
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
