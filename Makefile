.PHONY: help install run lint format test test-frontend precommit-install precommit-run \
        ingest chroma-up chroma-down chroma-logs chroma-clean clean

PORT ?= 8000
ARGS ?=

help:
	@echo "make install            install project + dev dependencies (uv sync)"
	@echo "make run                run the FastAPI backend with auto-reload"
	@echo "make lint               check code style with ruff"
	@echo "make format             auto-format code with ruff"
	@echo "make test               run the test suite with pytest"
	@echo "make test-frontend      run the plain-Node frontend regression tests"
	@echo "make precommit-install  install git pre-commit hooks"
	@echo "make precommit-run      run pre-commit hooks against all files"
	@echo "make ingest ARGS=...    run the batch podcast ingestion script"
	@echo "make chroma-up          start the ChromaDB server via docker compose"
	@echo "make chroma-down        stop the ChromaDB server"
	@echo "make chroma-logs        tail the ChromaDB server logs"
	@echo "make chroma-clean       delete all ingested chunks from ChromaDB"
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

test-frontend:
	for f in tests/frontend/test_*.js; do node "$$f" || exit 1; done

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

# Drops the transcription_chunks collection so the next ingest starts fresh.
chroma-clean:
	if [ -f backend/.env ]; then set -a; . backend/.env; set +a; fi; uv run python -c "\
	from backend.etl.embeddings import get_chroma, COLLECTION_NAME; \
	c = get_chroma(); \
	c.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [col.name for col in c.list_collections()] else None; \
	print(f'Deleted collection: {COLLECTION_NAME}')"

clean:
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
