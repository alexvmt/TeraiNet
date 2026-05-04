.PHONY: lint format typecheck test lint-notebooks all

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix
	uv run ruff format .

typecheck:
	uv run ty check src/terainet

test:
	uv run pytest -q

lint-notebooks:
	uv run nbqa ruff --fix .

all: lint format typecheck test
