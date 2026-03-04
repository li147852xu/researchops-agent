.PHONY: install dev lint fmt test demo clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest -v

demo:
	researchops run "demo topic" --mode fast --allow-net false

clean:
	rm -rf runs/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
