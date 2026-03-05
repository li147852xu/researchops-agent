.PHONY: install dev lint fmt test demo clean verify verify-llm verify-run

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
	researchops run "demo topic" --mode fast --allow-net false --llm none

clean:
	rm -rf runs/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

verify:
	python scripts/verify_repo.py

verify-llm:
	python scripts/verify_llm_path.py $(RUN)

verify-run:
	python scripts/verify_run_integrity.py $(RUN)
