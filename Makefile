.PHONY: install dev lint fmt test demo clean verify verify-llm verify-run verify-loop verify-quality evalset quickstart run-llm

quickstart:
	bash scripts/quickstart.sh

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
	researchops run "demo topic" --mode fast --allow-net false --llm none --sources demo

clean:
	rm -rf runs/ runs_batch/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

verify:
	python scripts/verify_repo.py

verify-llm:
	python scripts/verify_llm_path.py $(RUN)

verify-run:
	python scripts/verify_run_integrity.py $(RUN)

verify-loop:
	python scripts/verify_no_infinite_rollback.py $(RUN)

verify-quality:
	python scripts/verify_research_quality.py $(RUN)

evalset:
	python scripts/run_evalset.py

run-llm:
	@test -n "$(TOPIC)" || { echo "Usage: make run-llm TOPIC='your topic' [LLM_ARGS='--llm openai_compat ...']"; exit 1; }
	researchops run "$(TOPIC)" --mode deep --sources hybrid --retrieval hybrid --graph --judge $(LLM_ARGS)
