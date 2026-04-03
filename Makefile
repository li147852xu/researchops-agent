.PHONY: install dev lint fmt test demo clean verify verify-llm verify-run verify-loop verify-quality evalset quickstart run check web api

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

check: lint test

demo:
	@echo "Demo requires LLM API key. Set OPENAI_API_KEY or use --llm-api-key."
	@echo "Research:  researchops run 'demo topic' --app research --mode fast --sources demo"
	@echo "Market:    researchops run 'NVDA competitive position' --app market --ticker NVDA"

web:
	pip install -e ".[web]" && researchops web

api:
	pip install -e ".[api]" && researchops api

run:
	@test -n "$(TOPIC)" || { echo "Usage: make run TOPIC='your topic' [APP=research] [LLM_ARGS='--llm openai_compat ...']"; exit 1; }
	researchops run "$(TOPIC)" --app $(or $(APP),research) --mode deep --sources hybrid $(LLM_ARGS)

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
