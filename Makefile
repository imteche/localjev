# LocalJev — common tasks
PY ?= python3
PORT ?= 8000

.PHONY: help install install-dev run test bench demo health clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:         ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev:     ## Install runtime + test dependencies
	$(PY) -m pip install -r requirements-dev.txt

run:             ## Start the LocalJev server + dashboard (needs LM Studio running)
	LOCALJEV_PORT=$(PORT) $(PY) -m localjev.server

test:            ## Run the test suite (no LM Studio required)
	$(PY) -m pytest

bench:           ## Run the System One vs LLM benchmark in the terminal (needs LM Studio)
	$(PY) examples/benchmark_cli.py $(if $(MODEL),--model $(MODEL),) $(if $(LIMIT),--limit $(LIMIT),)

demo:            ## Run the CLI triage demo against a running server
	$(PY) examples/triage.py

health:          ## Check LocalJev + LM Studio connectivity
	@curl -s http://localhost:$(PORT)/health | $(PY) -m json.tool || echo "server not running"

clean:           ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; rm -rf .pytest_cache
