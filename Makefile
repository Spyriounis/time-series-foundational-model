# Data pipeline entry points. Every pull resumes from what is already stored.
# Extra CLI options go in ARGS, e.g. `make pull-oanda-fx ARGS="--only EUR_USD --granularity M5"`.

TSFM := uv run tsfm
ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help pull-all pull-oanda-fx pull-oanda-other pull-oanda-held-out pull-oanda-books pull-etoro pull-fred \
	pull-rates pull-cftc pull-indices pull-supply pull-events synth-train synth-val synth catalog lint test test-live

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "%-18s %s\n", $$1, $$2}'

pull-all: ## Pull every source
	$(TSFM) data pull-all $(ARGS)

pull-oanda-fx: ## G10 FX bars (targets)
	$(TSFM) data pull oanda-fx $(ARGS)

pull-oanda-other: ## OANDA indices, commodities, metals, bonds
	$(TSFM) data pull oanda-other $(ARGS)

pull-oanda-held-out: ## Non-G10 pairs for the zero-shot test, never trained on
	$(TSFM) data pull oanda-held-out $(ARGS)

pull-oanda-books: ## OANDA order and position books
	$(TSFM) data pull oanda-books $(ARGS)

pull-etoro: ## eToro stocks, ETFs, crypto
	$(TSFM) data pull etoro $(ARGS)

pull-fred: ## FRED macro and market series
	$(TSFM) data pull fred $(ARGS)

pull-rates: ## G10 policy rates
	$(TSFM) data pull rates $(ARGS)

pull-cftc: ## CFTC futures positioning
	$(TSFM) data pull cftc $(ARGS)

pull-indices: ## Published risk and supply-chain indices
	$(TSFM) data pull indices $(ARGS)

pull-supply: ## Port transits and petroleum stocks
	$(TSFM) data pull portwatch $(ARGS)
	$(TSFM) data pull eia $(ARGS)

pull-events: ## Release calendar and central-bank meetings
	$(TSFM) data pull fred-calendar $(ARGS)
	$(TSFM) data pull central-banks $(ARGS)

synth-train: ## Synthetic training split
	$(TSFM) synth generate train $(ARGS)

synth-val: ## Synthetic validation split
	$(TSFM) synth generate val $(ARGS)

synth: synth-train synth-val ## Both synthetic splits

catalog: ## Summarise what is stored
	$(TSFM) data catalog

lint: ## Ruff, numpydoc, mypy
	uv run ruff check . && uv run ruff format --check .
	uv run numpydoc lint $$(find src -name '*.py')
	uv run mypy

test: ## Offline test suite
	uv run pytest

test-live: ## Live check of every source (needs .env, about 100 requests)
	uv run pytest -m live -v
