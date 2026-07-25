PYTHON ?= .venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help py-test py-test-deps

help:  ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

py-test-deps:  ## Install the test dependencies into the project venv
	uv pip install --python $(PYTHON) pytest

py-test:  ## Run the pytest suite (pass extra flags via ARGS="-k foo")
	@$(PYTHON) -c "import pytest" >/dev/null 2>&1 || $(MAKE) py-test-deps
	$(PYTHON) -m pytest $(ARGS)
