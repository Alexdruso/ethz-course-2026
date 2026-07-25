PYTHON ?= .venv/bin/python
RUFF ?= uvx ruff
TY ?= uvx ty
PY_SRC ?= src tests

.DEFAULT_GOAL := help
.PHONY: help py-test py-test-deps py-fmt py-static

help:  ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

py-test-deps:  ## Install the test dependencies into the project venv
	uv pip install --python $(PYTHON) pytest

py-test:  ## Run the pytest suite (pass extra flags via ARGS="-k foo")
	@$(PYTHON) -c "import pytest" >/dev/null 2>&1 || $(MAKE) py-test-deps
	$(PYTHON) -m pytest $(ARGS)

py-fmt:  ## Auto-fix lint and formatting issues with ruff (check --fix, then format)
	-$(RUFF) check --fix $(PY_SRC)
	$(RUFF) format $(PY_SRC)

py-static:  ## Run astral's ty type checker
	$(TY) check --python $(PYTHON) $(PY_SRC)
