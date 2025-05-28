.PHONY: all format lint test tests integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_package: PYTHON_FILES=src

lint lint_package:
	[ "$(PYTHON_FILES)" = "" ] || ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format:
	[ "$(PYTHON_FILES)" = "" ] || ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || ruff check --select I --fix $(PYTHON_FILES)

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'