# Makefile for chest_xray_sim testing

.PHONY: test test-quick test-chexpert test-coverage help setup kernel jupyter

# Run all tests with pytest (verbose)
test:
	uv run pytest chest_xray_sim/tests -v -s

# Run all tests with pytest (quick)
test-quick:
	uv run pytest chest_xray_sim/tests

# Run CheXpert tests with pytest
test-chexpert:
	uv run pytest chest_xray_sim/tests/test_chexpert.py -v -s

# Run tests with coverage
test-coverage:
	uv run pytest chest_xray_sim/tests --cov=chest_xray_sim --cov-report=html --cov-report=term-missing

# Run tests with specific markers
test-data-loading:
	uv run pytest chest_xray_sim/tests -m data_loading -v

# Run tests in parallel (if pytest-xdist is installed)
test-parallel:
	uv run pytest chest_xray_sim/tests -n auto

# Bootstrap the full development environment (Python check, uv, deps, kernel)
setup:
	bash setup.sh

# Register the project venv as a named Jupyter kernel
kernel:
	uv run python -m ipykernel install --user --name chest-xray-sim --display-name "Python (chest-xray-sim)"

# Launch JupyterLab using the project venv
jupyter:
	uv run jupyter lab

# Show help
help:
	@echo "Available commands:"
	@echo "  make test                    - Run all tests with verbose output"
	@echo "  make test-quick              - Run all tests without verbose output"
	@echo "  make test-chexpert           - Run only CheXpert tests"
	@echo "  make test-coverage           - Run tests with coverage report"
	@echo "  make test-data-loading       - Run only data loading tests"
	@echo "  make test-parallel           - Run tests in parallel"
	@echo "  make setup                   - Bootstrap environment (uv sync + kernel register)"
	@echo "  make kernel                  - Register venv as 'chest-xray-sim' Jupyter kernel"
	@echo "  make jupyter                 - Launch JupyterLab with project venv"
	@echo "  make help                    - Show this help message"
