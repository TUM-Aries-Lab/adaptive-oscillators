SHELL := /bin/bash

init:
	python3 -m venv .venv
	poetry install
	pre-commit install
	poetry env info
	@echo "Created virtual environment"

test:
	poetry run pytest --cov=src/iit_rehab --cov-report=term-missing --no-cov-on-fail

format:
	ruff format --exclude src/raspberrypi_code
	ruff check --fix --exclude src/raspberrypi_code
	poetry run mypy src/ tests/ --ignore-missing-imports  --exclude src/raspberrypi_code

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf juninit-pytest.xml
	rm -rf logs/*
	find . -name ".coverage*" -delete
	find . -name --pycache__ -exec rm -r {} +

update:
	poetry cache clear pypi --all
	poetry update
