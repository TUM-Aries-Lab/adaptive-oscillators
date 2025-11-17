SHELL := /bin/bash

init:
	python3 -m venv .venv
	poetry install
	poetry pre-commit install
	poetry env info
	@echo "Created virtual environment"

test:
	poetry run pytest --cov=src/adaptive_oscillator --cov-report=term-missing --no-cov-on-fail --cov-report=xml --cov-fail-under=90
	rm .coverage

lint:
	poetry run ruff format
	poetry run ruff check --fix

typecheck:
	poetry run mypy src/ tests/ --ignore-missing-imports --disable-error-code=call-overload

format:
	make lint
	make typecheck
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

app:
	poetry run python src/__main__.py --log-dir data/walk_4 --plot
