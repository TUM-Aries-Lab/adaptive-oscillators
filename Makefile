SHELL := /bin/bash

init:
	python3 -m venv .venv
	poetry install
	poetry run pre-commit install
	poetry env info
	@echo "Created virtual environment"

lint:
	poetry run ruff format
	poetry run ruff check --fix

typecheck:
	poetry run mypy src/ tests/ --ignore-missing-imports --disable-error-code=call-overload

test:
	poetry run pytest --cov=src/adaptive_oscillator --cov-report=term-missing --no-cov-on-fail --cov-report=xml --cov-fail-under=75
	rm .coverage

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
	rm .coverage
	rm coverage.xml

update:
	poetry cache clear pypi --all
	poetry update

help:
	poetry run python -m src.adaptive_oscillator --help

tree:
	poetry run python repo_tree.py --update-readme
