.PHONY: install lint format test test-cov ingest index submit validate-submission

install:
	python -m pip install -e .

lint:
	ruff check .
	mypy app

format:
	ruff format .

test:
	pytest

test-cov:
	pytest --cov=app --cov-report=term-missing

ingest:
	python -m app.cli.main ingest

index:
	python -m app.cli.main index

submit:
	# TODO(phase-implementation): provide a real question file before running.
	python -m app.cli.main submit --questions data/questions/public_test.json

validate-submission:
	python -m app.cli.main validate-submission data/outputs/submission.json --questions data/questions/questions.json
