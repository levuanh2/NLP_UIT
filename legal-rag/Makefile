.PHONY: install lint format test test-cov ingest index corpus-index solve evaluate-baseline submit validate-submission

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

corpus-index:
	python -m app.cli.main build-corpus-index --corpus-zip selected-contexts.zip

solve:
	python -m app.cli.main solve --questions data/questions/public-official.json --train data/questions/train.json

evaluate-baseline:
	python -m app.cli.main evaluate-baseline --train data/questions/train.json

submit:
	python -m app.cli.main submit --questions data/questions/public_test.json --answers data/outputs/internal-results.json

validate-submission:
	python -m app.cli.main validate-submission data/outputs/submission.json --questions data/questions/questions.json
