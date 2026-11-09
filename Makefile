.PHONY: install test run all

install:
	pip install -e ".[dev]"

test:
	pytest -q

run:
	python -m prp.run --config config.yaml

all: install test run
