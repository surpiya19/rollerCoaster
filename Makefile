# --- VARIABLES ---
PYTHON = python3
SRC_ANALYSIS = scripts/coaster_analysis.py
SRC_MLMODEL = scripts/mlmodel.py
TESTS = tests/
COV_REPORT = html
ROOT = $(shell pwd)

# --- TASKS ---

install:
	$(PYTHON) -m pip install --upgrade pip && \
	$(PYTHON) -m pip install -r requirements.txt

format:
	black scripts $(TESTS)

lint:
	flake8 scripts $(TESTS)

test:
	PYTHONPATH=$(ROOT) $(PYTHON) -m pytest -vv --cov=scripts --cov-report=term-missing --disable-warnings $(TESTS)

run-analysis:
	PYTHONPATH=$(ROOT) $(PYTHON) $(SRC_ANALYSIS)

run-ml:
	PYTHONPATH=$(ROOT) $(PYTHON) $(SRC_MLMODEL)

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov plots subsets

all: install format lint test
