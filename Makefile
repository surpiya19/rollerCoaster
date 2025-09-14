# --- VARIABLES ---
PYTHON = python3
SRC = scripts/coaster_analysis.py
TESTS = tests/
COV_REPORT = html

# --- TASKS ---

install:
	$(PYTHON) -m pip install --upgrade pip && \
	$(PYTHON) -m pip install -r requirements.txt

format:
	black $(SRC) $(TESTS)

lint:
	flake8 $(SRC) $(TESTS)

test:
	$(PYTHON) -m pytest -vv --cov=scripts --cov-report=term-missing tests/

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov

all: install format lint test
