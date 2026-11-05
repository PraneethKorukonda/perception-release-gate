.PHONY: install test lint demo dashboard study

install:
	pip install -e ".[dev,dashboard]"

test:
	pytest -q

lint:
	ruff check src tests app
	ruff format --check src tests app

demo:
	releasegate demo --out demo

dashboard:
	streamlit run app/dashboard.py -- --report demo/reports/current_vs_candidate

# about 20 minutes on two cores
study:
	releasegate study --out reports/studies.md
