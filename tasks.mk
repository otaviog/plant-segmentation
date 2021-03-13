about:
	@echo "Project maintaining tasks."

pylint:
	python3 -m pylint plantseg ./workflow.py

pep8:
	python3 -m autopep8 --recursive --in-place module

ci-build:
	act -P ubuntu-18.04=nektos/act-environments-ubuntu:18.04 -j build
