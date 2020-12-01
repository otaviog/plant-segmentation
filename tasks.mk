about:
	@echo "Project maintaining tasks."

pylint:
	python3 -m pylint plantseg ./workflow.py

pep8:
	python3 -m autopep8 --recursive --in-place module

