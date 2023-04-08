PROJECT_NAME=llama_server

test:
	python3 -bb -m pytest --cov-config setup.cfg test -s -n auto

clean:
	find . -name \*.pyc -delete
	find . -name __pycache__ -delete
	rm -rf dist/ build/ *.egg-info/

lint:
	pre-commit run -a

mypy:
	mypy ${PROJECT_NAME}

debug:
	python3 setup.py build
	pip install -e .

release: clean
	python3 setup.py sdist
	python3 setup.py bdist_wheel

publish: clean
	python3 setup.py sdist
	twine check --strict dist/*
	twine upload --verbose dist/*

.PHONY: test clean lint mypy debug release publish
