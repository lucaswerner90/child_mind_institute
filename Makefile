# Define the virtual environment directory
VENV := .venv

.PHONY: all create_venv install_requirements run lint format test

all: create_venv install_requirements download_data mlflow

create_venv:
	@python3 -m venv $(VENV)

install_requirements: create_venv
	@$(VENV)/bin/pip install --upgrade pip
	@$(VENV)/bin/pip install -r requirements.txt

download_data:
	mkdir data
	cd data
	@$(VENV)/bin/kaggle competitions download -c child-mind-institute-problematic-internet-use
	unzip child-mind-institute-problematic-internet-use.zip
	rm child-mind-institute-problematic-internet-use.zip
	cd ..
run:
	@$(VENV)/bin/python main.py

mlflow:
	@$(VENV)/bin/mlflow server

jupyter:
	@$(VENV)/bin/jupyter notebook

lint:
	@$(VENV)/bin/flake8 .

format:
	@$(VENV)/bin/black .

test:
	@$(VENV)/bin/pytest