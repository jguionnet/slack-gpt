.DEFAULT_GOAL := help

SHELL = /bin/bash

export PYTHONPATH ?= $(PROJECT_ROOT)

help: ## Shows the help
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
        awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''

config: ## update and load requirements.txt
	pip3 install pipreqs 
	python3 -m  pipreqs.pipreqs . --force 
	python3 -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

install_dep: ## load requirements.txt
	pip3 install --upgrade pip
	pip3 cache purge
	python3 -m pip install -r requirements.txt 

install_dep4: ## load requirements.txt
	python3 -m pip install -r requirements4.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

load: ## Run 
	python3 IndexAndQuery.py -bs data 

query: ## Run 
	python3 IndexAndQuery.py -lq data

webapp: 
	python3 LoadAndQueryWeb.py
