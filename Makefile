.PHONY: dev format help
.DEFAULT_GOAL := help
include .env

rds_port = 5432

help:
	@awk -F ':.*?## ' '/^[a-zA-Z]/ && NF==2 {printf "\033[36m  %-25s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Setup dev environment
	poetry env use python3.10
	poetry update
	poetry install
	poetry run pre-commit install
