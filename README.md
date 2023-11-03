## Setup dev environment
poetry env use python3.10
poetry update
poetry install
poetry run pre-commit install
