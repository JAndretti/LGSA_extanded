ENV_FILE ?= .env

train:
	uv run --env-file $(ENV_FILE) python -m src.train

sweep:
	uv run --env-file $(ENV_FILE) python -m src.sweep

.PHONY: train sweep
