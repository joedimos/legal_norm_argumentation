.PHONY: test lint security migrate run compose-up compose-down

test:
	python -m pytest -q

lint:
	ruff check src tests

security:
	bandit -q -r src/service
	pip-audit

migrate:
	alembic upgrade head

run:
	uvicorn src.service.app:app --host 127.0.0.1 --port 8000

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down
