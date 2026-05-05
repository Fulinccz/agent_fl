.PHONY: help test test-go test-python lint lint-go lint-python build build-go build-python build-frontend up down logs clean

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "FulinAI Development Commands"
	@echo "============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

test: test-go test-python ## Run all tests (Go + Python)

test-go: ## Run Go tests with race detector
	cd backend/api-gateway && go test ./tests/... -v -race -count=1

test-python: ## Run Python tests with coverage
	cd backend/api-service && \
		pytest tests/ -v --tb=short \
			--cov=agents --cov=cache --cov=rag \
			--cov-report=term-missing 2>&1 || true

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

lint: lint-go lint-python ## Run all linters

lint-go: ## Go vet + staticcheck
	cd backend/api-gateway && go vet ./...

lint-python: ## Python ruff check
	cd backend/api-service && ruff check . || true

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build: build-go build-python build-frontend ## Build all Docker images

build-go: ## Build Go Gateway image
	docker compose build go-service

build-python: ## Build Python API image
	docker compose build python-service

build-frontend: ## Build Frontend image
	docker compose build frontend

# ---------------------------------------------------------------------------
# Docker Compose
# ---------------------------------------------------------------------------

up: ## Start all services
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## Tail all service logs
	docker compose logs -f

restart: ## Restart all services
	docker compose restart

status: ## Show service status
	docker compose ps

# ---------------------------------------------------------------------------
# Dev (local, without Docker)
# ---------------------------------------------------------------------------

dev-go: ## Run Go Gateway locally
	cd backend/api-gateway && go run cmd/server/main.go

dev-python: ## Run Python API locally
	cd backend/api-service && python main.py

dev-frontend: ## Run Frontend dev server
	cd frontend && npm run dev

dev: dev-python ## Start Python API (main entry point for local dev)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

db-migrate: ## Run Alembic migrations
	cd raw_data && alembic upgrade head

db-init: ## Initialize database from SQL files
	mysql -u root -p < raw_data/sql/job_init.sql
	mysql -u root -p < raw_data/sql/resume_init.sql

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean: ## Remove caches, builds, and containers
	docker compose down -v --remove-orphans
	cd backend/api-gateway && go clean -cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -node name dist -exec rm -rf {} + 2>/dev/null || true
