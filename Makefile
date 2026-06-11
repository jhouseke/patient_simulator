.PHONY: help proto db-up db-down db-logs db-test engine-run test lint

PYTHON := python3
PIP := pip3
PROTO_SRC := proto/simulation.proto
PROTO_OUT := src/engine/proto/v1

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

proto: ## Regenerate Python gRPC/protobuf stubs
	$(PYTHON) -m grpc_tools.protoc \
		-I./proto \
		--python_out=$(PROTO_OUT) \
		--grpc_python_out=$(PROTO_OUT) \
		$(PROTO_SRC)
	# Patch relative import in generated grpc file
	sed -i '' 's/^import simulation_pb2 as simulation__pb2/from . import simulation_pb2 as simulation__pb2/' \
		$(PROTO_OUT)/simulation_pb2_grpc.py

db-up: ## Start PostgreSQL with triple extensions
	docker compose up -d --build db

db-down: ## Stop PostgreSQL
	docker compose down

db-logs: ## Tail PostgreSQL logs
	docker compose logs -f db

db-test: ## Run Week 1 triple-extension load test (requires db-up)
	$(PYTHON) scripts/load_test_db.py

engine-run: ## Start the gRPC engine server
	PYTHONPATH=src $(PYTHON) -m engine.main

test: ## Run pytest suite
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v

lint: ## Run ruff / basic formatting check
	$(PYTHON) -m py_compile src/engine/main.py
	$(PYTHON) -m py_compile src/engine/server.py
	$(PYTHON) -m py_compile src/engine/store/postgres.py
	$(PYTHON) -m py_compile src/engine/pdes/clock.py
	$(PYTHON) -m py_compile src/engine/pkpd/models.py
