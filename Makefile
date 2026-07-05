# Convenience targets for the MCP (Multi-Agent Control Plane) subsystem.
# All MCP targets run fully offline (in-memory bus, no Docker/Redis required).

PYTHON ?= python
FLAKE8_ARGS = --max-line-length=100 --extend-ignore=E203,W503

.PHONY: mcp-start mcp-test mcp-lint mcp-discover mcp-simulate

## mcp-start: discover agents and report MCP runtime readiness
mcp-start:
	ENABLE_MCP=true $(PYTHON) scripts/mcp_cli.py start

## mcp-discover: list discovered agents
mcp-discover:
	$(PYTHON) scripts/mcp_cli.py discover

## mcp-simulate: run the urban accident response mission in the simulator
mcp-simulate:
	$(PYTHON) scripts/mcp_cli.py simulate urban_accident_response

## mcp-test: run the MCP unit + integration tests (offline)
mcp-test:
	ENABLE_MCP=true $(PYTHON) -m pytest -q -p no:cacheprovider --no-cov \
		tests/test_mcp_registry.py \
		tests/test_mcp_planner.py \
		tests/test_mcp_executor.py \
		tests/test_example_agents.py \
		tests/test_mcp_mission_flow.py

## mcp-lint: flake8 the MCP runtime, agents and CLI
mcp-lint:
	$(PYTHON) -m flake8 dvsa_api/mcp dvsa_api/api/mcp_router.py \
		mcp/agents mcp/simulators scripts/mcp_cli.py $(FLAKE8_ARGS)
