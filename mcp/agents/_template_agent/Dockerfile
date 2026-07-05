# Optional container packaging for a `type: "container"` MCP agent.
#
# A container agent must expose the tiny HTTP shim the executor speaks:
#   GET  /health       -> {"status": "ok"}
#   POST /handle_task   -> {"status", "result", "metadata"}
# The example below wraps agent.py behind that shim (see server.py you provide).
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Add any agent-specific deps here (kept empty for the pure-stdlib template).
# RUN pip install --no-cache-dir -r requirements.txt

# Bind the HTTP shim (implement server.py to import AgentAdapter from agent.py).
EXPOSE 8080
ENV MCP_AGENT_PORT=8080
CMD ["python", "server.py"]
