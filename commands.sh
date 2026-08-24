#!/usr/bin/env bash
set -e

# 1. Create Python virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Copy environment configuration template (add your OPENROUTER_API_KEY in .env)
cp -n .env.example .env

# 5. Start Langfuse observability & tracing Docker containers
docker compose -f observability/docker-compose.yml up -d

# 6. Start the candidate matching web server (UI: http://localhost:8000 | Traces: http://localhost:3001)
python -m agent.server
