#!/usr/bin/env bash
# Apply docker/postgres/init SQL to the running Compose database.
# Use this when the Postgres volume already existed before init scripts were added
# (init only runs automatically on first empty data directory).
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose exec postgres psql -U postgres -d electric -v ON_ERROR_STOP=1 \
	-f /docker-entrypoint-initdb.d/01-research-projects.sql

echo "Schema applied on database 'electric' (table research_projects)."
