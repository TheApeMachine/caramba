#!/usr/bin/env bash
# Apply docker/postgres/init SQL to the running Compose database.
# Use this when the Postgres volume already existed before init scripts were added
# (init only runs automatically on first empty data directory).
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose exec postgres psql -U postgres -d electric -v ON_ERROR_STOP=1 \
	-f /docker-entrypoint-initdb.d/01-research-projects.sql

docker compose exec postgres psql -U postgres -d electric -v ON_ERROR_STOP=1 \
	-f /docker-entrypoint-initdb.d/02-kanban-boards.sql

docker compose exec postgres psql -U postgres -d electric -v ON_ERROR_STOP=1 \
	-f /docker-entrypoint-initdb.d/03-research-papers.sql

docker compose exec postgres psql -U postgres -d electric -v ON_ERROR_STOP=1 \
	-f /docker-entrypoint-initdb.d/05-research-project-members.sql

docker compose exec postgres psql -U postgres -d electric -v ON_ERROR_STOP=1 \
	-f /docker-entrypoint-initdb.d/06-researcher-profiles.sql

echo "Schema applied on database 'electric' (research_projects, kanban, research_papers, project_members, researcher_profiles)."
