# Project Board MCP Tool

A hierarchical project management system for organizing research, development, and experimental work in Caramba.

## Overview

The Project Board tool provides a structured way to track work items with four levels of hierarchy:

- **Projects** - Top-level initiatives (e.g., "Implement DBA Attention")
- **Epics** - Major components within a project (e.g., "Core Algorithm", "Benchmarking Suite")
- **Stories** - User-facing features or capabilities (e.g., "As a researcher, I can run DBA experiments via manifest")
- **Tasks** - Specific implementation steps (e.g., "Write triton kernel", "Add config schema")

## Architecture

### Components

1. **Data Models** (`models.py`) - Pydantic models defining the hierarchy
2. **Storage Layer** (`storage.py`) - PostgreSQL persistence with asyncpg
3. **MCP Server** (`tool.py`) - FastMCP server exposing project management tools
4. **Project Manager Agent** (`config/personas/project_manager.yml`) - AI agent that parses unstructured text into structured work items

### Data Model

```
Project
├── Epic 1
│   ├── Story 1
│   │   ├── Task 1
│   │   ├── Task 2
│   │   └── Task 3 (depends on Task 1)
│   └── Story 2
│       └── Task 4
└── Epic 2
    └── ...
```

Each work item has:
- **Status**: todo, in_progress, blocked, done, cancelled
- **Priority**: low, medium, high, critical
- **Dependencies**: List of other items this depends on
- **Tags**: For categorization and filtering
- **Timestamps**: created_at, updated_at, completed_at

## Usage

### Starting the Services

```bash
# Start PostgreSQL and ProjectBoard MCP server
docker-compose up -d postgres-projectboard projectboard-mcp

# Check logs
docker-compose logs -f projectboard-mcp
```

### Accessing via Agent

The `project_manager` agent has access to the projectboard tool and can:

1. **Parse unstructured text** into projects/epics/stories/tasks
2. **Create work items** with proper hierarchy
3. **Identify dependencies** between items
4. **Track progress** by updating statuses
5. **Query work items** by various filters

Example conversation:

```
User: We need to implement the project management system with PostgreSQL backend

Project Manager: I'll break this down into structured work items...
[Creates project "Project Management System"]
[Creates epics for "Data Model", "MCP Server", "Agent Integration"]
[Creates stories and tasks with dependencies]
```

### Direct API Access

The MCP server exposes these tools:

#### Projects

```python
# Create a project
create_project(
    title="Project Management System",
    description="Hierarchical task tracking for research work",
    objectives=["Enable structured work planning", "Track dependencies"],
    tags=["infrastructure", "agents"]
)

# List projects
list_projects(status="in_progress")

# Update project status
update_project_status(
    project_id="uuid-here",
    status="done"
)
```

#### Epics

```python
# Create an epic
create_epic(
    project_id="project-uuid",
    title="Data Model Implementation",
    description="Design and implement Pydantic models",
    goals=["Support full hierarchy", "Enable dependencies"],
    priority="high",
    tags=["backend"]
)

# List epics
list_epics(project_id="project-uuid", status="in_progress")

# Update epic status
update_epic_status(epic_id="epic-uuid", status="done")
```

#### Stories

```python
# Create a story
create_story(
    epic_id="epic-uuid",
    title="Define hierarchical data models",
    description="Create Pydantic models for Project/Epic/Story/Task",
    acceptance_criteria=[
        "Models support full CRUD",
        "Dependencies are trackable",
        "Timestamps are automatic"
    ],
    priority="high",
    assignee="developer",
    tags=["backend", "data"]
)

# List stories
list_stories(epic_id="epic-uuid", status="todo")

# Update story
update_story_status(
    story_id="story-uuid",
    status="in_progress",
    assignee="ml_expert"
)
```

#### Tasks

```python
# Create a task
create_task(
    story_id="story-uuid",
    title="Write Pydantic models",
    description="Implement Project, Epic, Story, Task models with validation",
    priority="high",
    assignee="developer",
    depends_on=["prerequisite-task-uuid"],
    tags=["coding"]
)

# List tasks
list_tasks(story_id="story-uuid", status="in_progress")

# Update task
update_task_status(
    task_id="task-uuid",
    status="done",
    assignee="developer"
)
```

#### Queries

```python
# Get full project hierarchy
get_project_hierarchy(project_id="project-uuid")
# Returns: {project: {...}, epics: [{epic: {...}, stories: [{story: {...}, tasks: [...]}]}]}

# Search across all work items
search_work_items(
    query="authentication",
    item_types=["story", "task"],
    status="in_progress",
    priority="high",
    tags=["security"]
)
```

## Database Schema

PostgreSQL schema with four main tables:

- `projects` - Top-level projects
- `epics` - Epics within projects (FK to projects)
- `stories` - Stories within epics (FK to epics)
- `tasks` - Tasks within stories (FK to stories)

All tables have:
- UUID primary keys
- Status and priority enums
- JSON fields for arrays (objectives, goals, criteria, dependencies, tags)
- Timestamps (created_at, updated_at, completed_at)
- Indexes on common queries (status, parent IDs)

Cascade deletes ensure:
- Deleting a project removes all its epics, stories, and tasks
- Deleting an epic removes all its stories and tasks
- Deleting a story removes all its tasks

## Environment Variables

```bash
# PostgreSQL connection (used by projectboard-mcp service)
PROJECTBOARD_DATABASE_URL=postgresql://projectboard:projectboard@postgres-projectboard:5432/projectboard
```

## Docker Compose Configuration

### PostgreSQL Service

```yaml
postgres-projectboard:
  image: postgres:16-alpine
  environment:
    - POSTGRES_USER=projectboard
    - POSTGRES_PASSWORD=projectboard
    - POSTGRES_DB=projectboard
  ports:
    - "5433:5432"
  volumes:
    - projectboard_data:/var/lib/postgresql/data
```

### ProjectBoard MCP Service

```yaml
projectboard-mcp:
  build:
    context: .
    dockerfile: ai/tools/projectboard/Dockerfile
  depends_on:
    postgres-projectboard:
      condition: service_healthy
  ports:
    - "8007:8001"
  environment:
    - PROJECTBOARD_DATABASE_URL=postgresql://projectboard:projectboard@postgres-projectboard:5432/projectboard
```

### Project Manager Persona

```yaml
project-manager:
  build:
    context: .
    dockerfile: docker/Dockerfile.persona-a2a
  environment:
    - PERSONA_FILE=config/personas/project_manager.yml
    - CARAMBA_PROJECTBOARD_HOST=projectboard-mcp
    - CARAMBA_PROJECTBOARD_PORT=8001
```

## Integration with Root Agent

The root agent has access to both:
1. **projectboard tool** directly for quick queries
2. **project_manager sub-agent** for parsing unstructured input

When a user describes work to be done, the root agent can:
- Consult the project_manager to structure the work
- Create projects, epics, stories, and tasks
- Track progress over time
- Query status and dependencies

## Development

### Running Locally

```bash
# Start PostgreSQL
docker-compose up -d postgres-projectboard

# Run MCP server locally (for development)
cd ai/tools/projectboard
python -m tool --host 0.0.0.0 --port 8001 --database-url postgresql://projectboard:projectboard@localhost:5433/projectboard
```

### Testing

```bash
# Test database connection
docker-compose exec postgres-projectboard psql -U projectboard -c "\\dt"

# Test MCP server
curl http://localhost:8007/health

# Test tool call (via MCP)
curl -X POST http://localhost:8007/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/call", "params": {"name": "list_projects"}}'
```

## Example Workflows

### 1. Planning a New Feature

```
User: "I want to add user authentication to the system"

Root Agent: [Calls project_manager]

Project Manager:
1. Creates Project: "User Authentication System"
2. Creates Epics:
   - "Authentication Backend"
   - "Session Management"
   - "Security Hardening"
3. Creates Stories:
   - "As a user, I can log in with username/password"
   - "As a user, I can stay logged in across sessions"
   - "As an admin, I can see login audit logs"
4. Creates Tasks with dependencies:
   - "Design user schema" (no deps)
   - "Implement password hashing" (depends on schema)
   - "Create login API endpoint" (depends on hashing)
   - etc.

Root Agent: [Summarizes the structure to user]
"I've organized the authentication work into 1 project, 3 epics, 8 stories, and 24 tasks.
The critical path starts with designing the user schema, then implementing password hashing..."
```

### 2. Tracking Progress

```
User: "What's the status of our authentication work?"

Root Agent: [Queries projectboard]
"The User Authentication System project is in progress:
- Authentication Backend epic: 60% complete (6/10 stories done)
- Session Management epic: Just started (1/5 stories done)
- Security Hardening epic: Not started yet

Currently blocked on:
- Task: 'Implement JWT refresh tokens' (waiting for 'Add JWT library dependency')

Next up:
- Story: 'As a user, I can reset my password' (priority: high)"
```

### 3. Identifying Dependencies

```
User: "Can we start the mobile app development now?"

Project Manager: [Analyzes dependencies]
"The mobile app depends on:
1. ✅ REST API endpoints (Done)
2. ⏳ Authentication system (In progress - 60% complete)
3. ❌ Push notifications service (Not started)

Recommendation: Wait for authentication to complete (est. 2-3 more stories).
We can start on offline-first features that don't require auth."
```

## Future Enhancements

Potential improvements:
- **Time estimates** - Add story points or time estimates to tasks
- **Burndown charts** - Generate visualizations of progress
- **Gantt charts** - Show timeline with dependencies
- **Sprint planning** - Group stories into sprints/iterations
- **Assignee workload** - Track capacity and allocation
- **Comments/discussion** - Add threaded discussions to work items
- **File attachments** - Attach designs, screenshots, etc.
- **Custom fields** - Allow user-defined metadata
- **Webhooks** - Notify external systems of status changes
- **GraphQL API** - Alternative query interface

## License

Part of the Caramba project.
