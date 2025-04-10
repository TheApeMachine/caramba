# Terminal Game Development Instructions

This document provides guidance for the planner and executor agents to collaborate on developing a simple terminal-based adventure game.

## Game Overview

The game should be a text-based adventure where the player navigates through different locations, collects items, solves puzzles, and tries to achieve a goal. The template provides a basic structure for movement, item management, and location descriptions.

## Collaboration Workflow

### Planner Agent

The planner agent should:

1. Design the overall game concept, story, and objectives
2. Plan the game world: locations, connections, and items
3. Design puzzles and challenges
4. Create win/lose conditions
5. Communicate the plan to the executor agent via the system_message tool

### Executor Agent

The executor agent should:

1. Receive the design plan from the planner agent
2. Implement the code based on the plan
3. Fill in the TODOs in the game_template.py
4. Test the game for functionality
5. Request clarifications from the planner if needed

## Development Process

1. The planner agent creates a detailed design document including:

   - Game title and story
   - Map of locations and connections
   - List of items and their properties
   - Puzzles and their solutions
   - Win/lose conditions

2. The planner agent sends the design to the executor using the system_message tool

3. The executor agent uses the environment tool to:

   - Access the template file
   - Implement the game according to the plan
   - Test the implementation
   - Request clarifications if needed

4. The planner agent reviews progress and provides additional guidance as needed

## Communication Protocol

When communicating between agents:

1. Always specify which part of the game you're referring to
2. Be explicit about any changes to the original plan
3. Use specific examples when asking questions
4. Acknowledge messages received from the other agent

## Technical Requirements

The game should:

1. Run on Python 3.6+
2. Use only standard library modules
3. Have clear, commented code
4. Handle player input gracefully (including invalid commands)
5. Provide clear feedback to the player

## Testing

The executor agent should test the game by:

1. Verifying all locations are accessible
2. Testing all item interactions
3. Checking win/lose conditions
4. Ensuring error handling works properly

## Final Deliverable

The final game should be a complete, playable adventure that:

1. Tells an engaging story
2. Has interesting locations to explore
3. Features meaningful item interactions
4. Includes at least one puzzle to solve
5. Has clear win/lose conditions
