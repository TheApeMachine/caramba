Okay, we're working on redesigning our agent framework, and we need to get this test setup working.

The test setup should be a planner agent, and a developer agent, working together.

Agents should be able to work concurrently, but also communicate with each other, and even be able to wait until another agent completes a requirement first.

Further more, while we are working on the local first setup, eventually we will need to be able to deploy agents, providers, and tools on separate servers, and still have a single "grid" that works as a cohesive agent setup.

For this we are using Cap n Proto.

This means that for the most part, all methods and functionality needs to be implemented using Cap n Proto schemas.

If you need to create new Cap n Proto schema, please make sure to update the Makefile in the root of this project to add the generation command in there.

Then you should implement the server code for the cap n proto interfaces, like the agent and provider packages already do somewhat.

Potentially you should also make some conveniance methods to keep the public API simple, like what happens in the datura package.

Please make sure you read code first, before you jump to conclusions and start generating all kinds of code, there is already a lot, though some of it may need to be removed.

If you encounter something that seems to have a missing implementation, it is most likely old and needs to be removed. Use good judgement in these situations, or ask me.

Also, mind you that we do not create all kinds of "types" of agents, instead we have one agent type that can be configured to become any type of agent.

Finally, agents also need to be able to be used as a tool, given that agents in some cases should be able to create new agents and delegate tasks to them.

Please make the public API as simple as possible to use, this is extremely important. We want developers that use the framework to be able to focus on creating agent setups and experiment, and not worry about complexity and boilerplate, so please hide that away behind abstractions that are clean and comfortable to use, with extreme focus on minimizing boilerplate.

Work in small steps, and verify with me if you are on the right track, do not just generate huge amounts of code, I am very specific about clean code.
