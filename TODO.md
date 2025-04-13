# A2A Implementation TODOs

Based on a review of the A2A specification and the current codebase, the following areas may need attention:

**1. JSON-RPC Method Alignment:**

- [x] Refactor `tasks/sendSubscribe` to `tasks/send` with a `subscribe` parameter (align with `SendTaskRequest`). (Note: Decided `tasks/send` itself handles streaming logic; separate subscribe parameter not needed as client initiates SSE connection separately).
- [x] Implement the `tasks/resubscribe` method (`TaskResubscriptionRequest`). (Note: Initial implementation returns current status; history retrieval based on `historyLength` is a TODO).
- [x] Review/Standardize `task.artifact.get`. (Decision: Removed custom handler; rely on standard A2A artifact delivery via SSE `TaskArtifactUpdateEvent` or `FilePart`/`DataPart` in final message).

**2. Parameter Handling in Methods:**

- [x] Ensure `task.get` handles the `historyLength` parameter (`GetTaskRequest`).
- [x] Ensure `task.send` handles `id`, `sessionId`, `message`, `pushNotification`, `historyLength`, and `metadata` parameters (`SendTaskRequest`). (Note: `pushNotification` and `historyLength` are now used; `sessionId` and `metadata` are parsed but not yet utilized).
- [x] Ensure `task.cancel` handles the optional `metadata` parameter (`CancelTaskRequest`). (Note: Parameter is parsed; error codes -32001 and -32002 added).

**3. Response Structure Alignment:**

- [ ] Align `task.get` response with `Task` structure (`GetTaskResponse`). (Needs update: Spec expects Task object).
- [ ] Align `task.cancel` response with `Task` structure (`CancelTaskResponse`). (Needs update: Spec expects Task object).
- [ ] Align `task.setPushNotification` response with `TaskPushNotificationConfig` structure (`SetTaskPushNotificationResponse`). (Needs update: Spec expects TaskPushNotificationConfig object).
- [ ] Align `task.getPushNotification` response with `TaskPushNotificationConfig` or `null` when no config exists (`GetTaskPushNotificationResponse`). (Needs update: Spec expects TaskPushNotificationConfig or null).

**4. Error Code Alignment:**

- [~] Replace generic `-32000` errors with specific A2A error codes (`-32001`, `-32002`, `-32003`, `-32004`, etc.). (Note: Implemented for `tasks/send` in `pkg/task/manager/manager.go`. Needs review/implementation for other handlers like get, cancel, resubscribe etc. as they are built/refactored.)

**5. Data Structure Verification:**

- [ ] Verify `TaskState` serialization/deserialization between Go `iota` and spec string enums.
- [ ] Compare Go `AgentCard` structure (and related `Capabilities`, `Authentication`, `Skill`) against the A2A JSON schema definitions.
- [ ] Compare Go `Artifact` structure against the A2A JSON schema definition, especially regarding `index`, `append`, `lastChunk`.

**6. Streaming (SSE) Event Formatting:**

- [ ] Ensure SSE events use `event: task.status` / `event: task.artifact`.
- [ ] Ensure SSE event `data:` field contains correctly JSON-formatted `TaskStatusUpdateEvent` / `TaskArtifactUpdateEvent`.

**7. Functional Gaps:**

- [ ] Implement actual task cancellation logic in `HandleTaskCancel`.
- [ ] Implement handling for the `input-required` task state.
- [ ] Verify implementation and handling of `AgentAuthentication` schemes.
- [ ] Verify implementation and handling of `AgentCapabilities` (e.g., `stateTransitionHistory`).

# MCP Implementation TODOs

Based on a review of the MCP specification and the current codebase, the following areas need attention:

**1. Resource Management:**

- [x] Implement `resources/list` endpoint for resource discovery
- [x] Implement `resources/read` endpoint for reading resource contents
- [x] Implement resource templates for dynamic resources
- [x] Add support for both text and binary resources
- [x] Implement resource subscription mechanism
- [ ] Add resource update notifications
- [x] Implement proper MIME type handling
- [ ] Add resource access controls

**2. Prompt Management:**

- [x] Implement `prompts/list` endpoint for prompt discovery
- [x] Implement `prompts/get` endpoint for retrieving prompts
- [x] Add support for dynamic prompts with embedded resource context
- [x] Implement multi-step workflows
- [ ] Add UI integration support (slash commands, quick actions)
- [x] Implement prompt versioning
- [ ] Add prompt validation

**3. Sampling Support:**

- [x] Implement `sampling/createMessage` endpoint
- [x] Add support for model preferences
- [x] Implement context inclusion options
- [x] Add sampling parameters (temperature, maxTokens, etc.)
- [x] Implement proper error handling for sampling
- [ ] Add rate limiting for sampling requests
- [ ] Implement cost monitoring

**4. Roots Management:**

- [x] Implement roots capability declaration
- [x] Add support for multiple root URIs
- [x] Implement root change notifications
- [x] Add root validation
- [x] Implement root-based access controls
- [x] Add root monitoring

**5. Tool Enhancements:**

- [ ] Add proper tool annotations (readOnlyHint, destructiveHint, etc.)
- [ ] Implement tool update notifications
- [ ] Add tool versioning support
- [ ] Enhance error handling with specific error codes
- [ ] Implement proper input validation
- [ ] Add rate limiting for tool calls
- [ ] Implement tool usage monitoring
- [ ] Add tool documentation generation

**6. Security Enhancements:**

- [ ] Implement proper authentication for all endpoints
- [ ] Add authorization checks for resources and tools
- [ ] Implement rate limiting across all features
- [ ] Add audit logging
- [ ] Implement proper error handling
- [ ] Add input sanitization
- [ ] Implement timeout handling
- [ ] Add security monitoring

**7. Testing Requirements:**

- [ ] Add unit tests for all new features
- [ ] Implement integration tests
- [ ] Add security testing
- [ ] Implement performance testing
- [ ] Add error handling tests
- [ ] Implement load testing
- [ ] Add documentation tests

# TODO List

## Completed

- [x] Implement resource templates for dynamic resources
- [x] Implement resource subscription mechanism
- [x] Implement prompt management system
- [x] Implement sampling system with model preferences and context support

## In Progress

- [ ] Implement resource caching
- [ ] Add resource validation
- [ ] Implement resource versioning

## Planned

- [ ] Add resource compression
- [ ] Implement resource encryption
- [ ] Add resource access control
- [ ] Implement resource backup/restore
- [ ] Add resource monitoring
- [ ] Implement resource cleanup
- [ ] Add resource logging
- [ ] Implement resource metrics
- [ ] Add resource documentation
