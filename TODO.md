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

- [x] Align `task.get` response with `TaskStatusUpdateEvent` structure (`GetTaskResponse`). (Verified: Handler returns `task.TaskStatusUpdateEvent`).
- [x] Align `task.cancel` response with boolean `result` (`CancelTaskResponse`). (Verified: Handler returns `true`).
- [x] Align `task.setPushNotification` response with `null` result (`SetTaskPushNotificationResponse`). (Verified: Handler returns `nil`).
- [x] Align `task.getPushNotification` response with `null` when no config exists (`GetTaskPushNotificationResponse`). (Verified: Handler returns `nil`).

**4. Error Code Alignment:**

- [ ] Replace generic `-32000` errors with specific A2A error codes (`-32001`, `-32002`, `-32003`, `-32004`, etc.). (Note: Some already updated in `task_get.go`, `task_resubscribe.go`, `task_cancel.go`)

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
