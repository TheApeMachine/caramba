package devteam

import (
	"encoding/json"
	"fmt"
	"strings"
)

/*
checkDoneGate verifies that the agent has satisfied the test contract before
accepting a done() signal. It returns a non-empty rejection message when any
condition is unmet, which is fed back to the LLM as a tool result so it knows
exactly what to fix.
*/
func (developer *Developer) checkDoneGate(input map[string]any) string {
	testFile, _ := input["test_file"].(string)
	testCommand, _ := input["test_command"].(string)

	if testFile == "" {
		return "REJECTED: test_file is required. Write at least one test file before calling done()."
	}

	if testCommand == "" {
		return "REJECTED: test_command is required. Run go test before calling done()."
	}

	if rejection := validateGoTestCommand(testCommand); rejection != "" {
		return rejection
	}

	if _, err := developer.editor.sandbox.ReadFile(testFile); err != nil {
		return fmt.Sprintf(
			"REJECTED: test file %q does not exist in the workspace. Create it before calling done().",
			testFile,
		)
	}

	testOutput, err := developer.editor.sandbox.Exec(testCommand)

	if err != nil {
		return fmt.Sprintf(
			"REJECTED: test command exited with an error. Output seen:\n%s",
			truncate(testOutput, 800),
		)
	}

	if strings.Contains(testCommand, "-json") {
		parsed, passed := parseGoTestJSON(testOutput)

		if parsed && !passed {
			return fmt.Sprintf(
				"REJECTED: go test -json reported a failing package or test. Output seen:\n%s",
				truncate(testOutput, 800),
			)
		}
	}

	return ""
}

func validateGoTestCommand(command string) string {
	trimmed := strings.TrimSpace(command)

	if strings.ContainsAny(trimmed, "\n\r;&|<>`$") {
		return "REJECTED: test_command must be a single go test invocation without shell control characters."
	}

	fields := strings.Fields(trimmed)

	if len(fields) < 2 || fields[0] != "go" || fields[1] != "test" {
		return "REJECTED: test_command must start with go test."
	}

	return ""
}

func parseGoTestJSON(output string) (bool, bool) {
	parsed := false
	passed := false

	for _, line := range strings.Split(output, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}

		var event struct {
			Action string `json:"Action"`
		}

		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}

		if event.Action == "" {
			continue
		}

		parsed = true

		if event.Action == "fail" {
			return true, false
		}

		if event.Action == "pass" {
			passed = true
		}
	}

	return parsed, passed
}
