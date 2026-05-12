package devteam

import (
	"fmt"
	"strings"
)

/*
EditRequest describes a targeted change: a contiguous block of original lines
to replace with new lines. Both slices are line-oriented (no trailing newline
required on individual items).
*/
type EditRequest struct {
	Path        string
	OldLines    []string
	NewLines    []string
	ClaimIntent string
}

/*
SearchResult is a single grep hit returned by VirtualEditor.Search.
*/
type SearchResult struct {
	Path string
	Line uint32
	Text string
}

/*
VirtualEditor wraps a Sandbox and provides a structured, safety-checked
interface for the developer agent. It enforces two invariants:

  1. Read-before-write: a file must be viewed (or created fresh) before any
     edit is accepted. This prevents blind overwrites and forces the agent to
     reason about existing content before mutating it.

  2. Claim-before-write: before mutating a file the agent must hold the
     FileLockRegistry claim for that path. Concurrent agents working in
     separate containers therefore cannot unknowingly produce conflicting
     changes to the same logical path.
*/
type VirtualEditor struct {
	agentID  string
	sandbox  *Sandbox
	locks    *FileLockRegistry
	readSet  map[string]struct{}
}

/*
NewVirtualEditor constructs a VirtualEditor for one agent's lifetime.
*/
func NewVirtualEditor(
	agentID string,
	sandbox *Sandbox,
	locks *FileLockRegistry,
) *VirtualEditor {
	return &VirtualEditor{
		agentID: agentID,
		sandbox: sandbox,
		locks:   locks,
		readSet: make(map[string]struct{}),
	}
}

/*
Search runs a case-sensitive grep inside /workspace and returns up to maxResults
hits. Pattern is a basic regex understood by grep -En.
*/
func (editor *VirtualEditor) Search(pattern string, maxResults int) ([]SearchResult, error) {
	cmd := fmt.Sprintf(
		`grep -rEn --include="*.go" --include="*.py" --include="*.ts" --include="*.js" \
         --include="*.c" --include="*.cpp" --include="*.h" --include="*.rs" \
         -m %d %q . 2>/dev/null || true`,
		maxResults, pattern,
	)

	out, err := editor.sandbox.Exec(cmd)

	if err != nil {
		return nil, fmt.Errorf("editor: search: %w", err)
	}

	results := make([]SearchResult, 0)

	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if line == "" {
			continue
		}

		// format: path:linenum:text
		parts := strings.SplitN(line, ":", 3)

		if len(parts) < 3 {
			continue
		}

		var lineNum uint32
		fmt.Sscanf(parts[1], "%d", &lineNum)

		results = append(results, SearchResult{
			Path: parts[0],
			Line: lineNum,
			Text: parts[2],
		})
	}

	return results, nil
}

/*
View reads a file and records it as read. fromLine and toLine are 1-based and
inclusive; pass 0/0 to read the entire file. The returned string includes
line-number prefixes so the agent can reference exact positions in EditRequest.
*/
func (editor *VirtualEditor) View(path string, fromLine, toLine uint32) (string, error) {
	content, err := editor.sandbox.ReadFile(path)

	if err != nil {
		return "", fmt.Errorf("editor: view %q: %w", path, err)
	}

	lines := strings.Split(content, "\n")
	start, end := uint32(1), uint32(len(lines))

	if fromLine > 0 {
		start = fromLine
	}

	if toLine > 0 && toLine < end {
		end = toLine
	}

	if start > uint32(len(lines)) {
		start = uint32(len(lines))
	}

	var sb strings.Builder

	for i := start; i <= end; i++ {
		fmt.Fprintf(&sb, "%4d\t%s\n", i, lines[i-1])
	}

	editor.readSet[path] = struct{}{}

	return sb.String(), nil
}

/*
Edit performs a targeted replacement inside an existing file. The agent must
have previously viewed the file (enforced by readSet) and must be able to
claim the file lock. OldLines must match a contiguous run of lines verbatim
(after trimming trailing whitespace); NewLines replaces that run.
*/
func (editor *VirtualEditor) Edit(req EditRequest) error {
	if _, read := editor.readSet[req.Path]; !read {
		return fmt.Errorf(
			"editor: must view %q before editing it", req.Path,
		)
	}

	result := editor.locks.Claim(editor.agentID, req.Path, req.ClaimIntent)

	if !result.Acquired {
		return fmt.Errorf(
			"editor: %q is claimed by agent %s (%s)",
			req.Path, result.HolderID, result.Intent,
		)
	}

	content, err := editor.sandbox.ReadFile(req.Path)

	if err != nil {
		return fmt.Errorf("editor: edit read %q: %w", req.Path, err)
	}

	updated, err := applyLineEdit(content, req.OldLines, req.NewLines)

	if err != nil {
		return fmt.Errorf("editor: edit %q: %w", req.Path, err)
	}

	return editor.sandbox.WriteFile(req.Path, updated)
}

/*
Create writes a new file. No read-gate is required (file does not exist yet)
but the lock is still claimed to prevent a race with another agent creating the
same file simultaneously.
*/
func (editor *VirtualEditor) Create(path, content, claimIntent string) error {
	result := editor.locks.Claim(editor.agentID, path, claimIntent)

	if !result.Acquired {
		return fmt.Errorf(
			"editor: %q is claimed by agent %s (%s)",
			path, result.HolderID, result.Intent,
		)
	}

	editor.readSet[path] = struct{}{}

	return editor.sandbox.WriteFile(path, content)
}

/*
applyLineEdit finds the first occurrence of oldLines (verbatim, trimmed) inside
the file content and replaces it with newLines. Returns an error when the block
cannot be located, preventing silent no-ops.
*/
func applyLineEdit(content string, oldLines, newLines []string) (string, error) {
	fileLines := strings.Split(content, "\n")

	idx := findBlock(fileLines, oldLines)

	if idx < 0 {
		return "", fmt.Errorf(
			"old_lines block not found in file (first line: %q)",
			firstOrEmpty(oldLines),
		)
	}

	result := make([]string, 0, len(fileLines)-len(oldLines)+len(newLines))
	result = append(result, fileLines[:idx]...)
	result = append(result, newLines...)
	result = append(result, fileLines[idx+len(oldLines):]...)

	return strings.Join(result, "\n"), nil
}

func findBlock(fileLines, block []string) int {
	if len(block) == 0 {
		return -1
	}

outer:
	for i := 0; i <= len(fileLines)-len(block); i++ {
		for j, blockLine := range block {
			if strings.TrimRight(fileLines[i+j], " \t") != strings.TrimRight(blockLine, " \t") {
				continue outer
			}
		}

		return i
	}

	return -1
}

func firstOrEmpty(lines []string) string {
	if len(lines) > 0 {
		return lines[0]
	}

	return ""
}
