package markymark

import "strings"

/*
Down is an object that formats text as Markdown.
*/
type Down struct {
}

/*
NewDown returns a new Down object.
*/
func NewDown() *Down {
	return &Down{}
}

/*
Quote wraps the text in a quote block.
*/
func (down *Down) Quote(text string) string {
	lines := strings.Split(text, "\n")

	for i, line := range lines {
		lines[i] = "> " + strings.TrimSpace(line)
	}

	return strings.Join(lines, "\n")
}
