package devteam

import (
	"fmt"
	"strings"
)

func (orchestrator *Orchestrator) extractContext(event ColumnEvent) string {
	keywords := keywordsFromCard(event.Title, event.Description)
	radius, err := orchestrator.extractor.Extract(".", keywords, orchestrator.cfg.BlastRadiusDepth)

	if err != nil || radius == nil {
		return ""
	}

	return radius.Format()
}

func featureBranch(event ColumnEvent) string {
	slug := strings.ToLower(event.Title)
	slug = strings.Map(func(character rune) rune {
		if (character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') {
			return character
		}

		return '-'
	}, slug)

	parts := strings.FieldsFunc(slug, func(character rune) bool { return character == '-' })

	if len(parts) > 6 {
		parts = parts[:6]
	}

	return "devteam/" + event.ID[:8] + "-" + strings.Join(parts, "-")
}

func keywordsFromCard(title, description string) []string {
	combined := title + " " + description
	words := strings.FieldsFunc(combined, func(character rune) bool {
		return !(character >= 'a' && character <= 'z') &&
			!(character >= 'A' && character <= 'Z') &&
			!(character >= '0' && character <= '9')
	})

	seen := make(map[string]struct{})
	keywords := make([]string, 0, len(words))

	for _, word := range words {
		lower := strings.ToLower(word)

		if len(lower) < 3 {
			continue
		}

		if _, ok := seen[lower]; ok {
			continue
		}

		seen[lower] = struct{}{}
		keywords = append(keywords, lower)
	}

	return keywords
}

/*
formatSubtaskContext renders the subtask's stored context snapshot into a
markdown block for injection into the developer agent's system prompt.
*/
func formatSubtaskContext(subtask Subtask) string {
	snap := subtask.ContextSnapshot
	var builder strings.Builder

	if snap.BlastRadius != "" {
		builder.WriteString(snap.BlastRadius)
		builder.WriteString("\n")
	}

	if len(snap.FilesInScope) > 0 {
		builder.WriteString("### Files in scope for this subtask\n")

		for _, file := range snap.FilesInScope {
			fmt.Fprintf(&builder, "- %s\n", file)
		}

		builder.WriteString("\n")
	}

	if len(snap.KeySymbols) > 0 {
		builder.WriteString("### Key symbols\n")

		for _, symbol := range snap.KeySymbols {
			fmt.Fprintf(&builder, "- `%s`\n", symbol)
		}

		builder.WriteString("\n")
	}

	if len(snap.SiblingNotes) > 0 {
		builder.WriteString("### Sibling subtask conflicts to be aware of\n")

		for title, note := range snap.SiblingNotes {
			fmt.Fprintf(&builder, "- **%s**: %s\n", title, note)
		}

		builder.WriteString("\n")
	}

	return builder.String()
}
