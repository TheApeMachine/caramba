package devteam

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	sitter "github.com/tree-sitter/go-tree-sitter"
	golang "github.com/tree-sitter/tree-sitter-go/bindings/go"
)

// symbolKinds are the tree-sitter node types we treat as top-level declarations.
var symbolKinds = map[string]bool{
	"function_declaration":  true,
	"method_declaration":    true,
	"type_declaration":      true,
	"const_declaration":     true,
	"var_declaration":       true,
	"short_var_declaration": true,
}

/*
Symbol is a named declaration extracted from a source file.
*/
type Symbol struct {
	File   string
	Line   uint32
	Kind   string
	Name   string
	Source string
}

/*
BlastRadius describes the full set of files and symbols that are transitively
reachable from an initial set of keyword matches. It is injected into the
developer agent's system prompt so the agent understands the exact scope of
change before writing a single line.
*/
type BlastRadius struct {
	// RootSymbols are the declarations directly matched by the card's keywords.
	RootSymbols []Symbol
	// ReachableFiles is the de-duplicated set of files that reference any root symbol.
	ReachableFiles []string
	// CallGraph maps symbol names to the symbols that call them.
	CallGraph map[string][]Symbol
}

/*
ContextExtractor walks a local repository tree, parses every Go source file
with tree-sitter, and builds a blast-radius graph for a given set of keywords
derived from a feature card.
*/
type ContextExtractor struct {
	language *sitter.Language
	parser   *sitter.Parser
}

/*
NewContextExtractor initialises the tree-sitter parser with the Go grammar.
*/
func NewContextExtractor() (*ContextExtractor, error) {
	language := sitter.NewLanguage(golang.Language())
	parser := sitter.NewParser()

	if err := parser.SetLanguage(language); err != nil {
		return nil, fmt.Errorf("context: set language: %w", err)
	}

	return &ContextExtractor{language: language, parser: parser}, nil
}

/*
Close releases tree-sitter resources.
*/
func (extractor *ContextExtractor) Close() {
	extractor.parser.Close()
}

/*
Extract walks repoRoot, parses every .go file, and returns a BlastRadius for
the provided keywords. The depth parameter controls how many hops of the
call-graph to follow when expanding the blast radius (2 is usually sufficient).
*/
func (extractor *ContextExtractor) Extract(repoRoot string, keywords []string, depth int) (*BlastRadius, error) {
	allSymbols, callerIndex, err := extractor.indexRepo(repoRoot)

	if err != nil {
		return nil, err
	}

	roots := matchSymbols(allSymbols, keywords)
	reachable := make(map[string]struct{})
	callGraph := make(map[string][]Symbol)

	extractor.expandBlast(roots, callerIndex, reachable, callGraph, depth)

	files := make([]string, 0, len(reachable))
	for f := range reachable {
		files = append(files, f)
	}

	return &BlastRadius{
		RootSymbols:    roots,
		ReachableFiles: files,
		CallGraph:      callGraph,
	}, nil
}

/*
Format renders the BlastRadius as a compact markdown block suitable for
injection into an LLM system prompt.
*/
func (radius *BlastRadius) Format() string {
	var sb strings.Builder

	sb.WriteString("### Blast Radius\n\n")
	sb.WriteString("**Directly matched symbols:**\n")

	for _, sym := range radius.RootSymbols {
		fmt.Fprintf(&sb, "- `%s` (%s) — %s:%d\n", sym.Name, sym.Kind, sym.File, sym.Line)
	}

	if len(radius.ReachableFiles) > 0 {
		sb.WriteString("\n**Files in blast radius (callers/callees):**\n")

		for _, f := range radius.ReachableFiles {
			fmt.Fprintf(&sb, "- %s\n", f)
		}
	}

	if len(radius.CallGraph) > 0 {
		sb.WriteString("\n**Call graph edges (symbol → callers):**\n")

		for sym, callers := range radius.CallGraph {
			names := make([]string, 0, len(callers))

			for _, caller := range callers {
				names = append(names, fmt.Sprintf("`%s`@%s:%d", caller.Name, caller.File, caller.Line))
			}

			fmt.Fprintf(&sb, "- `%s` ← %s\n", sym, strings.Join(names, ", "))
		}
	}

	return sb.String()
}

// ─── internal ────────────────────────────────────────────────────────────────

/*
callerIndex maps a symbol name to every symbol that calls it. Built during
repo indexing via the call_expression query.
*/
type callerIndex map[string][]Symbol

func (extractor *ContextExtractor) indexRepo(
	repoRoot string,
) ([]Symbol, callerIndex, error) {
	var allSymbols []Symbol

	index := make(callerIndex)

	err := filepath.WalkDir(repoRoot, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}

		if entry.IsDir() {
			name := entry.Name()

			if name == "vendor" || name == ".git" || strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}

			return nil
		}

		if !strings.HasSuffix(path, ".go") {
			return nil
		}

		src, err := os.ReadFile(path)

		if err != nil {
			return nil // skip unreadable files
		}

		rel, _ := filepath.Rel(repoRoot, path)
		syms, calls := extractor.parseFile(rel, src)
		allSymbols = append(allSymbols, syms...)

		// Populate the caller index: for each call in this file, record the
		// enclosing declaration as a caller of the called symbol.
		for calledName, callers := range calls {
			index[calledName] = append(index[calledName], callers...)
		}

		return nil
	})

	return allSymbols, index, err
}

// goSymbolQuery captures top-level function, method, and type declarations.
const goSymbolQuery = `
(function_declaration
  name: (identifier) @name) @decl

(method_declaration
  name: (field_identifier) @name) @decl

(type_declaration
  (type_spec name: (type_identifier) @name)) @decl
`

// goCallQuery captures call expressions to extract caller→callee edges.
const goCallQuery = `
(call_expression
  function: [
    (identifier) @callee
    (selector_expression field: (field_identifier) @callee)
  ]) @call
`

func (extractor *ContextExtractor) parseFile(
	rel string, src []byte,
) ([]Symbol, map[string][]Symbol) {
	tree := extractor.parser.Parse(src, nil)
	defer tree.Close()

	root := tree.RootNode()
	symbols := extractor.extractSymbols(rel, src, root)
	calls := extractor.extractCalls(rel, src, root, symbols)

	return symbols, calls
}

func (extractor *ContextExtractor) extractSymbols(
	rel string, src []byte, root *sitter.Node,
) []Symbol {
	query, qErr := sitter.NewQuery(extractor.language, goSymbolQuery)

	if qErr != nil {
		return nil
	}

	defer query.Close()

	cursor := sitter.NewQueryCursor()
	defer cursor.Close()

	matches := cursor.Matches(query, root, src)
	nameIdx, _ := query.CaptureIndexForName("name")

	var symbols []Symbol

	for match := matches.Next(); match != nil; match = matches.Next() {
		for _, node := range match.NodesForCaptureIndex(nameIdx) {
			sym := Symbol{
				File: rel,
				Line: uint32(node.StartPosition().Row + 1),
				Kind: node.Parent().Kind(),
				Name: node.Utf8Text(src),
			}

			// Capture a compact source snippet (first line of the declaration).
			start := node.Parent().StartByte()
			end := node.Parent().EndByte()
			snippet := string(src[start:end])

			if nl := strings.Index(snippet, "\n"); nl > 0 {
				snippet = snippet[:nl]
			}

			sym.Source = strings.TrimSpace(snippet)
			symbols = append(symbols, sym)
		}
	}

	return symbols
}

func (extractor *ContextExtractor) extractCalls(
	rel string, src []byte, root *sitter.Node, enclosing []Symbol,
) map[string][]Symbol {
	query, qErr := sitter.NewQuery(extractor.language, goCallQuery)

	if qErr != nil {
		return nil
	}

	defer query.Close()

	cursor := sitter.NewQueryCursor()
	defer cursor.Close()

	matches := cursor.Matches(query, root, src)
	calleeIdx, _ := query.CaptureIndexForName("callee")

	calls := make(map[string][]Symbol)

	for match := matches.Next(); match != nil; match = matches.Next() {
		for _, node := range match.NodesForCaptureIndex(calleeIdx) {
			callee := node.Utf8Text(src)
			caller := nearestEnclosing(enclosing, uint32(node.StartPosition().Row+1))

			if caller != nil {
				calls[callee] = append(calls[callee], *caller)
			}
		}
	}

	return calls
}

// nearestEnclosing finds the symbol whose declaration starts at or before line
// and is the closest one to line from above.
func nearestEnclosing(symbols []Symbol, line uint32) *Symbol {
	var best *Symbol

	for i := range symbols {
		if symbols[i].Line <= line {
			if best == nil || symbols[i].Line > best.Line {
				best = &symbols[i]
			}
		}
	}

	return best
}

func matchSymbols(symbols []Symbol, keywords []string) []Symbol {
	var matched []Symbol

	lower := make([]string, len(keywords))

	for i, kw := range keywords {
		lower[i] = strings.ToLower(kw)
	}

	for _, sym := range symbols {
		name := strings.ToLower(sym.Name)

		for _, kw := range lower {
			if strings.Contains(name, kw) || strings.Contains(strings.ToLower(sym.File), kw) {
				matched = append(matched, sym)
				break
			}
		}
	}

	return matched
}

func (extractor *ContextExtractor) expandBlast(
	roots []Symbol,
	index callerIndex,
	reachable map[string]struct{},
	callGraph map[string][]Symbol,
	depth int,
) {
	if depth == 0 || len(roots) == 0 {
		return
	}

	var nextWave []Symbol

	for _, sym := range roots {
		reachable[sym.File] = struct{}{}
		callers, ok := index[sym.Name]

		if !ok {
			continue
		}

		callGraph[sym.Name] = append(callGraph[sym.Name], callers...)

		for _, caller := range callers {
			if _, seen := reachable[caller.File]; !seen {
				nextWave = append(nextWave, caller)
			}
		}
	}

	extractor.expandBlast(nextWave, index, reachable, callGraph, depth-1)
}
