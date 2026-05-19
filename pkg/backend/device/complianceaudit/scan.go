package complianceaudit

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

/*
FindingKind classifies a backend compliance violation.
*/
type FindingKind string

const (
	FindingForbiddenPhrase FindingKind = "forbidden_phrase"
	FindingCrossISACall    FindingKind = "cross_isa_call"
	FindingScalarTailLoop  FindingKind = "scalar_tail_loop"
	FindingLooseTolerance  FindingKind = "loose_tolerance"
)

/*
Finding is one compliance audit hit.
*/
type Finding struct {
	Kind     FindingKind
	Path     string
	Line     int
	Summary  string
}

var forbiddenPhrasePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\bfor now\b`),
	regexp.MustCompile(`(?i)\bshortcut\b`),
	regexp.MustCompile(`(?i)\bpreview\b`),
	regexp.MustCompile(`(?i)approximation acceptable`),
	regexp.MustCompile(`(?i)required vs optional`),
	regexp.MustCompile(`(?i)fallback to go\b`),
	regexp.MustCompile(`(?i)\bfallback implementation\b`),
}

var crossISACallPattern = regexp.MustCompile(`CALL ·([A-Za-z0-9_]+)\(SB\)`)

var looseAbsTolerancePattern = regexp.MustCompile(
	`math\.Abs\([^)]+\)\s*>\s*(1e-[0-5]|0\.0*1[^0-9])`,
)

var shouldAlmostLoosePattern = regexp.MustCompile(
	`ShouldAlmostEqual,\s*[^,]+,\s*(1e-[0-5]|0\.0*1[^0-9])`,
)

var shouldBeLessThanLoosePattern = regexp.MustCompile(
	`ShouldBeLessThan,\s*(1e-[0-5]|0\.0*1[^0-9])`,
)

var assertNearLoosePattern = regexp.MustCompile(
	`assertFloat32SlicesNear\([^,]+,[^,]+,[^,]+,\s*(1e-[0-5]|0\.0*1[^0-9])`,
)

/*
ScanCPUTree audits pkg/backend/device/cpu for compliance violations.
*/
func ScanCPUTree(cpuRoot string) ([]Finding, error) {
	var findings []Finding

	walkErr := filepath.WalkDir(cpuRoot, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if entry.IsDir() {
			return nil
		}

		switch filepath.Ext(path) {
		case ".go":
			fileFindings, scanErr := scanGoFile(path)
			if scanErr != nil {
				return scanErr
			}

			findings = append(findings, fileFindings...)
		case ".s":
			fileFindings, scanErr := scanAssemblyFile(path)
			if scanErr != nil {
				return scanErr
			}

			findings = append(findings, fileFindings...)
		}

		return nil
	})

	if walkErr != nil {
		return nil, walkErr
	}

	return findings, nil
}

func scanGoFile(path string) ([]Finding, error) {
	contentBytes, readErr := os.ReadFile(path)
	if readErr != nil {
		return nil, readErr
	}

	content := string(contentBytes)
	lines := strings.Split(content, "\n")
	findings := make([]Finding, 0)

	for lineIndex, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "//") {
			if phraseFinding := matchForbiddenPhrase(path, lineIndex+1, trimmed); phraseFinding != nil {
				findings = append(findings, *phraseFinding)
			}
		}

		if strings.HasSuffix(path, "_test.go") {
			if looseFinding := matchLooseTolerance(path, lineIndex+1, line); looseFinding != nil {
				findings = append(findings, *looseFinding)
			}
		}
	}

	if !strings.HasSuffix(path, "_test.go") {
		return findings, nil
	}

	fileSet := token.NewFileSet()
	parsed, parseErr := parser.ParseFile(fileSet, path, contentBytes, 0)
	if parseErr != nil {
		return findings, nil
	}

	ast.Inspect(parsed, func(node ast.Node) bool {
		callExpr, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}

		ident, ok := callExpr.Fun.(*ast.Ident)
		if !ok || ident.Name != "assertFloat32SlicesNear" {
			return true
		}

		if len(callExpr.Args) < 4 {
			return true
		}

		literal, ok := callExpr.Args[3].(*ast.BasicLit)
		if !ok {
			return true
		}

		if isLooseLiteral(literal.Value) {
			position := fileSet.Position(callExpr.Pos())
			findings = append(findings, Finding{
				Kind:    FindingLooseTolerance,
				Path:    path,
				Line:    position.Line,
				Summary: "assertFloat32SlicesNear uses absolute epsilon " + literal.Value,
			})
		}

		return true
	})

	return findings, nil
}

func scanAssemblyFile(path string) ([]Finding, error) {
	if !strings.HasSuffix(path, "_amd64.s") {
		return nil, nil
	}

	fileHandle, openErr := os.Open(path)
	if openErr != nil {
		return nil, openErr
	}
	defer fileHandle.Close()

	findings := make([]Finding, 0)
	scanner := bufio.NewScanner(fileHandle)
	fileISA := isaFromAssemblyPath(path)

	var lineNumber int
	var inScalarBlock bool
	var scalarLabel string

	for scanner.Scan() {
		lineNumber++
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "//") {
			if phraseFinding := matchForbiddenPhrase(path, lineNumber, trimmed); phraseFinding != nil {
				findings = append(findings, *phraseFinding)
			}
		}

		if callMatch := crossISACallPattern.FindStringSubmatch(trimmed); len(callMatch) == 2 {
			targetISA := isaFromSymbol(callMatch[1])
			if targetISA != "" && fileISA != "" && targetISA != fileISA {
				findings = append(findings, Finding{
					Kind:    FindingCrossISACall,
					Path:    path,
					Line:    lineNumber,
					Summary: fmt.Sprintf("CALL %s from %s file", callMatch[1], fileISA),
				})
			}
		}

		if strings.HasSuffix(trimmed, "_scalar:") || strings.HasSuffix(trimmed, "_sloop:") {
			inScalarBlock = true
			scalarLabel = trimmed
			continue
		}

		if inScalarBlock {
			if strings.Contains(trimmed, "MOVSS") || strings.Contains(trimmed, "MOVSD") {
				findings = append(findings, Finding{
					Kind:    FindingScalarTailLoop,
					Path:    path,
					Line:    lineNumber,
					Summary: "scalar tail loop at " + scalarLabel,
				})
				inScalarBlock = false
			}

			if strings.HasSuffix(trimmed, "_done:") {
				inScalarBlock = false
			}
		}
	}

	return findings, scanner.Err()
}

func matchForbiddenPhrase(path string, lineNumber int, line string) *Finding {
	for _, pattern := range forbiddenPhrasePatterns {
		if pattern.MatchString(line) {
			return &Finding{
				Kind:    FindingForbiddenPhrase,
				Path:    path,
				Line:    lineNumber,
				Summary: pattern.String(),
			}
		}
	}

	return nil
}

func matchLooseTolerance(path string, lineNumber int, line string) *Finding {
	if looseAbsTolerancePattern.MatchString(line) {
		return &Finding{
			Kind:    FindingLooseTolerance,
			Path:    path,
			Line:    lineNumber,
			Summary: "absolute tolerance comparison",
		}
	}

	if shouldAlmostLoosePattern.MatchString(line) {
		return &Finding{
			Kind:    FindingLooseTolerance,
			Path:    path,
			Line:    lineNumber,
			Summary: "ShouldAlmostEqual with loose epsilon",
		}
	}

	if shouldBeLessThanLoosePattern.MatchString(line) {
		return &Finding{
			Kind:    FindingLooseTolerance,
			Path:    path,
			Line:    lineNumber,
			Summary: "ShouldBeLessThan with loose epsilon",
		}
	}

	if assertNearLoosePattern.MatchString(line) {
		return &Finding{
			Kind:    FindingLooseTolerance,
			Path:    path,
			Line:    lineNumber,
			Summary: "assertFloat32SlicesNear with loose epsilon",
		}
	}

	return nil
}

func isLooseLiteral(literal string) bool {
	switch literal {
	case "1e-2", "1e-3", "1e-4", "1e-5", "0.1", "0.01", "0.001":
		return true
	default:
		return false
	}
}

func isaFromAssemblyPath(path string) string {
	lower := strings.ToLower(path)

	switch {
	case strings.Contains(lower, "avx512"):
		return "avx512"
	case strings.Contains(lower, "avx2"):
		return "avx2"
	case strings.Contains(lower, "sse2"):
		return "sse2"
	case strings.Contains(lower, "neon"):
		return "neon"
	default:
		return ""
	}
}

func isaFromSymbol(symbol string) string {
	upper := strings.ToUpper(symbol)

	switch {
	case strings.Contains(upper, "AVX512"):
		return "avx512"
	case strings.Contains(upper, "AVX2"):
		return "avx2"
	case strings.Contains(upper, "SSE2"):
		return "sse2"
	case strings.Contains(upper, "NEON"):
		return "neon"
	default:
		return ""
	}
}
