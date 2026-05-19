package complianceaudit

import (
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
	FindingForbiddenPhrase    FindingKind = "forbidden_phrase"
	FindingCrossISACall       FindingKind = "cross_isa_call"
	FindingScalarTailLoop     FindingKind = "scalar_tail_loop"
	FindingScalarInSIMDKernel FindingKind = "scalar_in_simd_kernel"
	FindingLooseTolerance     FindingKind = "loose_tolerance"
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
	isAmd64 := strings.HasSuffix(path, "_amd64.s")
	isArm64NEON := strings.HasSuffix(path, "_neon_arm64.s") ||
		strings.HasSuffix(path, "_arm64.s")

	if !isAmd64 && !isArm64NEON {
		return nil, nil
	}

	contentBytes, readErr := os.ReadFile(path)
	if readErr != nil {
		return nil, readErr
	}

	lines := strings.Split(string(contentBytes), "\n")
	findings := make([]Finding, 0)
	fileISA := isaFromAssemblyPath(path)

	for lineIndex, line := range lines {
		lineNumber := lineIndex + 1
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
	}

	if isAmd64 {
		findings = append(findings, scanAmd64ScalarTailLoops(path, lines)...)
	}

	if isArm64NEON {
		findings = append(findings, scanNEONScalarHotLoops(path, lines)...)
	}

	return findings, nil
}

func scanAmd64ScalarTailLoops(path string, lines []string) []Finding {
	findings := make([]Finding, 0)
	var inScalarBlock bool
	var scalarLabel string

	for lineIndex, line := range lines {
		lineNumber := lineIndex + 1
		trimmed := strings.TrimSpace(line)

		if strings.HasSuffix(trimmed, "_scalar:") || strings.HasSuffix(trimmed, "_sloop:") {
			inScalarBlock = true
			scalarLabel = trimmed
			continue
		}

		if !inScalarBlock {
			continue
		}

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

	return findings
}

func scanNEONScalarHotLoops(path string, lines []string) []Finding {
	findings := make([]Finding, 0)

	for lineIndex, line := range lines {
		trimmed := strings.TrimSpace(line)

		if !strings.HasSuffix(trimmed, ":") {
			continue
		}

		if strings.Contains(trimmed, "_tail:") ||
			strings.Contains(trimmed, "_scalar_loop:") ||
			strings.Contains(trimmed, "_done:") ||
			strings.Contains(trimmed, "_reduce:") ||
			strings.Contains(trimmed, "_finalize:") {
			continue
		}

		if !neonHotLoopLabel(trimmed) {
			continue
		}

		block := collectBackwardBranchBlock(lines, lineIndex)
		if neonBlockIsScalarHotLoop(block) {
			findings = append(findings, Finding{
				Kind:    FindingScalarInSIMDKernel,
				Path:    path,
				Line:    lineIndex + 1,
				Summary: "scalar FP hot loop at " + trimmed,
			})
		}
	}

	return findings
}

func neonHotLoopLabel(label string) bool {
	if strings.Contains(label, "_loop") {
		return true
	}

	if strings.Contains(label, "_col_loop:") {
		return true
	}

	return strings.Contains(label, "_kh_loop:") ||
		strings.Contains(label, "_kw_loop:")
}

func collectBackwardBranchBlock(lines []string, startIndex int) []string {
	block := make([]string, 0, 32)
	startLabel := strings.TrimSuffix(strings.TrimSpace(lines[startIndex]), ":")

	for offset := 1; offset < len(lines) && offset < 48; offset++ {
		lineIndex := startIndex + offset
		trimmed := strings.TrimSpace(lines[lineIndex])

		block = append(block, trimmed)

		if strings.HasPrefix(trimmed, "B ") || strings.HasPrefix(trimmed, "CBNZ ") ||
			strings.HasPrefix(trimmed, "CBZ ") {
			if strings.Contains(trimmed, startLabel) {
				break
			}
		}
	}

	return block
}

func neonBlockIsScalarHotLoop(block []string) bool {
	hasVectorLoad := false
	hasVectorAccum := false
	hasScalarHotAccum := false

	for _, line := range block {
		if neonLineHasVectorLoad(line) {
			hasVectorLoad = true
		}

		if neonLineHasVectorAccum(line) {
			hasVectorAccum = true
		}

		if neonLineHasScalarHotAccum(line) {
			hasScalarHotAccum = true
		}
	}

	if !hasVectorLoad || hasVectorAccum {
		return false
	}

	return hasScalarHotAccum
}

func neonLineHasVectorLoad(line string) bool {
	return strings.Contains(line, "VLD1") ||
		strings.Contains(line, "VLD2") ||
		strings.Contains(line, "VLD3") ||
		strings.Contains(line, "VLD4")
}

func neonLineHasVectorAccum(line string) bool {
	if strings.Contains(line, "VFADD") ||
		strings.Contains(line, "VFMUL") ||
		strings.Contains(line, "VFMLA") ||
		strings.Contains(line, "VFMAX") ||
		strings.Contains(line, "VFMIN") {
		return true
	}

	return strings.Contains(line, "WORD $") &&
		(strings.Contains(line, "0x4E20") ||
			strings.Contains(line, "0x6E20") ||
			strings.Contains(line, "0x4E21"))
}

func neonLineHasScalarHotAccum(line string) bool {
	if strings.Contains(line, "FADDS") || strings.Contains(line, "FMULS") {
		return true
	}

	return strings.Contains(line, "FMOVS (R") && strings.Contains(line, "F1")
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
