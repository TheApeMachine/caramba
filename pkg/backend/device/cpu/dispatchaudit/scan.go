package dispatchaudit

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
)

var (
	amd64SelectName = regexp.MustCompile(`^select_.*amd64.*\.go$`)
	arm64SelectName = regexp.MustCompile(`^select_.*arm64.*\.go$`)
)

/*
locateCPUDomainsRoot returns pkg/backend/device/cpu (parent of dispatchaudit).
*/
func locateCPUDomainsRoot() (string, error) {
	_, filePath, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("dispatchaudit: runtime.Caller failed")
	}

	root := filepath.Clean(filepath.Join(filepath.Dir(filePath), ".."))

	info, err := os.Stat(root)
	if err != nil {
		return "", fmt.Errorf("dispatchaudit: stat cpu root: %w", err)
	}

	if !info.IsDir() {
		return "", fmt.Errorf("dispatchaudit: cpu root is not a directory: %s", root)
	}

	return root, nil
}

func listOperationDomains(cpuRoot string) ([]string, error) {
	entries, err := os.ReadDir(cpuRoot)
	if err != nil {
		return nil, err
	}

	names := make([]string, 0, len(entries))

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		name := entry.Name()

		if isExcludedDomain(name) {
			continue
		}

		names = append(names, name)
	}

	sort.Strings(names)

	return names, nil
}

func isExcludedDomain(name string) bool {
	if strings.HasPrefix(name, ".") {
		return true
	}

	switch name {
	case "neon", "dispatchaudit":
		return true
	default:
		return false
	}
}

func scanDomain(cpuRoot, domainName string) (DomainDispatchRow, error) {
	domainPath := filepath.Join(cpuRoot, domainName)

	entries, err := os.ReadDir(domainPath)
	if err != nil {
		return DomainDispatchRow{}, err
	}

	evidence := map[ISAPath][]string{
		ISAPathScalar: {},
		ISAPathAVX512: {},
		ISAPathAVX2:   {},
		ISAPathSSE2:   {},
		ISAPathNEON:   {},
	}

	var amd64SelectContents []string
	var arm64SelectContents []string
	var genericSelectContents []string

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		fileName := entry.Name()

		if strings.HasSuffix(fileName, ".s") {
			classifyAssembly(fileName, evidence)
			continue
		}

		if !strings.HasSuffix(fileName, ".go") {
			continue
		}

		if fileName == "select_generic.go" || strings.HasSuffix(fileName, "_generic.go") {
			evidence[ISAPathScalar] = appendUnique(evidence[ISAPathScalar], fileName)
		}

		if amd64SelectName.MatchString(fileName) {
			body, readErr := os.ReadFile(filepath.Join(domainPath, fileName))
			if readErr != nil {
				return DomainDispatchRow{}, readErr
			}

			amd64SelectContents = append(amd64SelectContents, string(body))
			continue
		}

		if arm64SelectName.MatchString(fileName) {
			body, readErr := os.ReadFile(filepath.Join(domainPath, fileName))
			if readErr != nil {
				return DomainDispatchRow{}, readErr
			}

			arm64SelectContents = append(arm64SelectContents, string(body))
			continue
		}

		if fileName == "select_generic.go" {
			body, readErr := os.ReadFile(filepath.Join(domainPath, fileName))
			if readErr != nil {
				return DomainDispatchRow{}, readErr
			}

			genericSelectContents = append(genericSelectContents, string(body))
		}
	}

	for _, body := range amd64SelectContents {
		classifySelectAMD64(body, evidence)
	}

	for _, body := range arm64SelectContents {
		classifySelectARM64(body, evidence)
	}

	for _, body := range genericSelectContents {
		classifySelectGeneric(body, evidence)
	}

	if len(evidence[ISAPathScalar]) == 0 {
		if domainHasScalarGo(domainPath, entries) {
			evidence[ISAPathScalar] = appendUnique(evidence[ISAPathScalar], "go_implementation")
		}
	}

	row := DomainDispatchRow{
		Domain:   domainName,
		Evidence: evidence,
	}

	row.Scalar = registrationFromEvidence(evidence[ISAPathScalar])
	row.AVX512 = registrationFromEvidence(evidence[ISAPathAVX512])
	row.AVX2 = registrationFromEvidence(evidence[ISAPathAVX2])
	row.SSE2 = registrationFromEvidence(evidence[ISAPathSSE2])
	row.NEON = registrationFromEvidence(evidence[ISAPathNEON])

	return row, nil
}

func classifyAssembly(fileName string, evidence map[ISAPath][]string) {
	lowerName := strings.ToLower(fileName)

	if strings.Contains(lowerName, "avx512") {
		evidence[ISAPathAVX512] = appendUnique(evidence[ISAPathAVX512], fileName)
	}

	if strings.Contains(lowerName, "avx2") {
		evidence[ISAPathAVX2] = appendUnique(evidence[ISAPathAVX2], fileName)
	}

	if strings.Contains(lowerName, "sse2") {
		evidence[ISAPathSSE2] = appendUnique(evidence[ISAPathSSE2], fileName)
	}

	if strings.Contains(lowerName, "neon") && strings.Contains(lowerName, "arm64") {
		evidence[ISAPathNEON] = appendUnique(evidence[ISAPathNEON], fileName)
	}
}

func classifySelectAMD64(body string, evidence map[ISAPath][]string) {
	if containsISAToken(body, "avx512") {
		evidence[ISAPathAVX512] = appendUnique(evidence[ISAPathAVX512], "select_amd64:avx512")
	}

	if containsISAToken(body, "avx2") {
		evidence[ISAPathAVX2] = appendUnique(evidence[ISAPathAVX2], "select_amd64:avx2")
	}

	if containsISAToken(body, "sse2") {
		evidence[ISAPathSSE2] = appendUnique(evidence[ISAPathSSE2], "select_amd64:sse2")
	}

	if containsISAToken(body, "generic") {
		evidence[ISAPathScalar] = appendUnique(evidence[ISAPathScalar], "select_amd64:generic")
	}
}

func classifySelectARM64(body string, evidence map[ISAPath][]string) {
	if containsISAToken(body, "neon") {
		evidence[ISAPathNEON] = appendUnique(evidence[ISAPathNEON], "select_arm64:neon")
	}

	if containsISAToken(body, "generic") {
		evidence[ISAPathScalar] = appendUnique(evidence[ISAPathScalar], "select_arm64:generic")
	}
}

func classifySelectGeneric(body string, evidence map[ISAPath][]string) {
	if containsISAToken(body, "generic") {
		evidence[ISAPathScalar] = appendUnique(evidence[ISAPathScalar], "select_generic")
	}
}

func containsISAToken(body, token string) bool {
	lowerBody := strings.ToLower(body)
	upperToken := strings.ToUpper(token)

	if strings.Contains(lowerBody, `"`+token+`"`) {
		return true
	}

	if strings.Contains(body, upperToken) {
		return true
	}

	return strings.Contains(lowerBody, token)
}

func domainHasScalarGo(domainPath string, entries []os.DirEntry) bool {
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		fileName := entry.Name()

		if !strings.HasSuffix(fileName, ".go") {
			continue
		}

		if strings.HasSuffix(fileName, "_test.go") {
			continue
		}

		if amd64SelectName.MatchString(fileName) || arm64SelectName.MatchString(fileName) {
			continue
		}

		fullPath := filepath.Join(domainPath, fileName)

		body, err := os.ReadFile(fullPath)
		if err != nil {
			continue
		}

		text := string(body)

		if strings.Contains(text, "package ") {
			return true
		}
	}

	return false
}

func registrationFromEvidence(items []string) ISARegistration {
	if len(items) == 0 {
		return ISANotRegistered
	}

	return ISARegistered
}

func appendUnique(slice []string, value string) []string {
	for _, existing := range slice {
		if existing == value {
			return slice
		}
	}

	return append(slice, value)
}
