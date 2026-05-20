package dispatchaudit

import (
	"fmt"
	"sort"
	"strings"
)

/*
ISAPath names one CPU execution path audited per domain.
*/
type ISAPath string

const (
	ISAPathScalar ISAPath = "scalar"
	ISAPathAVX512 ISAPath = "avx512"
	ISAPathAVX2   ISAPath = "avx2"
	ISAPathSSE2   ISAPath = "sse2"
	ISAPathNEON   ISAPath = "neon"
)

/*
ISARegistration records whether a path is wired for a domain.
*/
type ISARegistration string

const (
	ISARegistered    ISARegistration = "registered"
	ISANotRegistered ISARegistration = "not_registered"
)

/*
DomainDispatchRow is the per-domain CPU dispatch status for one ISA path.
*/
type DomainDispatchRow struct {
	Domain     string
	Scalar     ISARegistration
	AVX512     ISARegistration
	AVX2       ISARegistration
	SSE2       ISARegistration
	NEON       ISARegistration
	Evidence   map[ISAPath][]string
}

/*
CPUDispatchMatrix is the full per-domain registration audit.
*/
type CPUDispatchMatrix struct {
	Rows []DomainDispatchRow
}

/*
BuildCPUDispatchMatrix scans pkg/backend/device/cpu/* operation domains.
*/
func BuildCPUDispatchMatrix() (*CPUDispatchMatrix, error) {
	root, err := locateCPUDomainsRoot()
	if err != nil {
		return nil, err
	}

	domainNames, err := listOperationDomains(root)
	if err != nil {
		return nil, err
	}

	rows := make([]DomainDispatchRow, 0, len(domainNames))

	for _, domainName := range domainNames {
		row, scanErr := scanDomain(root, domainName)
		if scanErr != nil {
			return nil, fmt.Errorf("dispatchaudit: domain %q: %w", domainName, scanErr)
		}

		rows = append(rows, row)
	}

	return &CPUDispatchMatrix{Rows: rows}, nil
}

/*
ValidateCPUDispatchMatrix checks structural invariants on the audit matrix.
*/
func ValidateCPUDispatchMatrix(matrix *CPUDispatchMatrix) error {
	if matrix == nil {
		return fmt.Errorf("dispatchaudit: nil matrix")
	}

	expectedDomains := 32

	if len(matrix.Rows) != expectedDomains {
		return fmt.Errorf(
			"dispatchaudit: want %d operation domains, got %d",
			expectedDomains,
			len(matrix.Rows),
		)
	}

	seen := make(map[string]bool, len(matrix.Rows))

	for _, row := range matrix.Rows {
		if row.Domain == "" {
			return fmt.Errorf("dispatchaudit: empty domain name")
		}

		if seen[row.Domain] {
			return fmt.Errorf("dispatchaudit: duplicate domain %q", row.Domain)
		}

		seen[row.Domain] = true

		if row.Scalar != ISARegistered {
			return fmt.Errorf(
				"dispatchaudit: domain %q: scalar must be registered, got %s",
				row.Domain,
				row.Scalar,
			)
		}
	}

	return nil
}

/*
RenderMarkdown emits a human-readable matrix for docs/cpu-dispatch-matrix.md.
*/
func RenderMarkdown(matrix *CPUDispatchMatrix) string {
	if matrix == nil {
		return ""
	}

	var builder strings.Builder

	builder.WriteString("# CPU dispatch matrix (T1.3)\n\n")
	builder.WriteString("Per-domain registration of scalar (Go reference) and SIMD paths ")
	builder.WriteString("(AVX-512, AVX2, SSE2 on amd64; NEON on arm64). ")
	builder.WriteString("**registered** means at least one assembly file or dispatch-table ")
	builder.WriteString("entry exists in that domain's package; it does not assert full ")
	builder.WriteString("operation coverage.\n\n")
	builder.WriteString("Machine-checkable source: `pkg/backend/device/cpu/dispatchaudit/`, ")
	builder.WriteString("validated by `dispatchaudit_test.go`.\n\n")
	builder.WriteString("Combined coverage (T1.5): [`backend-coverage.md`](./backend-coverage.md).\n\n")

	builder.WriteString("## Summary\n\n")

	counts := summarize(matrix)

	builder.WriteString("| ISA path | Domains registered |\n")
	builder.WriteString("|----------|-------------------:|\n")
	builder.WriteString(fmt.Sprintf("| Scalar (Go) | %d / %d |\n", counts[ISAPathScalar], len(matrix.Rows)))
	builder.WriteString(fmt.Sprintf("| AVX-512 (amd64) | %d / %d |\n", counts[ISAPathAVX512], len(matrix.Rows)))
	builder.WriteString(fmt.Sprintf("| AVX2 (amd64) | %d / %d |\n", counts[ISAPathAVX2], len(matrix.Rows)))
	builder.WriteString(fmt.Sprintf("| SSE2 (amd64) | %d / %d |\n", counts[ISAPathSSE2], len(matrix.Rows)))
	builder.WriteString(fmt.Sprintf("| NEON (arm64) | %d / %d |\n", counts[ISAPathNEON], len(matrix.Rows)))
	builder.WriteString("\n")

	builder.WriteString("## Per-domain matrix\n\n")
	builder.WriteString("| Domain | Scalar | AVX-512 | AVX2 | SSE2 | NEON |\n")
	builder.WriteString("|--------|:------:|:-------:|:----:|:----:|:----:|\n")

	for _, row := range matrix.Rows {
		builder.WriteString(fmt.Sprintf(
			"| %s | %s | %s | %s | %s | %s |\n",
			row.Domain,
			mark(row.Scalar),
			mark(row.AVX512),
			mark(row.AVX2),
			mark(row.SSE2),
			mark(row.NEON),
		))
	}

	builder.WriteString("\n")

	avx512Domains := domainNamesWith(matrix, ISAPathAVX512)
	if len(avx512Domains) > 0 {
		builder.WriteString("### AVX-512 registered domains\n\n")

		for _, domainName := range avx512Domains {
			builder.WriteString(fmt.Sprintf("- `%s`\n", domainName))
		}

		builder.WriteString("\n")
	}

	builder.WriteString("## Registration rules\n\n")
	builder.WriteString("1. **Scalar** — `select_generic.go`, `*_generic.go`, or dispatch tables listing `\"generic\"` / `*Generic` in the domain package.\n")
	builder.WriteString("2. **AVX-512 / AVX2 / SSE2** — `*avx512*`, `*avx2*`, or `*sse2*` assembly under the domain, and/or `select_amd64.go` (or amd64 select shards) declaring matching symbols or `\"avx512\"` / `\"avx2\"` / `\"sse2\"` dispatch names.\n")
	builder.WriteString("3. **NEON** — `*_neon_arm64.s` or `select_arm64.go` with `NEON` symbols or `\"neon\"` dispatch names.\n")
	builder.WriteString("4. **`cpu/neon`** — shared ARM64 helpers; excluded from this domain table.\n")

	return builder.String()
}

func mark(registration ISARegistration) string {
	if registration == ISARegistered {
		return "yes"
	}

	return "—"
}

func summarize(matrix *CPUDispatchMatrix) map[ISAPath]int {
	counts := map[ISAPath]int{
		ISAPathScalar: 0,
		ISAPathAVX512: 0,
		ISAPathAVX2:   0,
		ISAPathSSE2:   0,
		ISAPathNEON:   0,
	}

	for _, row := range matrix.Rows {
		if row.Scalar == ISARegistered {
			counts[ISAPathScalar]++
		}

		if row.AVX512 == ISARegistered {
			counts[ISAPathAVX512]++
		}

		if row.AVX2 == ISARegistered {
			counts[ISAPathAVX2]++
		}

		if row.SSE2 == ISARegistered {
			counts[ISAPathSSE2]++
		}

		if row.NEON == ISARegistered {
			counts[ISAPathNEON]++
		}
	}

	return counts
}

func domainNamesWith(matrix *CPUDispatchMatrix, path ISAPath) []string {
	names := make([]string, 0, 4)

	for _, row := range matrix.Rows {
		if registrationFor(row, path) == ISARegistered {
			names = append(names, row.Domain)
		}
	}

	sort.Strings(names)

	return names
}

func registrationFor(row DomainDispatchRow, path ISAPath) ISARegistration {
	switch path {
	case ISAPathScalar:
		return row.Scalar
	case ISAPathAVX512:
		return row.AVX512
	case ISAPathAVX2:
		return row.AVX2
	case ISAPathSSE2:
		return row.SSE2
	case ISAPathNEON:
		return row.NEON
	default:
		return ISANotRegistered
	}
}
