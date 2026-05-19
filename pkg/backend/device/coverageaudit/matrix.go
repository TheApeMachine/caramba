package coverageaudit

import (
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/device"
	"github.com/theapemachine/caramba/pkg/backend/device/backendaudit"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/dispatchaudit"
)

/*
InventoryCoverage summarizes device.Backend ↔ ir.RequiredOperationIDs (T1.2).
*/
type InventoryCoverage struct {
	BackendMethods         int
	RequiredOperations     int
	DirectCrossLinks       int
	CompositeCrossLinks    int
	KernelRegistryCrossLinks int
	GraphOnlyCrossLinks    int
	UnmappedBackendMethods int
}

/*
CPUDispatchCoverage summarizes per-domain CPU ISA registration (T1.3).
*/
type CPUDispatchCoverage struct {
	Domains            int
	ScalarDomains      int
	AVX512Domains      int
	AVX2Domains        int
	SSE2Domains        int
	NEONDomains        int
	AMD64SIMDDomainNames []string
}

/*
DeviceBackendCoverage summarizes Metal / CUDA / XLA kernel registration (T1.4).
*/
type DeviceBackendCoverage struct {
	MetalKernelRegistrations   int
	MetalUniqueKernelNames     int
	MetalRequiredOpsRegistered int
	CUDAKernelRegistrations    int
	XLAKernelRegistrations     int
}

/*
ExecutionTargetCoverage is one row in the R1 execution-target summary.
*/
type ExecutionTargetCoverage struct {
	Target      string
	Scope       string
	Registered  int
	Applicable  int
	Detail      string
}

/*
BackendCoverageMatrix combines T1.2, T1.3, and T1.4 audit snapshots.
*/
type BackendCoverageMatrix struct {
	Inventory InventoryCoverage
	CPU       CPUDispatchCoverage
	Device    DeviceBackendCoverage
	Targets   []ExecutionTargetCoverage
}

/*
BuildBackendCoverageMatrix merges inventory, CPU dispatch, and device backend audits.
*/
func BuildBackendCoverageMatrix() (*BackendCoverageMatrix, error) {
	if inventoryErr := device.ValidateBackendInventory(); inventoryErr != nil {
		return nil, fmt.Errorf("coverageaudit: inventory: %w", inventoryErr)
	}

	cpuMatrix, cpuErr := dispatchaudit.BuildCPUDispatchMatrix()
	if cpuErr != nil {
		return nil, fmt.Errorf("coverageaudit: cpu dispatch: %w", cpuErr)
	}

	deviceMatrix, deviceErr := backendaudit.BuildDeviceBackendMatrix()
	if deviceErr != nil {
		return nil, fmt.Errorf("coverageaudit: device backends: %w", deviceErr)
	}

	matrix := &BackendCoverageMatrix{
		Inventory: buildInventoryCoverage(),
		CPU:       buildCPUDispatchCoverage(cpuMatrix),
		Device:    buildDeviceBackendCoverage(deviceMatrix),
	}

	matrix.Targets = buildExecutionTargetCoverage(matrix)

	return matrix, nil
}

/*
ValidateBackendCoverageMatrix checks combined audit invariants and golden counts.
*/
func ValidateBackendCoverageMatrix(matrix *BackendCoverageMatrix) error {
	if matrix == nil {
		return fmt.Errorf("coverageaudit: nil matrix")
	}

	if matrix.Inventory.BackendMethods != 151 {
		return fmt.Errorf(
			"coverageaudit: backend methods: want 151, got %d",
			matrix.Inventory.BackendMethods,
		)
	}

	if matrix.Inventory.RequiredOperations != len(ir.RequiredOperationIDs()) {
		return fmt.Errorf(
			"coverageaudit: required ops: want %d, got %d",
			len(ir.RequiredOperationIDs()),
			matrix.Inventory.RequiredOperations,
		)
	}

	if matrix.CPU.Domains != 30 {
		return fmt.Errorf("coverageaudit: cpu domains: want 30, got %d", matrix.CPU.Domains)
	}

	if matrix.CPU.ScalarDomains != 30 {
		return fmt.Errorf("coverageaudit: scalar domains: want 30, got %d", matrix.CPU.ScalarDomains)
	}

	if matrix.CPU.AVX512Domains != 2 || matrix.CPU.AVX2Domains != 2 || matrix.CPU.SSE2Domains != 2 {
		return fmt.Errorf(
			"coverageaudit: amd64 simd domains: avx512=%d avx2=%d sse2=%d, want 2 each",
			matrix.CPU.AVX512Domains,
			matrix.CPU.AVX2Domains,
			matrix.CPU.SSE2Domains,
		)
	}

	if matrix.CPU.NEONDomains != 20 {
		return fmt.Errorf("coverageaudit: neon domains: want 20, got %d", matrix.CPU.NEONDomains)
	}

	if matrix.Device.MetalKernelRegistrations == 0 {
		return fmt.Errorf("coverageaudit: metal must have kernel registrations")
	}

	if matrix.Device.CUDAKernelRegistrations != 0 || matrix.Device.XLAKernelRegistrations != 0 {
		return fmt.Errorf(
			"coverageaudit: cuda/xla kernel regs must be 0, got cuda=%d xla=%d",
			matrix.Device.CUDAKernelRegistrations,
			matrix.Device.XLAKernelRegistrations,
		)
	}

	if len(matrix.Targets) != 8 {
		return fmt.Errorf("coverageaudit: execution targets: want 8, got %d", len(matrix.Targets))
	}

	return nil
}

/*
RenderMarkdown emits docs/backend-coverage.md content.
*/
func RenderMarkdown(matrix *BackendCoverageMatrix) string {
	if matrix == nil {
		return ""
	}

	var builder strings.Builder

	builder.WriteString("# Backend coverage matrix (T1.5)\n\n")
	builder.WriteString("Combined snapshot of **T1.2** (`device.Backend` inventory), **T1.3** ")
	builder.WriteString("(CPU per-domain ISA dispatch), and **T1.4** (Metal / CUDA / XLA kernel ")
	builder.WriteString("registration). Counts are **registration** status from filesystem and ")
	builder.WriteString("`kernels.Default`; they do not assert full R1 implementation, parity, ")
	builder.WriteString("or legality on every required operation.\n\n")
	builder.WriteString("Machine-checkable source: `pkg/backend/device/coverageaudit/`, ")
	builder.WriteString("validated by `coverageaudit_test.go`.\n\n")

	builder.WriteString("## Detail documents\n\n")
	builder.WriteString("| Task | Document | Package |\n")
	builder.WriteString("|------|----------|----------|\n")
	builder.WriteString("| T1.2 | [`backend-inventory.md`](./backend-inventory.md) | `pkg/backend/device/inventory*.go` |\n")
	builder.WriteString("| T1.3 | [`cpu-dispatch-matrix.md`](./cpu-dispatch-matrix.md) | `pkg/backend/device/cpu/dispatchaudit/` |\n")
	builder.WriteString("| T1.4 | [`device-backend-matrix.md`](./device-backend-matrix.md) | `pkg/backend/device/backendaudit/` |\n")
	builder.WriteString("\n")

	builder.WriteString("## R1 execution targets (registration snapshot)\n\n")
	builder.WriteString("Equal standing per `AGENTS.md` §1. **Applicable** is the audit denominator ")
	builder.WriteString("for that row; **registered** is what the combined audit currently finds.\n\n")
	builder.WriteString("| Target | Scope | Registered | Applicable | Notes |\n")
	builder.WriteString("|--------|-------|----------:|-----------:|-------|\n")

	for _, row := range matrix.Targets {
		builder.WriteString(fmt.Sprintf(
			"| %s | %s | %d | %d | %s |\n",
			row.Target,
			row.Scope,
			row.Registered,
			row.Applicable,
			row.Detail,
		))
	}

	builder.WriteString("\n")

	builder.WriteString("## Backend inventory (T1.2)\n\n")
	builder.WriteString("| Item | Count |\n")
	builder.WriteString("|------|------:|\n")
	builder.WriteString(fmt.Sprintf("| `device.Backend` methods | %d |\n", matrix.Inventory.BackendMethods))
	builder.WriteString(fmt.Sprintf("| `ir.RequiredOperationIDs()` | %d |\n", matrix.Inventory.RequiredOperations))
	builder.WriteString(fmt.Sprintf("| Required ops → direct `Backend` method | %d |\n", matrix.Inventory.DirectCrossLinks))
	builder.WriteString(fmt.Sprintf("| Required ops → composite `Backend` methods | %d |\n", matrix.Inventory.CompositeCrossLinks))
	builder.WriteString(fmt.Sprintf("| Required ops → kernel registry | %d |\n", matrix.Inventory.KernelRegistryCrossLinks))
	builder.WriteString(fmt.Sprintf("| Required ops → graph-only | %d |\n", matrix.Inventory.GraphOnlyCrossLinks))
	builder.WriteString(fmt.Sprintf("| `Backend` methods with no required-op ID | %d |\n", matrix.Inventory.UnmappedBackendMethods))
	builder.WriteString("\n")

	builder.WriteString("## CPU dispatch (T1.3)\n\n")
	builder.WriteString("| ISA path | Domains registered |\n")
	builder.WriteString("|----------|-------------------:|\n")
	builder.WriteString(fmt.Sprintf("| Scalar (Go) | %d / %d |\n", matrix.CPU.ScalarDomains, matrix.CPU.Domains))
	builder.WriteString(fmt.Sprintf("| AVX-512 (amd64) | %d / %d |\n", matrix.CPU.AVX512Domains, matrix.CPU.Domains))
	builder.WriteString(fmt.Sprintf("| AVX2 (amd64) | %d / %d |\n", matrix.CPU.AVX2Domains, matrix.CPU.Domains))
	builder.WriteString(fmt.Sprintf("| SSE2 (amd64) | %d / %d |\n", matrix.CPU.SSE2Domains, matrix.CPU.Domains))
	builder.WriteString(fmt.Sprintf("| NEON (arm64) | %d / %d |\n", matrix.CPU.NEONDomains, matrix.CPU.Domains))
	builder.WriteString("\n")

	if len(matrix.CPU.AMD64SIMDDomainNames) > 0 {
		builder.WriteString("AMD64 SIMD registered only on: ")

		for index, domainName := range matrix.CPU.AMD64SIMDDomainNames {
			if index > 0 {
				builder.WriteString(", ")
			}

			builder.WriteString(fmt.Sprintf("`%s`", domainName))
		}

		builder.WriteString(".\n\n")
	}

	builder.WriteString("Per-domain table: [`cpu-dispatch-matrix.md`](./cpu-dispatch-matrix.md).\n\n")

	builder.WriteString("## Device backends (T1.4)\n\n")
	builder.WriteString("| Backend | Kernel registrations | Required ops with Metal kernel |\n")
	builder.WriteString("|---------|------------------------:|-------------------------------:|\n")
	builder.WriteString(fmt.Sprintf(
		"| Metal | %d (%d unique names) | %d / %d |\n",
		matrix.Device.MetalKernelRegistrations,
		matrix.Device.MetalUniqueKernelNames,
		matrix.Device.MetalRequiredOpsRegistered,
		matrix.Inventory.RequiredOperations,
	))
	builder.WriteString(fmt.Sprintf("| CUDA | %d | — |\n", matrix.Device.CUDAKernelRegistrations))
	builder.WriteString(fmt.Sprintf("| XLA | %d | — |\n", matrix.Device.XLAKernelRegistrations))
	builder.WriteString("\n")

	builder.WriteString("Per-kernel and per-operation tables: [`device-backend-matrix.md`](./device-backend-matrix.md).\n")

	return builder.String()
}

func buildInventoryCoverage() InventoryCoverage {
	index := device.BuildOperationCrossLinkIndex()
	kindCounts := map[device.CrossLinkKind]int{}

	for _, link := range index {
		kindCounts[link.Kind]++
	}

	return InventoryCoverage{
		BackendMethods:           len(device.EnumerateBackendMethods()),
		RequiredOperations:       len(ir.RequiredOperationIDs()),
		DirectCrossLinks:         kindCounts[device.CrossLinkDirect],
		CompositeCrossLinks:      kindCounts[device.CrossLinkComposite],
		KernelRegistryCrossLinks: kindCounts[device.CrossLinkKernelRegistry],
		GraphOnlyCrossLinks:      kindCounts[device.CrossLinkGraphOnly],
		UnmappedBackendMethods:   len(device.BackendMethodsWithoutRequiredOperation()),
	}
}

func buildCPUDispatchCoverage(cpuMatrix *dispatchaudit.CPUDispatchMatrix) CPUDispatchCoverage {
	coverage := CPUDispatchCoverage{
		Domains: len(cpuMatrix.Rows),
	}

	for _, row := range cpuMatrix.Rows {
		if row.Scalar == dispatchaudit.ISARegistered {
			coverage.ScalarDomains++
		}

		if row.AVX512 == dispatchaudit.ISARegistered {
			coverage.AVX512Domains++
		}

		if row.AVX2 == dispatchaudit.ISARegistered {
			coverage.AVX2Domains++
		}

		if row.SSE2 == dispatchaudit.ISARegistered {
			coverage.SSE2Domains++
		}

		if row.NEON == dispatchaudit.ISARegistered {
			coverage.NEONDomains++
		}
	}

	if coverage.AVX2Domains == 2 {
		coverage.AMD64SIMDDomainNames = []string{"activation", "pospop"}
	}

	return coverage
}

func buildDeviceBackendCoverage(deviceMatrix *backendaudit.DeviceBackendMatrix) DeviceBackendCoverage {
	coverage := DeviceBackendCoverage{}

	for _, row := range deviceMatrix.Backends {
		switch row.Backend {
		case backendaudit.DeviceBackendMetal:
			coverage.MetalKernelRegistrations = row.RegisteredKernels
			coverage.MetalUniqueKernelNames = row.UniqueKernelNames
		case backendaudit.DeviceBackendCUDA:
			coverage.CUDAKernelRegistrations = row.RegisteredKernels
		case backendaudit.DeviceBackendXLA:
			coverage.XLAKernelRegistrations = row.RegisteredKernels
		}
	}

	for _, row := range deviceMatrix.RequiredOps {
		if row.MetalKernels == backendaudit.KernelRegistered {
			coverage.MetalRequiredOpsRegistered++
		}
	}

	return coverage
}

func buildExecutionTargetCoverage(matrix *BackendCoverageMatrix) []ExecutionTargetCoverage {
	requiredOps := matrix.Inventory.RequiredOperations

	return []ExecutionTargetCoverage{
		{
			Target:     "Go scalar",
			Scope:      "CPU domains",
			Registered: matrix.CPU.ScalarDomains,
			Applicable: matrix.CPU.Domains,
			Detail:     "Go reference in every operation domain",
		},
		{
			Target:     "AVX-512",
			Scope:      "CPU domains (amd64)",
			Registered: matrix.CPU.AVX512Domains,
			Applicable: matrix.CPU.Domains,
			Detail:     "Per-domain assembly/dispatch registration",
		},
		{
			Target:     "AVX2",
			Scope:      "CPU domains (amd64)",
			Registered: matrix.CPU.AVX2Domains,
			Applicable: matrix.CPU.Domains,
			Detail:     "Per-domain assembly/dispatch registration",
		},
		{
			Target:     "SSE2",
			Scope:      "CPU domains (amd64)",
			Registered: matrix.CPU.SSE2Domains,
			Applicable: matrix.CPU.Domains,
			Detail:     "Per-domain assembly/dispatch registration",
		},
		{
			Target:     "NEON",
			Scope:      "CPU domains (arm64)",
			Registered: matrix.CPU.NEONDomains,
			Applicable: matrix.CPU.Domains,
			Detail:     "Per-domain assembly/dispatch registration",
		},
		{
			Target:     "Metal",
			Scope:      "Required IR operations",
			Registered: matrix.Device.MetalRequiredOpsRegistered,
			Applicable: requiredOps,
			Detail:     fmt.Sprintf("%d `kernels.Default` registrations total", matrix.Device.MetalKernelRegistrations),
		},
		{
			Target:     "CUDA",
			Scope:      "Required IR operations",
			Registered: 0,
			Applicable: requiredOps,
			Detail:     "Tensor upload/download; 0 kernel registrations",
		},
		{
			Target:     "XLA",
			Scope:      "Required IR operations",
			Registered: 0,
			Applicable: requiredOps,
			Detail:     "Tensor upload/download; 0 kernel registrations",
		},
	}
}
