package backendaudit

import (
	"fmt"
	"sort"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/backend/device"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
DeviceBackendName identifies Metal, CUDA, or XLA tensor backends.
*/
type DeviceBackendName string

const (
	DeviceBackendMetal DeviceBackendName = "metal"
	DeviceBackendCUDA  DeviceBackendName = "cuda"
	DeviceBackendXLA   DeviceBackendName = "xla"
)

/*
KernelRegistrationStatus records whether a kernel is registered for a backend.
*/
type KernelRegistrationStatus string

const (
	KernelRegistered    KernelRegistrationStatus = "registered"
	KernelNotRegistered KernelRegistrationStatus = "not_registered"
)

/*
DeviceBackendRow summarizes one tensor backend's registration surface.
*/
type DeviceBackendRow struct {
	Backend            DeviceBackendName
	SupportedDTypes    []dtype.DType
	KernelSources      int
	DarwinDispatchFiles int
	StubDispatchFiles  int
	RegisteredKernels  int
	UniqueKernelNames  int
	TensorAPIMethods   int
}

/*
KernelCoverageRow maps one kernels.Default name to per-backend registration.
*/
type KernelCoverageRow struct {
	KernelName string
	Metal      KernelRegistrationStatus
	CUDA       KernelRegistrationStatus
	XLA        KernelRegistrationStatus
	DTypeCount int
}

/*
RequiredOpCoverageRow records Metal kernel coverage for one required IR op.
*/
type RequiredOpCoverageRow struct {
	OperationID   ir.OpType
	CrossLinkKind device.CrossLinkKind
	MetalKernels  KernelRegistrationStatus
	KernelNames   []string
}

/*
DeviceBackendMatrix is the Metal / CUDA / XLA audit snapshot.
*/
type DeviceBackendMatrix struct {
	Backends    []DeviceBackendRow
	Kernels     []KernelCoverageRow
	RequiredOps []RequiredOpCoverageRow
}

/*
BuildDeviceBackendMatrix audits pkg/backend/device/{metal,cuda,xla}.
Importing this package loads Metal init registrations via load_metal.go.
*/
func BuildDeviceBackendMatrix() (*DeviceBackendMatrix, error) {
	root, err := locateDeviceBackendRoot()
	if err != nil {
		return nil, err
	}

	byLocation := indexKernelsByLocation(kernels.Default.Snapshot())

	matrix := &DeviceBackendMatrix{
		Backends: []DeviceBackendRow{
			buildBackendRow(root, DeviceBackendMetal, byLocation[tensor.Metal]),
			buildBackendRow(root, DeviceBackendCUDA, byLocation[tensor.CUDA]),
			buildBackendRow(root, DeviceBackendXLA, byLocation[tensor.XLA]),
		},
		Kernels:     buildKernelCoverageRows(byLocation),
		RequiredOps: buildRequiredOpCoverage(byLocation[tensor.Metal]),
	}

	return matrix, nil
}

/*
ValidateDeviceBackendMatrix checks structural invariants on the audit matrix.
*/
func ValidateDeviceBackendMatrix(matrix *DeviceBackendMatrix) error {
	if matrix == nil {
		return fmt.Errorf("backendaudit: nil matrix")
	}

	if len(matrix.Backends) != 3 {
		return fmt.Errorf("backendaudit: want 3 backends, got %d", len(matrix.Backends))
	}

	metalRow := backendRowByName(matrix, DeviceBackendMetal)
	cudaRow := backendRowByName(matrix, DeviceBackendCUDA)
	xlaRow := backendRowByName(matrix, DeviceBackendXLA)

	if metalRow.RegisteredKernels == 0 {
		return fmt.Errorf("backendaudit: metal must have registered kernels")
	}

	if cudaRow.RegisteredKernels != 0 {
		return fmt.Errorf("backendaudit: cuda kernel count must be 0, got %d", cudaRow.RegisteredKernels)
	}

	if xlaRow.RegisteredKernels != 0 {
		return fmt.Errorf("backendaudit: xla kernel count must be 0, got %d", xlaRow.RegisteredKernels)
	}

	return nil
}

/*
RenderMarkdown emits docs/device-backend-matrix.md content.
*/
func RenderMarkdown(matrix *DeviceBackendMatrix) string {
	if matrix == nil {
		return ""
	}

	var builder strings.Builder

	builder.WriteString("# Device backend matrix (T1.4)\n\n")
	builder.WriteString("Per-backend registration for **Metal**, **CUDA**, and **XLA** ")
	builder.WriteString("tensor backends under `pkg/backend/device/`. ")
	builder.WriteString("**registered** means at least one `kernels.Default` entry with that ")
	builder.WriteString("`tensor.Location`; it does not assert full `device.Backend` coverage.\n\n")
	builder.WriteString("Machine-checkable source: `pkg/backend/device/backendaudit/`, ")
	builder.WriteString("validated by `backendaudit_test.go`. Metal registrations load via ")
	builder.WriteString("`load_metal.go` blank import.\n\n")
	builder.WriteString("CPU dispatch (T1.3): [`docs/cpu-dispatch-matrix.md`](./cpu-dispatch-matrix.md). ")
	builder.WriteString("Combined coverage (T1.5): [`backend-coverage.md`](./backend-coverage.md).\n\n")

	builder.WriteString("## Backend summary\n\n")
	builder.WriteString("| Backend | Supported dtypes | Kernel registrations | Unique kernel names | ")
	builder.WriteString("`.metal` / dispatch sources | `*_darwin.go` | `*_stub.go` | Tensor API methods |\n")
	builder.WriteString("|---------|------------------|------------------------:|----------------------:|")
	builder.WriteString("---------------------------:|--------------:|-------------:|-------------------:|\n")

	for _, row := range matrix.Backends {
		builder.WriteString(fmt.Sprintf(
			"| %s | %s | %d | %d | %d | %d | %d | %d |\n",
			row.Backend,
			formatDTypeList(row.SupportedDTypes),
			row.RegisteredKernels,
			row.UniqueKernelNames,
			row.KernelSources,
			row.DarwinDispatchFiles,
			row.StubDispatchFiles,
			row.TensorAPIMethods,
		))
	}

	builder.WriteString("\n")

	builder.WriteString("## Required IR operations — Metal kernel coverage\n\n")
	builder.WriteString("Maps each `ir.RequiredOperationIDs()` entry to expected kernel name(s) ")
	builder.WriteString("and whether any Metal registration exists.\n\n")
	builder.WriteString("| Operation ID | Cross-link | Metal kernels |\n")
	builder.WriteString("|--------------|------------|:-------------:|\n")

	metalYes := 0

	for _, row := range matrix.RequiredOps {
		mark := "—"

		if row.MetalKernels == KernelRegistered {
			mark = "yes"
			metalYes++
		}

		builder.WriteString(fmt.Sprintf(
			"| `%s` | %s | %s |\n",
			row.OperationID,
			row.CrossLinkKind,
			mark,
		))
	}

	builder.WriteString(fmt.Sprintf("\nMetal covers **%d / %d** required operation IDs via `kernels.Default`.\n\n", metalYes, len(matrix.RequiredOps)))

	builder.WriteString("## Kernel name index (Metal / CUDA / XLA)\n\n")
	builder.WriteString("| Kernel name | Metal | CUDA | XLA | Dtype variants (Metal) |\n")
	builder.WriteString("|-------------|:-----:|:----:|:---:|-------------------------:|\n")

	for _, row := range matrix.Kernels {
		builder.WriteString(fmt.Sprintf(
			"| %s | %s | %s | %s | %d |\n",
			row.KernelName,
			markStatus(row.Metal),
			markStatus(row.CUDA),
			markStatus(row.XLA),
			row.DTypeCount,
		))
	}

	builder.WriteString("\n")

	return builder.String()
}

func markStatus(status KernelRegistrationStatus) string {
	if status == KernelRegistered {
		return "yes"
	}

	return "—"
}

func formatDTypeList(dtypes []dtype.DType) string {
	if len(dtypes) == 0 {
		return "—"
	}

	parts := make([]string, len(dtypes))

	for index, storageDType := range dtypes {
		parts[index] = storageDType.String()
	}

	return strings.Join(parts, ", ")
}

func buildBackendRow(
	root string,
	backendName DeviceBackendName,
	registrations []kernels.Kernel,
) DeviceBackendRow {
	uniqueNames := uniqueKernelNames(registrations)

	return DeviceBackendRow{
		Backend:             backendName,
		SupportedDTypes:     supportedDTypesForBackend(root, backendName),
		KernelSources:       countKernelSources(root, backendName),
		DarwinDispatchFiles: countDispatchFiles(root, backendName, "darwin"),
		StubDispatchFiles:   countDispatchFiles(root, backendName, "stub"),
		RegisteredKernels:   len(registrations),
		UniqueKernelNames:   len(uniqueNames),
		TensorAPIMethods:    countTensorAPIMethods(root, backendName),
	}
}

func indexKernelsByLocation(entries []kernels.Kernel) map[tensor.Location][]kernels.Kernel {
	index := map[tensor.Location][]kernels.Kernel{
		tensor.Metal: {},
		tensor.CUDA:  {},
		tensor.XLA:   {},
	}

	for _, entry := range entries {
		for _, location := range entry.Locations {
			index[location] = append(index[location], entry)
		}
	}

	return index
}

func buildKernelCoverageRows(byLocation map[tensor.Location][]kernels.Kernel) []KernelCoverageRow {
	nameSet := make(map[string]bool)

	for _, location := range []tensor.Location{tensor.Metal, tensor.CUDA, tensor.XLA} {
		for _, entry := range byLocation[location] {
			nameSet[entry.Name] = true
		}
	}

	names := make([]string, 0, len(nameSet))

	for name := range nameSet {
		names = append(names, name)
	}

	sort.Strings(names)

	rows := make([]KernelCoverageRow, 0, len(names))

	for _, name := range names {
		metalEntries := filterByName(byLocation[tensor.Metal], name)

		row := KernelCoverageRow{
			KernelName: name,
			Metal:      registrationStatus(len(metalEntries) > 0),
			CUDA:       registrationStatus(len(filterByName(byLocation[tensor.CUDA], name)) > 0),
			XLA:        registrationStatus(len(filterByName(byLocation[tensor.XLA], name)) > 0),
			DTypeCount: len(metalEntries),
		}

		rows = append(rows, row)
	}

	return rows
}

func buildRequiredOpCoverage(metalKernels []kernels.Kernel) []RequiredOpCoverageRow {
	index := device.BuildOperationCrossLinkIndex()
	requiredIDs := ir.RequiredOperationIDs()

	rows := make([]RequiredOpCoverageRow, 0, len(requiredIDs))

	for _, operationID := range requiredIDs {
		link := index[operationID]
		candidateNames := kernelNamesForOperation(operationID, link)
		registered := KernelNotRegistered

		for _, candidateName := range candidateNames {
			if len(filterByName(metalKernels, candidateName)) > 0 {
				registered = KernelRegistered
				break
			}
		}

		rows = append(rows, RequiredOpCoverageRow{
			OperationID:   operationID,
			CrossLinkKind: link.Kind,
			MetalKernels:  registered,
			KernelNames:   candidateNames,
		})
	}

	return rows
}

func registrationStatus(registered bool) KernelRegistrationStatus {
	if registered {
		return KernelRegistered
	}

	return KernelNotRegistered
}

func uniqueKernelNames(entries []kernels.Kernel) map[string]bool {
	names := make(map[string]bool, len(entries))

	for _, entry := range entries {
		names[entry.Name] = true
	}

	return names
}

func backendRowByName(matrix *DeviceBackendMatrix, name DeviceBackendName) DeviceBackendRow {
	for _, row := range matrix.Backends {
		if row.Backend == name {
			return row
		}
	}

	return DeviceBackendRow{}
}

func filterByName(entries []kernels.Kernel, name string) []kernels.Kernel {
	filtered := make([]kernels.Kernel, 0, 4)

	for _, entry := range entries {
		if entry.Name == name {
			filtered = append(filtered, entry)
		}
	}

	return filtered
}
