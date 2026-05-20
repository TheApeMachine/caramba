package backendaudit

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/device"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
locateDeviceBackendRoot returns pkg/backend/device.
*/
func locateDeviceBackendRoot() (string, error) {
	_, filePath, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("backendaudit: runtime.Caller failed")
	}

	root := filepath.Clean(filepath.Join(filepath.Dir(filePath), ".."))

	info, err := os.Stat(root)
	if err != nil {
		return "", err
	}

	if !info.IsDir() {
		return "", fmt.Errorf("backendaudit: device root is not a directory: %s", root)
	}

	return root, nil
}

func supportedDTypesForBackend(root string, backendName DeviceBackendName) []dtype.DType {
	backendPath := filepath.Join(root, string(backendName))

	dtypes := extractSupportedDTypesFromFile(filepath.Join(backendPath, "backend.go"))
	if len(dtypes) > 0 {
		return dtypes
	}

	if backendName == DeviceBackendCUDA {
		return extractSupportedDTypesFromFile(filepath.Join(backendPath, "bridge_real.go"))
	}

	return documentedDTypesForBackend(backendName)
}

func extractSupportedDTypesFromFile(filePath string) []dtype.DType {
	fileSet := token.NewFileSet()

	parsed, err := parser.ParseFile(fileSet, filePath, nil, parser.ParseComments)
	if err != nil {
		return nil
	}

	for _, declaration := range parsed.Decls {
		functionDeclaration, isFunction := declaration.(*ast.FuncDecl)
		if !isFunction {
			continue
		}

		if functionDeclaration.Name.Name != "SupportedDTypes" {
			continue
		}

		dtypes := extractDTypesFromFunction(functionDeclaration)
		if len(dtypes) > 0 {
			return dtypes
		}
	}

	return extractDTypesFromFileBody(parsed)
}

func extractDTypesFromFileBody(parsed *ast.File) []dtype.DType {
	dtypes := make([]dtype.DType, 0, 16)

	ast.Inspect(parsed, func(node ast.Node) bool {
		selector, isSelector := node.(*ast.SelectorExpr)
		if !isSelector {
			return true
		}

		identifier, isIdentifier := selector.X.(*ast.Ident)
		if !isIdentifier || identifier.Name != "dtype" {
			return true
		}

		parsedDType, ok := dtypeBySelector(selector.Sel.Name)
		if !ok {
			return true
		}

		dtypes = appendUniqueDType(dtypes, parsedDType)

		return true
	})

	return dtypes
}

func documentedDTypesForBackend(backendName DeviceBackendName) []dtype.DType {
	switch backendName {
	case DeviceBackendCUDA:
		return []dtype.DType{
			dtype.Float32,
			dtype.BFloat16,
			dtype.Float16,
			dtype.Int8,
			dtype.Int4,
			dtype.Bool,
		}
	default:
		return nil
	}
}

func appendUniqueDType(slice []dtype.DType, value dtype.DType) []dtype.DType {
	for _, existing := range slice {
		if existing == value {
			return slice
		}
	}

	return append(slice, value)
}

func extractDTypesFromFunction(functionDeclaration *ast.FuncDecl) []dtype.DType {
	if functionDeclaration.Body == nil {
		return nil
	}

	dtypes := make([]dtype.DType, 0, 16)

	ast.Inspect(functionDeclaration.Body, func(node ast.Node) bool {
		selector, isSelector := node.(*ast.SelectorExpr)
		if !isSelector {
			return true
		}

		identifier, isIdentifier := selector.X.(*ast.Ident)
		if !isIdentifier || identifier.Name != "dtype" {
			return true
		}

		parsedDType, ok := dtypeBySelector(selector.Sel.Name)
		if !ok {
			return true
		}

		dtypes = append(dtypes, parsedDType)

		return true
	})

	return dtypes
}

func dtypeBySelector(name string) (dtype.DType, bool) {
	switch name {
	case "Float64":
		return dtype.Float64, true
	case "Float32":
		return dtype.Float32, true
	case "Float16":
		return dtype.Float16, true
	case "BFloat16":
		return dtype.BFloat16, true
	case "Float8E4M3":
		return dtype.Float8E4M3, true
	case "Float8E5M2":
		return dtype.Float8E5M2, true
	case "Int64":
		return dtype.Int64, true
	case "Int32":
		return dtype.Int32, true
	case "Int16":
		return dtype.Int16, true
	case "Int8":
		return dtype.Int8, true
	case "Int4":
		return dtype.Int4, true
	case "Uint64":
		return dtype.Uint64, true
	case "Uint32":
		return dtype.Uint32, true
	case "Uint16":
		return dtype.Uint16, true
	case "Uint8":
		return dtype.Uint8, true
	case "Bool":
		return dtype.Bool, true
	default:
		return dtype.Invalid, false
	}
}

func countKernelSources(root string, backendName DeviceBackendName) int {
	if backendName != DeviceBackendMetal {
		return 0
	}

	backendPath := filepath.Join(root, string(backendName))

	entries, err := os.ReadDir(backendPath)
	if err != nil {
		return 0
	}

	count := 0

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		if strings.HasSuffix(entry.Name(), ".metal") {
			count++
		}
	}

	return count
}

func countDispatchFiles(root string, backendName DeviceBackendName, kind string) int {
	backendPath := filepath.Join(root, string(backendName))

	entries, err := os.ReadDir(backendPath)
	if err != nil {
		return 0
	}

	count := 0

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		fileName := entry.Name()

		if !strings.HasSuffix(fileName, ".go") {
			continue
		}

		if strings.Contains(fileName, "_"+kind+".go") {
			count++
		}
	}

	return count
}

func countTensorAPIMethods(root string, backendName DeviceBackendName) int {
	backendPath := filepath.Join(root, string(backendName))

	entries, err := os.ReadDir(backendPath)
	if err != nil {
		return 0
	}

	count := 0

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		fileName := entry.Name()

		if strings.HasSuffix(fileName, "_test.go") {
			continue
		}

		if !strings.HasSuffix(fileName, ".go") {
			continue
		}

		body, readErr := os.ReadFile(filepath.Join(backendPath, fileName))
		if readErr != nil {
			continue
		}

		count += strings.Count(string(body), "func (backend *Backend)")
	}

	return count
}

func kernelNamesForOperation(
	operationID ir.OpType,
	link device.OperationCrossLink,
) []string {
	if names, ok := knownKernelNames[operationID]; ok {
		return names
	}

	if link.Kind == device.CrossLinkGraphOnly {
		return []string{}
	}

	if link.Kind == device.CrossLinkDirect || link.Kind == device.CrossLinkComposite {
		return kernelNamesFromMethodRefs(link.Methods)
	}

	operationText := string(operationID)
	parts := strings.Split(operationText, ".")

	if len(parts) == 0 {
		return []string{operationText}
	}

	suffix := parts[len(parts)-1]

	if strings.HasPrefix(operationText, "train.optimizer.") {
		return []string{suffix + "_step"}
	}

	if strings.HasPrefix(operationText, "train.loss.") {
		return []string{suffix}
	}

	if strings.HasPrefix(operationText, "train.grad.") {
		return []string{suffix}
	}

	if strings.HasPrefix(operationText, "activation.") {
		return []string{mapActivationKernelName(suffix)}
	}

	return []string{suffix}
}

func kernelNamesFromMethodRefs(methods []device.MethodRef) []string {
	names := make([]string, 0, len(methods))

	for _, methodRef := range methods {
		name := mapBackendMethodToKernelName(methodRef.Method)
		if name == "" {
			continue
		}

		names = append(names, name)
	}

	return names
}

func mapBackendMethodToKernelName(methodName string) string {
	switch methodName {
	case "Add":
		return "add"
	case "Mul":
		return "mul"
	case "Matmul":
		return "matmul"
	case "ReLU":
		return "relu"
	case "LeakyReLU":
		return "leaky_relu"
	case "Gelu":
		return "gelu"
	case "Tanh":
		return "tanh"
	case "Sigmoid":
		return "sigmoid"
	case "SwiGLU":
		return "swiglu"
	case "Swish":
		return "swish"
	case "SELU":
		return "selu"
	case "ScaledDotProductAttention":
		return "attention"
	case "MultiHeadAttention":
		return "multi_head_attention"
	case "FlashAttention":
		return "flash_attention"
	case "ApplyMask":
		return "apply_mask"
	case "CausalMask":
		return "causal_mask"
	case "ALiBiBias":
		return "alibi_bias"
	case "Lookup":
		return "embedding_lookup"
	case "Conv1D":
		return "conv1d"
	case "Conv2D":
		return "conv2d"
	case "Conv3D":
		return "conv3d"
	case "ConvTranspose2D":
		return "conv_transpose2d"
	case "MaxPool2D":
		return "max_pool2d"
	case "LayerNorm":
		return "layernorm"
	case "RMSNorm":
		return "rmsnorm"
	case "GroupNorm":
		return "groupnorm"
	case "RoPE":
		return "rope"
	case "Dropout":
		return "dropout"
	case "MSE":
		return "mse_loss"
	case "CrossEntropy":
		return "cross_entropy"
	case "Exp":
		return "exp"
	case "Log":
		return "log"
	case "Softmax":
		return "softmax"
	default:
		return strings.ToLower(methodName)
	}
}

func mapActivationKernelName(suffix string) string {
	switch suffix {
	case "gelu":
		return "gelu"
	case "leaky_relu":
		return "leaky_relu"
	case "swiglu":
		return "swiglu"
	default:
		return suffix
	}
}

var knownKernelNames = map[ir.OpType][]string{
	ir.OpAdd:                   {"add"},
	ir.OpMul:                   {"mul"},
	ir.OpMatmul:                {"matmul"},
	ir.OpReLU:                  {"relu"},
	ir.OpLeakyReLU:             {"leaky_relu"},
	ir.OpGELU:                  {"gelu"},
	ir.OpTanh:                  {"tanh"},
	ir.OpSigmoid:               {"sigmoid"},
	ir.OpSwiGLU:                {"swiglu"},
	ir.OpSwish:                 {"swish"},
	ir.OpSELU:                  {"selu"},
	"math.add":                 {"add"},
	"math.mul":                 {"mul"},
	"math.matmul":              {"matmul"},
	"math.softmax":             {"softmax"},
	"math.dropout":             {"dropout"},
	"math.rmsnorm":             {"rmsnorm"},
	"math.layernorm":           {"layernorm"},
	"math.groupnorm":           {"groupnorm"},
	"positional.rope":          {"rope"},
	"positional.alibi":         {"alibi_bias"},
	"embedding.token":          {"embedding_lookup"},
	"attention.sdpa":           {"attention"},
	"attention.mqa":            {"multi_head_attention", "flash_attention"},
	"attention.gqa":            {"grouped_query_attention", "flash_attention"},
	"attention.sliding_window": {"sliding_window_attention"},
	"masking.apply":            {"apply_mask"},
	"masking.causal":           {"causal_mask"},
	"train.optimizer.adam":     {"adam_step"},
	"train.optimizer.adamw":    {"adamw_step"},
	"train.optimizer.adamax":   {"adamax_step"},
	"train.optimizer.sgd":      {"sgd_step"},
	"train.optimizer.lion":     {"lion_step"},
	"train.optimizer.rmsprop":  {"rmsprop_step"},
	"train.optimizer.hebbian":  {"hebbian_step"},
	"train.optimizer.lars":     {"lars_step"},
	"train.optimizer.lamb":     {"lamb_step"},
	"train.optimizer.adagrad":  {"adagrad_step"},
	"train.optimizer.adadelta": {"adadelta_step"},
	"train.optimizer.lbfgs":    {"lbfgs_step"},
	"train.loss.mse":           {"mse_loss"},
	"train.loss.cross_entropy": {"cross_entropy"},
	"convolution.conv1d":       {"conv1d"},
	"convolution.conv2d":       {"conv2d"},
	"convolution.conv3d":            {"conv3d"},
	"convolution.conv_transpose2d":  {"conv_transpose2d"},
	"pooling.max_pool2d":            {"max_pool2d"},
	"math.sin":                      {"sin"},
	"math.cos":                 {"cos"},
	"math.logsumexp":           {"logsumexp"},
	"math.outer":               {"outer"},
	"math.sign":                {"sign"},
	"shape.reshape":            {"reshape"},
	"shape.transpose":          {"transpose"},
	"shape.concat":             {"concat"},
	"shape.split":              {"split2"},
	"shape.view_as_heads":      {"view_as_heads"},
	"shape.merge_heads":        {"merge_heads"},
	"shape.slice":              {"slice"},
}
