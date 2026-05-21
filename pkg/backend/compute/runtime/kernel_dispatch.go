package runtime

import (
	"strings"

	"github.com/theapemachine/manifesto/ir"
)

/*
KernelName maps manifest operation IDs to puter kernel registry names.
*/
func KernelName(operationID ir.OpID) string {
	if mapped, ok := kernelAliases[string(operationID)]; ok {
		return mapped
	}

	text := string(operationID)

	if index := strings.LastIndex(text, "."); index >= 0 {
		return text[index+1:]
	}

	return text
}

var kernelAliases = map[string]string{
	"projection.linear":            "linear",
	"projection.fused_qkv":         "fused_qkv",
	"math.rmsnorm":                 "rmsnorm",
	"math.layernorm":               "layernorm",
	"math.groupnorm":               "groupnorm",
	"math.add":                     "add",
	"math.mul":                     "mul",
	"math.matmul":                  "matmul",
	"math.softmax":                 "softmax",
	"embedding.token":              "embedding_lookup",
	"attention.gqa":                "grouped_query_attention",
	"attention.sdpa":               "attention",
	"activation.swiglu":            "swiglu",
	"activation.swish":             "swish",
	"activation.gelu":              "gelu",
	"activation.relu":              "relu",
	"shape.concat":                 "concat",
	"shape.slice":                  "slice",
	"shape.view_as_heads":          "view_as_heads",
	"shape.merge_heads":            "merge_heads",
	"positional.rope":              "rope",
	"pooling.max_pool2d":           "max_pool2d",
	"pooling.avg_pool2d":           "avg_pool2d",
	"convolution.conv2d":           "conv2d",
	"convolution.conv_transpose2d": "conv_transpose2d",
	"sampling.topk_sample":         "topk_sample",
}
