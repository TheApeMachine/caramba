package weights

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

type Binder struct {
	store *Store
}

func NewBinder(store *Store) *Binder {
	return &Binder{store: store}
}

func BindIR(graph *ir.Graph, store *Store) error {
	return NewBinder(store).BindIR(graph)
}

func (binder *Binder) BindIR(graph *ir.Graph) error {
	if graph == nil {
		return fmt.Errorf("weights: graph is required")
	}

	if binder == nil || binder.store == nil {
		return fmt.Errorf("weights: store is required")
	}

	for _, node := range graph.Nodes() {
		if err := binder.bindNode(node); err != nil {
			return err
		}
	}

	return nil
}

func (binder *Binder) bindNode(node *ir.Node) error {
	switch string(node.OperationID()) {
	case "embedding.token":
		return binder.bindVectorOrMatrix(node, "weight", weightNames(node.ID()))
	case "math.layernorm":
		if err := binder.bindVectorOrMatrix(node, "weight", weightNames(node.ID())); err != nil {
			return err
		}

		return binder.bindVectorOrMatrix(node, "bias", biasNames(node.ID()))
	case "math.rmsnorm":
		return binder.bindVectorOrMatrix(node, "weight", weightNames(node.ID()))
	case "projection.linear":
		return binder.bindLinear(node)
	default:
		return nil
	}
}

func (binder *Binder) bindVectorOrMatrix(
	node *ir.Node, metadataKey string, names []string,
) error {
	name, ok := binder.first(names)

	if !ok {
		return fmt.Errorf("weights: no tensor found for node %q %s", node.ID(), metadataKey)
	}

	values, err := binder.store.Values(name)

	if err != nil {
		return err
	}

	node.SetMetadata(metadataKey, values)

	return nil
}

func (binder *Binder) bindLinear(node *ir.Node) error {
	inFeatures := nodeConfigInt(node, "in_features")
	outFeatures := nodeConfigInt(node, "out_features")

	if inFeatures <= 0 || outFeatures <= 0 {
		return fmt.Errorf("weights: node %q linear dimensions are required", node.ID())
	}

	if weight, bias, ok, err := binder.fusedGateUp(node, inFeatures, outFeatures); ok || err != nil {
		if err != nil {
			return err
		}

		node.SetMetadata("weight", weight)

		if len(bias) > 0 {
			node.SetMetadata("bias", bias)
		}

		return nil
	}

	if weight, bias, ok, err := binder.fusedQKV(node, inFeatures, outFeatures); ok || err != nil {
		if err != nil {
			return err
		}

		node.SetMetadata("weight", weight)

		if len(bias) > 0 {
			node.SetMetadata("bias", bias)
		}

		return nil
	}

	weightName, ok := binder.first(weightNames(node.ID()))

	if !ok {
		return fmt.Errorf("weights: no tensor found for node %q weight", node.ID())
	}

	weight, err := binder.linearWeight(weightName, inFeatures, outFeatures)

	if err != nil {
		return fmt.Errorf("weights: node %q: %w", node.ID(), err)
	}

	node.SetMetadata("weight", weight)

	if biasName, ok := binder.first(biasNames(node.ID())); ok {
		bias, err := binder.linearBias(biasName, outFeatures)

		if err != nil {
			return fmt.Errorf("weights: node %q: %w", node.ID(), err)
		}

		node.SetMetadata("bias", bias)
	}

	return nil
}

func (binder *Binder) linearWeight(name string, inFeatures, outFeatures int) ([]float64, error) {
	key := fmt.Sprintf("linear:%s:%d:%d", name, inFeatures, outFeatures)

	return binder.store.Derived(key, func() ([]float64, error) {
		info, ok := binder.store.Info(name)

		if !ok {
			return nil, fmt.Errorf("tensor %q not found", name)
		}

		values, err := binder.store.Values(name)

		if err != nil {
			return nil, err
		}

		return orientLinearWeight(name, values, info.Shape, inFeatures, outFeatures)
	})
}

func (binder *Binder) linearBias(name string, outFeatures int) ([]float64, error) {
	values, err := binder.store.Values(name)

	if err != nil {
		return nil, err
	}

	if len(values) != outFeatures {
		return nil, fmt.Errorf(
			"bias %q length %d does not match out_features %d",
			name, len(values), outFeatures,
		)
	}

	return values, nil
}

func (binder *Binder) fusedQKV(
	node *ir.Node, inFeatures, outFeatures int,
) ([]float64, []float64, bool, error) {
	base, layer, ok := splitLayerNode(node.ID())

	if !ok {
		return nil, nil, false, nil
	}

	sliceIndex := map[string]int{
		"q_proj": 0,
		"k_proj": 1,
		"v_proj": 2,
	}[base]

	if base != "q_proj" && base != "k_proj" && base != "v_proj" {
		return nil, nil, false, nil
	}

	name, ok := binder.first([]string{
		"transformer.h." + layer + ".attn.c_attn.weight",
		"h." + layer + ".attn.c_attn.weight",
	})

	if !ok {
		return nil, nil, false, nil
	}

	info, _ := binder.store.Info(name)
	values, err := binder.store.Values(name)

	if err != nil {
		return nil, nil, true, err
	}

	weight, err := binder.store.Derived(
		fmt.Sprintf("packed-linear:%s:%d:%d:%d", name, inFeatures, outFeatures, sliceIndex),
		func() ([]float64, error) {
			return slicePackedLinear(values, info.Shape, inFeatures, outFeatures, sliceIndex)
		},
	)

	if err != nil {
		return nil, nil, true, fmt.Errorf("node %q: %w", node.ID(), err)
	}

	var bias []float64

	if biasName, ok := binder.first([]string{
		"transformer.h." + layer + ".attn.c_attn.bias",
		"h." + layer + ".attn.c_attn.bias",
	}); ok {
		packedBias, err := binder.store.Values(biasName)

		if err != nil {
			return nil, nil, true, err
		}

		bias, err = binder.store.Derived(
			fmt.Sprintf("packed-bias:%s:%d:%d", biasName, outFeatures, sliceIndex),
			func() ([]float64, error) {
				return slicePackedBias(packedBias, outFeatures, sliceIndex)
			},
		)

		if err != nil {
			return nil, nil, true, err
		}
	}

	return weight, bias, true, nil
}

func (binder *Binder) fusedGateUp(
	node *ir.Node, inFeatures, outFeatures int,
) ([]float64, []float64, bool, error) {
	base, layer, ok := splitLayerNode(node.ID())

	if !ok || base != "gate_up_proj" {
		return nil, nil, false, nil
	}

	if outFeatures%2 != 0 {
		return nil, nil, true, fmt.Errorf("node %q out_features must be even", node.ID())
	}

	gateName, gateOK := binder.first([]string{
		"model.layers." + layer + ".mlp.gate_proj.weight",
	})
	upName, upOK := binder.first([]string{
		"model.layers." + layer + ".mlp.up_proj.weight",
	})

	if !gateOK || !upOK {
		return nil, nil, false, nil
	}

	half := outFeatures / 2
	gate, err := binder.linearWeight(gateName, inFeatures, half)

	if err != nil {
		return nil, nil, true, err
	}

	up, err := binder.linearWeight(upName, inFeatures, half)

	if err != nil {
		return nil, nil, true, err
	}

	weight, err := binder.store.Derived(
		fmt.Sprintf("gate-up:%s:%s:%d:%d", gateName, upName, inFeatures, half),
		func() ([]float64, error) {
			return concatenateLinearColumns(gate, up, inFeatures, half), nil
		},
	)

	if err != nil {
		return nil, nil, true, err
	}

	return weight, nil, true, nil
}

func (binder *Binder) first(names []string) (string, bool) {
	for _, name := range names {
		if binder.store.Has(name) {
			return name, true
		}
	}

	return "", false
}

func weightNames(nodeID string) []string {
	return tensorNames(prefixes(nodeID), "weight")
}

func biasNames(nodeID string) []string {
	return tensorNames(prefixes(nodeID), "bias")
}

func tensorNames(prefixes []string, suffix string) []string {
	seen := make(map[string]bool, len(prefixes))
	names := make([]string, 0, len(prefixes))

	for _, prefix := range prefixes {
		name := prefix + "." + suffix

		if seen[name] {
			continue
		}

		seen[name] = true
		names = append(names, name)
	}

	return names
}

func prefixes(nodeID string) []string {
	out := []string{nodeID}

	switch nodeID {
	case "token_embedding":
		out = append(out, "transformer.wte", "wte")
	case "position_embedding":
		out = append(out, "transformer.wpe", "wpe")
	case "embed_tokens":
		out = append(out, "model.embed_tokens")
	case "final_norm":
		out = append(out, "transformer.ln_f", "ln_f")
	case "norm":
		out = append(out, "model.norm")
	case "lm_head":
		out = append(out, "lm_head", "transformer.wte", "wte", "model.embed_tokens")
	}

	base, layer, ok := splitLayerNode(nodeID)

	if !ok {
		return out
	}

	switch base {
	case "ln_1", "ln_2":
		out = append(out, "transformer.h."+layer+"."+base, "h."+layer+"."+base)
	case "attention_projection":
		out = append(
			out,
			"transformer.h."+layer+".attn.c_proj",
			"h."+layer+".attn.c_proj",
		)
	case "mlp_fc":
		out = append(out, "transformer.h."+layer+".mlp.c_fc", "h."+layer+".mlp.c_fc")
	case "mlp_projection":
		out = append(out, "transformer.h."+layer+".mlp.c_proj", "h."+layer+".mlp.c_proj")
	case "q_proj", "k_proj", "v_proj":
		out = append(out, "model.layers."+layer+".self_attn."+base)
	case "o_proj":
		out = append(out, "model.layers."+layer+".self_attn.o_proj")
	case "input_layernorm", "post_attention_layernorm":
		out = append(out, "model.layers."+layer+"."+base)
	case "down_proj":
		out = append(out, "model.layers."+layer+".mlp.down_proj")
	}

	return out
}

func splitLayerNode(nodeID string) (string, string, bool) {
	index := strings.LastIndex(nodeID, "_")

	if index < 0 {
		return "", "", false
	}

	before := nodeID[:index]
	after := nodeID[index+1:]

	if _, err := strconv.Atoi(after); err != nil {
		return "", "", false
	}

	return before, after, true
}

func orientLinearWeight(
	name string,
	values []float64,
	shape []int,
	inFeatures,
	outFeatures int,
) ([]float64, error) {
	if len(values) != inFeatures*outFeatures {
		return nil, fmt.Errorf(
			"weight length %d does not match in_features*out_features=%d",
			len(values), inFeatures*outFeatures,
		)
	}

	if len(shape) != 2 {
		return values, nil
	}

	if torchLinearWeight(name) {
		if shape[0] != outFeatures || shape[1] != inFeatures {
			return nil, fmt.Errorf(
				"weight shape %v does not match torch linear [%d %d]",
				shape,
				outFeatures,
				inFeatures,
			)
		}

		return transpose(values, outFeatures, inFeatures), nil
	}

	switch {
	case shape[0] == inFeatures && shape[1] == outFeatures:
		return values, nil
	case shape[0] == outFeatures && shape[1] == inFeatures:
		return transpose(values, outFeatures, inFeatures), nil
	default:
		return nil, fmt.Errorf(
			"weight shape %v does not match [%d %d] or [%d %d]",
			shape, inFeatures, outFeatures, outFeatures, inFeatures,
		)
	}
}

func torchLinearWeight(name string) bool {
	if name == "lm_head.weight" || name == "model.embed_tokens.weight" {
		return true
	}

	return strings.HasPrefix(name, "model.layers.")
}

func slicePackedLinear(
	values []float64, shape []int, inFeatures, outFeatures, index int,
) ([]float64, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("packed qkv tensor must be rank 2, got %v", shape)
	}

	switch {
	case shape[0] == inFeatures && shape[1] == outFeatures*3:
		return sliceColumns(values, inFeatures, outFeatures*3, index*outFeatures, outFeatures), nil
	case shape[0] == outFeatures*3 && shape[1] == inFeatures:
		rows := sliceRows(values, outFeatures*3, inFeatures, index*outFeatures, outFeatures)

		return transpose(rows, outFeatures, inFeatures), nil
	default:
		return nil, fmt.Errorf(
			"packed qkv shape %v does not match [%d %d] or [%d %d]",
			shape, inFeatures, outFeatures*3, outFeatures*3, inFeatures,
		)
	}
}

func slicePackedBias(values []float64, outFeatures, index int) ([]float64, error) {
	if len(values) != outFeatures*3 {
		return nil, fmt.Errorf(
			"packed qkv bias length %d does not match %d",
			len(values), outFeatures*3,
		)
	}

	start := index * outFeatures

	return append([]float64(nil), values[start:start+outFeatures]...), nil
}

func sliceColumns(values []float64, rows, cols, start, width int) []float64 {
	out := make([]float64, rows*width)

	for row := 0; row < rows; row++ {
		copy(out[row*width:(row+1)*width], values[row*cols+start:row*cols+start+width])
	}

	return out
}

func sliceRows(values []float64, rows, cols, start, height int) []float64 {
	out := make([]float64, height*cols)

	for row := 0; row < height; row++ {
		copy(out[row*cols:(row+1)*cols], values[(start+row)*cols:(start+row+1)*cols])
	}

	return out
}

func transpose(values []float64, rows, cols int) []float64 {
	out := make([]float64, len(values))

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			out[col*rows+row] = values[row*cols+col]
		}
	}

	return out
}

func concatenateLinearColumns(left, right []float64, rows, cols int) []float64 {
	out := make([]float64, rows*cols*2)

	for row := 0; row < rows; row++ {
		copy(out[row*cols*2:row*cols*2+cols], left[row*cols:(row+1)*cols])
		copy(out[row*cols*2+cols:(row+1)*cols*2], right[row*cols:(row+1)*cols])
	}

	return out
}

func nodeConfigInt(node *ir.Node, key string) int {
	value := node.Metadata()[key]

	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case float32:
		return int(typed)
	default:
		return 0
	}
}
