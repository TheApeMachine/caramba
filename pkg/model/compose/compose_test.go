package compose

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
)

// fakeCatalog is the minimum implementation of TensorCatalog the
// pattern tests need. We do not touch disk or open real safetensors
// files — a literal map of names to shapes is enough to exercise the
// compiler end-to-end.
type fakeCatalog map[string]modelweights.TensorInfo

func (catalog fakeCatalog) Names() []string {
	names := make([]string, 0, len(catalog))
	for name := range catalog {
		names = append(names, name)
	}
	return names
}

func (catalog fakeCatalog) Info(name string) (modelweights.TensorInfo, bool) {
	info, ok := catalog[name]
	return info, ok
}

func TestFromSafetensors(t *testing.T) {
	Convey("Given a tensor catalog containing one of every leaf pattern", t, func() {
		catalog := fakeCatalog{
			// Token embedding: rank-2, name suffix in embeddingSuffixes.
			"transformer.wte.weight": {Name: "transformer.wte.weight", DType: "F32", Shape: []int{50257, 768}},

			// LayerNorm: rank-1 weight + rank-1 bias.
			"transformer.ln_f.weight": {Name: "transformer.ln_f.weight", DType: "F32", Shape: []int{768}},
			"transformer.ln_f.bias":   {Name: "transformer.ln_f.bias", DType: "F32", Shape: []int{768}},

			// RMSNorm: rank-1 weight alone.
			"model.norm.weight": {Name: "model.norm.weight", DType: "F32", Shape: []int{2560}},

			// Linear: rank-2 weight + optional bias.
			"lm_head.weight":         {Name: "lm_head.weight", DType: "F32", Shape: []int{50257, 768}},
			"transformer.proj.weight": {Name: "transformer.proj.weight", DType: "F32", Shape: []int{768, 768}},
			"transformer.proj.bias":   {Name: "transformer.proj.bias", DType: "F32", Shape: []int{768}},
		}

		Convey("FromSafetensors should emit one node per recognized leaf", func() {
			graph, err := FromSafetensors(catalog, Hints{
				Inputs: []InputSpec{{Name: "input_ids", Kind: "tokens"}},
				// Output omitted — declared output validation only
				// fires when set, so this exercise focuses on node
				// emission alone.
			})

			So(err, ShouldBeNil)
			So(graph, ShouldNotBeNil)

			nodesByID := map[string]string{}
			for _, node := range graph.Nodes() {
				nodesByID[node.ID] = node.OpID
			}

			// Embedding pattern (priority 30) wins over the linear
			// pattern (priority 10) for transformer.wte because the
			// prefix ends in one of the recognized embedding suffixes.
			So(nodesByID["transformer.wte"], ShouldEqual, "embedding.token")

			// LayerNorm wins over RMSNorm when both weight and bias
			// are rank-1 vectors of equal length.
			So(nodesByID["transformer.ln_f"], ShouldEqual, "math.layernorm")

			// RMSNorm claims rank-1 weight-only groups.
			So(nodesByID["model.norm"], ShouldEqual, "math.rmsnorm")

			// Linear catches the remaining rank-2 weights. lm_head
			// doesn't end in an embedding suffix so it stays linear
			// (chat tied-weights logic happens at bind time, not
			// during graph construction).
			So(nodesByID["lm_head"], ShouldEqual, "projection.linear")
			So(nodesByID["transformer.proj"], ShouldEqual, "projection.linear")
		})

		Convey("FromSafetensors should reject unclaimed prefixes loudly", func() {
			catalogWithUnknown := fakeCatalog{}
			for k, v := range catalog {
				catalogWithUnknown[k] = v
			}
			// Add a rank-3 tensor that no leaf pattern recognizes — a
			// real model with conv weights, for example. The compiler
			// surfaces this as an error rather than emitting half a
			// graph, so the gap is visible.
			catalogWithUnknown["model.conv.weight"] = modelweights.TensorInfo{
				Name: "model.conv.weight", DType: "F32", Shape: []int{64, 32, 3, 3},
			}

			_, err := FromSafetensors(catalogWithUnknown, Hints{})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "no pattern claims prefix")
		})

		Convey("FromSafetensorsWithRegistry should accept a caller-supplied pattern set", func() {
			empty := NewRegistry()
			_, err := FromSafetensorsWithRegistry(catalog, Hints{}, empty)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "no pattern claims")
		})

		Convey("Declared outputs that no pattern produces should error", func() {
			_, err := FromSafetensors(catalog, Hints{
				Inputs: []InputSpec{{Name: "input_ids"}},
				Output: "logits", // nothing produces this binding yet
			})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "is not produced")
		})
	})
}

func TestRegistryPriority(t *testing.T) {
	Convey("Given a registry with patterns of different priorities", t, func() {
		registry := NewRegistry()
		registry.Register(linearPattern{})        // priority 10
		registry.Register(layerNormPattern{})     // priority 20
		registry.Register(embeddingPattern{})     // priority 30

		Convey("Patterns() should return them in descending priority order", func() {
			patterns := registry.Patterns()
			So(len(patterns), ShouldEqual, 3)
			So(patterns[0].Name(), ShouldEqual, "embedding.token")
			So(patterns[1].Name(), ShouldEqual, "layernorm")
			So(patterns[2].Name(), ShouldEqual, "linear")
		})
	})
}
