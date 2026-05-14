package model_test

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/model"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type stateOperation interface {
	Forward(*state.Dict) (*state.Dict, error)
}

func forwardModel(
	operation stateOperation, stateDict *state.Dict, inputs ...[]float64,
) []float64 {
	values := make([]any, len(inputs))

	for index := range inputs {
		values[index] = inputs[index]
	}

	if len(values) > 0 {
		stateDict.WithInputs(values...)
	}

	outputState, err := operation.Forward(stateDict)

	So(err, ShouldBeNil)

	return outputState.Out
}

// seedRegistry populates the global registry with a synthetic weight map
// so tests do not need to touch disk or the network.
func seedRegistry(source string, weights model.WeightMap) {
	loader := model.NewLoader(source, "", "")
	// Bypass the file load by storing directly through a round-trip call
	// on a loader whose weights field we can set via the public registry.
	model.GlobalRegistry().StoreForTest(source, weights)
	_ = loader
}

func TestWeightMap_Select(t *testing.T) {
	Convey("Given a WeightMap with layered keys", t, func() {
		weights := model.WeightMap{
			"transformer.h.0.attn.q": []float64{1, 2},
			"transformer.h.0.attn.v": []float64{3, 4},
			"transformer.h.0.mlp.fc": []float64{5, 6},
			"transformer.h.1.attn.q": []float64{7, 8},
		}

		Convey("Select with exact path returns one entry", func() {
			result := weights.Select("transformer.h.0.attn.q")
			So(result, ShouldHaveLength, 1)
			So(result["transformer.h.0.attn.q"], ShouldResemble, []float64{1, 2})
		})

		Convey("Select with wildcard segment matches all attn.q keys", func() {
			result := weights.Select("transformer.h.*.attn.q")
			So(result, ShouldHaveLength, 2)
		})

		Convey("Select with ** wildcard matches across segments", func() {
			result := weights.Select("**.attn.**")
			So(result, ShouldHaveLength, 3)
		})

		Convey("Select with non-matching pattern returns empty map", func() {
			result := weights.Select("nonexistent.key")
			So(result, ShouldHaveLength, 0)
		})
	})
}

func TestSurgery_Remove(t *testing.T) {
	Convey("Given a Surgery node configured to remove a layer", t, func() {
		source := "test/surgery-remove"
		weights := model.WeightMap{
			"transformer.h.0.attn.q": []float64{1},
			"transformer.h.0.attn.v": []float64{2},
			"transformer.h.1.attn.q": []float64{3},
		}
		model.GlobalRegistry().StoreForTest(source, weights)

		op := model.NewSurgery(source, "remove", "transformer.h.0", "", "", nil)

		Convey("It should remove all keys under the target prefix", func() {
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.Op = "remove"
			stateDict.At = "transformer.h.0"
			result := forwardModel(op, stateDict, []float64{1})
			So(result[0], ShouldBeGreaterThanOrEqualTo, 0)

			remaining, _ := model.GlobalRegistry().Get(source)
			So(remaining, ShouldNotContainKey, "transformer.h.0.attn.q")
			So(remaining, ShouldNotContainKey, "transformer.h.0.attn.v")
			So(remaining, ShouldContainKey, "transformer.h.1.attn.q")
		})
	})
}

func TestSurgery_Replace(t *testing.T) {
	Convey("Given a Surgery node configured to replace a layer", t, func() {
		source := "test/surgery-replace"
		weights := model.WeightMap{
			"transformer.h.0.attn.q": []float64{1, 2, 3},
		}
		model.GlobalRegistry().StoreForTest(source, weights)

		newWeights := []float64{9, 9, 9}
		op := model.NewSurgery(source, "replace", "transformer.h.0.attn.q", "", "transformer.h.0.attn.q_new", newWeights)

		Convey("It should replace the target key with the new weights", func() {
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.Op = "replace"
			stateDict.At = "transformer.h.0.attn.q"
			stateDict.Name = "transformer.h.0.attn.q_new"
			stateDict.Layer = newWeights
			forwardModel(op, stateDict, []float64{1})

			updated, _ := model.GlobalRegistry().Get(source)
			So(updated["transformer.h.0.attn.q_new"], ShouldResemble, []float64{9, 9, 9})
		})
	})
}

func TestGraft_ReadMode(t *testing.T) {
	Convey("Given a Graft node in read mode", t, func() {
		source := "test/graft-read"
		weights := model.WeightMap{
			"transformer.h.6.attn.v": []float64{1, 2, 3, 4},
		}
		model.GlobalRegistry().StoreForTest(source, weights)

		op := model.NewGraft(source, "transformer.h.6.attn.v", "read")

		Convey("It should emit the layer weights without modification", func() {
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.At = "transformer.h.6.attn.v"
			stateDict.Mode = "read"
			result := forwardModel(op, stateDict, []float64{1})
			So(result, ShouldResemble, []float64{1, 2, 3, 4})

			unchanged, _ := model.GlobalRegistry().Get(source)
			So(unchanged["transformer.h.6.attn.v"], ShouldResemble, []float64{1, 2, 3, 4})
		})
	})
}

func TestGraft_ReadWrite(t *testing.T) {
	Convey("Given a Graft node in read_write mode", t, func() {
		source := "test/graft-rw"
		weights := model.WeightMap{
			"transformer.h.12.attn.v": []float64{1, 1, 1},
		}
		model.GlobalRegistry().StoreForTest(source, weights)

		op := model.NewGraft(source, "transformer.h.12.attn.v", "read_write")

		Convey("It should add the injection vector back into the layer", func() {
			injection := []float64{0.5, 0.5, 0.5}
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.At = "transformer.h.12.attn.v"
			stateDict.Mode = "read_write"
			result := forwardModel(op, stateDict, []float64{1}, injection)
			So(result, ShouldResemble, []float64{1.5, 1.5, 1.5})
		})
	})
}

func TestLoRA_Forward(t *testing.T) {
	Convey("Given a LoRA node with qv preset", t, func() {
		source := "test/lora"
		weights := model.WeightMap{
			"transformer.h.0.attn.q": []float64{1, 2, 3, 4},
			"transformer.h.0.attn.v": []float64{5, 6, 7, 8},
			"transformer.h.0.mlp.fc": []float64{9, 10},
		}
		model.GlobalRegistry().StoreForTest(source, weights)

		op := model.NewLoRA(source, "qv", nil, 2, 4)

		Convey("It should adapt Q and V weights but not MLP", func() {
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.Preset = "qv"
			stateDict.Rank = 2
			stateDict.Alpha = 4
			result := forwardModel(op, stateDict, []float64{1})
			So(result[0], ShouldEqual, 2)

			adapted, _ := model.GlobalRegistry().Get(source)
			So(adapted["transformer.h.0.mlp.fc"], ShouldResemble, []float64{9, 10})
		})
	})
}

func TestFreeze_Forward(t *testing.T) {
	Convey("Given a Freeze node targeting early layers", t, func() {
		source := "test/freeze"
		weights := model.WeightMap{
			"transformer.h.0.attn.q": []float64{1},
			"transformer.h.1.attn.q": []float64{2},
		}
		model.GlobalRegistry().StoreForTest(source, weights)

		op := model.NewFreeze(source, "transformer.h.*", "", true)

		Convey("It should mark matching keys as frozen", func() {
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.Pattern = "transformer.h.*"
			stateDict.Frozen = true
			forwardModel(op, stateDict, []float64{1})

			updated, _ := model.GlobalRegistry().Get(source)
			So(model.IsFrozen(updated, "transformer.h.0.attn.q"), ShouldBeTrue)
			So(model.IsFrozen(updated, "transformer.h.1.attn.q"), ShouldBeTrue)
		})
	})
}

func TestFreeze_Except(t *testing.T) {
	Convey("Given a Freeze node with an except pattern", t, func() {
		source := "test/freeze-except"
		weights := model.WeightMap{
			"transformer.h.0.attn.q":  []float64{1},
			"transformer.h.10.attn.q": []float64{2},
		}
		model.GlobalRegistry().StoreForTest(source, weights)

		op := model.NewFreeze(source, "transformer.h.*", "transformer.h.10.*", true)

		Convey("It should freeze the matched layer but not the excepted one", func() {
			stateDict := state.NewDict()
			stateDict.Source = source
			stateDict.Pattern = "transformer.h.*"
			stateDict.Except = "transformer.h.10.*"
			stateDict.Frozen = true
			forwardModel(op, stateDict, []float64{1})

			updated, _ := model.GlobalRegistry().Get(source)
			So(model.IsFrozen(updated, "transformer.h.0.attn.q"), ShouldBeTrue)
			So(model.IsFrozen(updated, "transformer.h.10.attn.q"), ShouldBeFalse)
		})
	})
}

func BenchmarkWeightMap_Select(b *testing.B) {
	weights := make(model.WeightMap, 1000)

	for idx := range 100 {
		for _, part := range []string{"attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down", "mlp.down2", "norm.w", "norm.b"} {
			key := "transformer.h." + itoa(idx) + "." + part
			weights[key] = []float64{1, 2, 3, 4}
		}
	}

	b.ResetTimer()

	for range b.N {
		_ = weights.Select("transformer.h.*.attn.*")
	}
}

func itoa(n int) string {
	return string(rune('0' + n%10))
}
