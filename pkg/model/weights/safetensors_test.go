package weights

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type testTensor struct {
	name   string
	shape  []int
	values []float64
}

func TestOpen(test *testing.T) {
	Convey("Given a safetensors checkpoint", test, func() {
		path := filepath.Join(test.TempDir(), "model.safetensors")
		err := writeTestSafeTensors(path, []testTensor{
			{
				name:   "token_embedding.weight",
				shape:  []int{2, 2},
				values: []float64{1, 2, 3, 4},
			},
		})
		So(err, ShouldBeNil)

		Convey("It should read tensor metadata and values", func() {
			store, err := Open(path)
			So(err, ShouldBeNil)

			info, ok := store.Info("token_embedding.weight")
			So(ok, ShouldBeTrue)
			So(info.Shape, ShouldResemble, []int{2, 2})

			values, err := store.Values("token_embedding.weight")
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 3, 4})
		})
	})
}

func TestResolve(test *testing.T) {
	Convey("Given a local checkpoint with a component-scoped safetensors index", test, func() {
		root := test.TempDir()
		component := filepath.Join(root, "text_encoder")
		So(os.MkdirAll(component, 0o755), ShouldBeNil)

		shard := filepath.Join(component, "model-00001-of-00001.safetensors")
		err := writeTestSafeTensors(shard, []testTensor{
			{
				name:   "model.embed_tokens.weight",
				shape:  []int{2, 2},
				values: []float64{1, 2, 3, 4},
			},
		})
		So(err, ShouldBeNil)

		indexPath := filepath.Join(component, "model.safetensors.index.json")
		indexData, err := json.Marshal(safeTensorsIndex{
			WeightMap: map[string]string{
				"model.embed_tokens.weight": "model-00001-of-00001.safetensors",
			},
		})
		So(err, ShouldBeNil)
		So(os.WriteFile(indexPath, indexData, 0o644), ShouldBeNil)

		Convey("It should resolve the explicit file relative to the component directory", func() {
			store, err := Resolve(context.Background(), Source{
				Source: root,
				File:   "text_encoder/model.safetensors.index.json",
			})
			So(err, ShouldBeNil)

			values, err := store.Values("model.embed_tokens.weight")
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 3, 4})
		})
	})
}

func TestBindIR(test *testing.T) {
	Convey("Given a safetensors store and manifest-lowered IR", test, func() {
		path := filepath.Join(test.TempDir(), "model.safetensors")
		err := writeTestSafeTensors(path, []testTensor{
			{
				name:   "token_embedding.weight",
				shape:  []int{2, 2},
				values: []float64{1, 2, 3, 4},
			},
			{
				name:   "projection.weight",
				shape:  []int{3, 2},
				values: []float64{1, 2, 3, 4, 5, 6},
			},
			{
				name:   "projection.bias",
				shape:  []int{3},
				values: []float64{7, 8, 9},
			},
		})
		So(err, ShouldBeNil)

		store, err := Open(path)
		So(err, ShouldBeNil)

		shape, err := tensor.NewShape([]int{1, 1, 2})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		embedding := ir.NewNode("token_embedding", ir.OpType("embedding.token"), shape)
		embedding.SetOperationID("embedding.token")
		projection := ir.NewNode("projection", ir.OpType("projection.linear"), shape)
		projection.SetOperationID("projection.linear")
		projection.SetMetadata("in_features", 2)
		projection.SetMetadata("out_features", 3)
		graph.AddNode(embedding)
		graph.AddNode(projection)

		Convey("It should bind operation weights without manifest tensor names", func() {
			err := BindIR(graph, store)
			So(err, ShouldBeNil)

			So(embedding.Metadata()["weight"], ShouldResemble, []float64{1, 2, 3, 4})
			So(projection.Metadata()["weight"], ShouldResemble, []float64{1, 3, 5, 2, 4, 6})
			So(projection.Metadata()["bias"], ShouldResemble, []float64{7, 8, 9})
		})
	})

	Convey("Given compact GPT-2 safetensors names", test, func() {
		path := filepath.Join(test.TempDir(), "model.safetensors")
		err := writeTestSafeTensors(path, []testTensor{
			{
				name:   "h.0.ln_1.weight",
				shape:  []int{2},
				values: []float64{1, 2},
			},
			{
				name:   "h.0.ln_1.bias",
				shape:  []int{2},
				values: []float64{3, 4},
			},
			{
				name:   "h.0.attn.c_attn.weight",
				shape:  []int{2, 6},
				values: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			},
			{
				name:   "h.0.attn.c_attn.bias",
				shape:  []int{6},
				values: []float64{13, 14, 15, 16, 17, 18},
			},
			{
				name:   "h.0.attn.c_proj.weight",
				shape:  []int{2, 2},
				values: []float64{25, 26, 27, 28},
			},
			{
				name:   "wte.weight",
				shape:  []int{3, 2},
				values: []float64{19, 20, 21, 22, 23, 24},
			},
		})
		So(err, ShouldBeNil)

		store, err := Open(path)
		So(err, ShouldBeNil)

		shape, err := tensor.NewShape([]int{1, 1, 2})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		layerNorm := ir.NewNode("ln_1_0", ir.OpType("math.layernorm"), shape)
		layerNorm.SetOperationID("math.layernorm")
		qProjection := ir.NewNode("q_proj_0", ir.OpType("projection.linear"), shape)
		qProjection.SetOperationID("projection.linear")
		qProjection.SetMetadata("in_features", 2)
		qProjection.SetMetadata("out_features", 2)
		attentionProjection := ir.NewNode("attention_projection_0", ir.OpType("projection.linear"), shape)
		attentionProjection.SetOperationID("projection.linear")
		attentionProjection.SetMetadata("in_features", 2)
		attentionProjection.SetMetadata("out_features", 2)
		lmHead := ir.NewNode("lm_head", ir.OpType("projection.linear"), shape)
		lmHead.SetOperationID("projection.linear")
		lmHead.SetMetadata("in_features", 2)
		lmHead.SetMetadata("out_features", 3)
		graph.AddNode(layerNorm)
		graph.AddNode(qProjection)
		graph.AddNode(attentionProjection)
		graph.AddNode(lmHead)

		Convey("It should bind compact GPT-2 aliases", func() {
			err := BindIR(graph, store)
			So(err, ShouldBeNil)

			So(layerNorm.Metadata()["weight"], ShouldResemble, []float64{1, 2})
			So(layerNorm.Metadata()["bias"], ShouldResemble, []float64{3, 4})
			So(qProjection.Metadata()["weight"], ShouldResemble, []float64{1, 2, 7, 8})
			So(qProjection.Metadata()["bias"], ShouldResemble, []float64{13, 14})
			So(attentionProjection.Metadata()["weight"], ShouldResemble, []float64{25, 26, 27, 28})
			So(lmHead.Metadata()["weight"], ShouldResemble, []float64{19, 21, 23, 20, 22, 24})
		})

		Convey("It should cache derived GPT-2 weight views", func() {
			err := BindIR(graph, store)
			So(err, ShouldBeNil)
			qWeight := qProjection.Metadata()["weight"].([]float64)
			lmHeadWeight := lmHead.Metadata()["weight"].([]float64)

			err = BindIR(graph, store)
			So(err, ShouldBeNil)
			qWeightAgain := qProjection.Metadata()["weight"].([]float64)
			lmHeadWeightAgain := lmHead.Metadata()["weight"].([]float64)

			So(&qWeight[0] == &qWeightAgain[0], ShouldBeTrue)
			So(&lmHeadWeight[0] == &lmHeadWeightAgain[0], ShouldBeTrue)
		})
	})

	Convey("Given Llama PyTorch linear safetensors names", test, func() {
		path := filepath.Join(test.TempDir(), "model.safetensors")
		err := writeTestSafeTensors(path, []testTensor{
			{
				name:   "model.layers.0.self_attn.q_proj.weight",
				shape:  []int{3, 2},
				values: []float64{1, 2, 3, 4, 5, 6},
			},
			{
				name:   "model.layers.0.self_attn.o_proj.weight",
				shape:  []int{2, 2},
				values: []float64{5, 6, 7, 8},
			},
			{
				name:   "lm_head.weight",
				shape:  []int{2, 2},
				values: []float64{9, 10, 11, 12},
			},
		})
		So(err, ShouldBeNil)

		store, err := Open(path)
		So(err, ShouldBeNil)

		shape, err := tensor.NewShape([]int{1, 1, 2})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		qProjection := ir.NewNode("q_proj_0", ir.OpType("projection.linear"), shape)
		qProjection.SetOperationID("projection.linear")
		qProjection.SetMetadata("in_features", 2)
		qProjection.SetMetadata("out_features", 3)
		oProjection := ir.NewNode("o_proj_0", ir.OpType("projection.linear"), shape)
		oProjection.SetOperationID("projection.linear")
		oProjection.SetMetadata("in_features", 2)
		oProjection.SetMetadata("out_features", 2)
		lmHead := ir.NewNode("lm_head", ir.OpType("projection.linear"), shape)
		lmHead.SetOperationID("projection.linear")
		lmHead.SetMetadata("in_features", 2)
		lmHead.SetMetadata("out_features", 2)
		graph.AddNode(qProjection)
		graph.AddNode(oProjection)
		graph.AddNode(lmHead)

		Convey("It should transpose PyTorch Linear tensors by name", func() {
			err := BindIR(graph, store)
			So(err, ShouldBeNil)

			So(qProjection.Metadata()["weight"], ShouldResemble, []float64{1, 3, 5, 2, 4, 6})
			So(oProjection.Metadata()["weight"], ShouldResemble, []float64{5, 7, 6, 8})
			So(lmHead.Metadata()["weight"], ShouldResemble, []float64{9, 11, 10, 12})
		})
	})
}

func BenchmarkOpen(benchmark *testing.B) {
	path := filepath.Join(benchmark.TempDir(), "model.safetensors")
	err := writeTestSafeTensors(path, []testTensor{
		{name: "tensor.weight", shape: []int{2, 2}, values: []float64{1, 2, 3, 4}},
	})

	if err != nil {
		benchmark.Fatal(err)
	}

	for benchmark.Loop() {
		_, _ = Open(path)
	}
}

func BenchmarkBindIR(benchmark *testing.B) {
	path := filepath.Join(benchmark.TempDir(), "model.safetensors")
	err := writeTestSafeTensors(path, []testTensor{
		{name: "projection.weight", shape: []int{3, 2}, values: []float64{1, 2, 3, 4, 5, 6}},
	})

	if err != nil {
		benchmark.Fatal(err)
	}

	store, err := Open(path)

	if err != nil {
		benchmark.Fatal(err)
	}

	shape, err := tensor.NewShape([]int{1, 1, 2})

	if err != nil {
		benchmark.Fatal(err)
	}

	for benchmark.Loop() {
		graph := ir.NewGraph()
		projection := ir.NewNode("projection", ir.OpType("projection.linear"), shape)
		projection.SetOperationID("projection.linear")
		projection.SetMetadata("in_features", 2)
		projection.SetMetadata("out_features", 3)
		graph.AddNode(projection)
		_ = BindIR(graph, store)
	}
}

func writeTestSafeTensors(path string, tensors []testTensor) error {
	header := make(map[string]any, len(tensors))
	data := make([]byte, 0)

	for _, tensor := range tensors {
		start := len(data)

		for _, value := range tensor.values {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(float32(value)))
			data = append(data, encoded[:]...)
		}

		header[tensor.name] = map[string]any{
			"dtype":        "F32",
			"shape":        tensor.shape,
			"data_offsets": []int{start, len(data)},
		}
	}

	headerBytes, err := json.Marshal(header)

	if err != nil {
		return err
	}

	output := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(output[:8], uint64(len(headerBytes)))
	copy(output[8:], headerBytes)
	copy(output[8+len(headerBytes):], data)

	return os.WriteFile(path, output, 0o644)
}
