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
	"github.com/theapemachine/caramba/pkg/qpool"
)

type testTensor struct {
	name   string
	shape  []int
	values []float64
}

func TestOpen(test *testing.T) {
	Convey("Given a safetensors checkpoint", test, func() {
		previousPublish := qpool.Publish
		events := make([]qpool.Event, 0)
		qpool.Publish = func(event qpool.Event) {
			events = append(events, event)
		}
		defer func() { qpool.Publish = previousPublish }()

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
			So(len(events) > 0, ShouldBeTrue)
			So(events[0].Component, ShouldEqual, "weights")
			So(events[0].Op, ShouldEqual, "open.file")
			So(events[len(events)-1].Op, ShouldEqual, "open.ready")
		})
	})
}

func TestResolve(test *testing.T) {
	Convey("Given a local checkpoint with a component-scoped safetensors index", test, func() {
		defer silenceWeightProgress()()

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
		defer silenceWeightProgress()()

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

	Convey("Given explicit SafeTensors binding metadata", test, func() {
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
		layerNorm := ir.NewNode("h.0.ln_1", ir.OpType("math.layernorm"), shape)
		layerNorm.SetOperationID("math.layernorm")
		qProjection := ir.NewNode("q_proj_0", ir.OpType("projection.linear"), shape)
		qProjection.SetOperationID("projection.linear")
		qProjection.SetMetadata("in_features", 2)
		qProjection.SetMetadata("out_features", 2)
		qProjection.SetMetadata("from_safetensors", map[string]any{
			"weight":      "h.0.attn.c_attn.weight",
			"bias":        "h.0.attn.c_attn.bias",
			"slice_axis":  "output",
			"slice_start": 0,
		})
		attentionProjection := ir.NewNode("h.0.attn.c_proj", ir.OpType("projection.linear"), shape)
		attentionProjection.SetOperationID("projection.linear")
		attentionProjection.SetMetadata("in_features", 2)
		attentionProjection.SetMetadata("out_features", 2)
		lmHead := ir.NewNode("lm_head", ir.OpType("projection.linear"), shape)
		lmHead.SetOperationID("projection.linear")
		lmHead.SetMetadata("in_features", 2)
		lmHead.SetMetadata("out_features", 3)
		lmHead.SetMetadata("from_safetensors", map[string]any{
			"weight": "wte.weight",
		})
		graph.AddNode(layerNorm)
		graph.AddNode(qProjection)
		graph.AddNode(attentionProjection)
		graph.AddNode(lmHead)

		Convey("It should bind exact names and declared tensor slices", func() {
			err := BindIR(graph, store)
			So(err, ShouldBeNil)

			So(layerNorm.Metadata()["weight"], ShouldResemble, []float64{1, 2})
			So(layerNorm.Metadata()["bias"], ShouldResemble, []float64{3, 4})
			So(qProjection.Metadata()["weight"], ShouldResemble, []float64{1, 2, 7, 8})
			So(qProjection.Metadata()["bias"], ShouldResemble, []float64{13, 14})
			So(attentionProjection.Metadata()["weight"], ShouldResemble, []float64{25, 26, 27, 28})
			So(lmHead.Metadata()["weight"], ShouldResemble, []float64{19, 21, 23, 20, 22, 24})
		})

		Convey("It should cache derived direct weight views", func() {
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

	Convey("Given VAE convolution and groupnorm safetensors names", test, func() {
		path := filepath.Join(test.TempDir(), "vae.safetensors")
		err := writeTestSafeTensors(path, []testTensor{
			{
				name:   "decoder.conv_in.weight",
				shape:  []int{2, 1, 1, 1},
				values: []float64{1, 2},
			},
			{
				name:   "decoder.conv_in.bias",
				shape:  []int{2},
				values: []float64{3, 4},
			},
			{
				name:   "decoder.norm_out.weight",
				shape:  []int{2},
				values: []float64{5, 6},
			},
			{
				name:   "decoder.norm_out.bias",
				shape:  []int{2},
				values: []float64{7, 8},
			},
			{
				name:   "decoder.up.weight",
				shape:  []int{2, 1, 1, 1},
				values: []float64{9, 10},
			},
			{
				name:   "decoder.up.bias",
				shape:  []int{1},
				values: []float64{11},
			},
		})
		So(err, ShouldBeNil)

		store, err := Open(path)
		So(err, ShouldBeNil)

		shape, err := tensor.NewShape([]int{1, 1, 1, 1})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		conv := ir.NewNode("decoder.conv_in", ir.OpType("convolution.conv2d"), shape)
		conv.SetOperationID("convolution.conv2d")
		conv.SetMetadata("in_channels", 1)
		conv.SetMetadata("out_channels", 2)
		conv.SetMetadata("kernel_h", 1)
		conv.SetMetadata("kernel_w", 1)
		norm := ir.NewNode("decoder.norm_out", ir.OpType("math.groupnorm"), shape)
		norm.SetOperationID("math.groupnorm")
		up := ir.NewNode("decoder.up", ir.OpType("convolution.conv_transpose2d"), shape)
		up.SetOperationID("convolution.conv_transpose2d")
		up.SetMetadata("in_channels", 2)
		up.SetMetadata("out_channels", 1)
		up.SetMetadata("kernel_h", 1)
		up.SetMetadata("kernel_w", 1)
		graph.AddNode(conv)
		graph.AddNode(norm)
		graph.AddNode(up)

		Convey("It should bind convolution and groupnorm tensors without transposing", func() {
			err := BindIR(graph, store)
			So(err, ShouldBeNil)

			So(conv.Metadata()["weight"], ShouldResemble, []float64{1, 2})
			So(conv.Metadata()["bias"], ShouldResemble, []float64{3, 4})
			So(norm.Metadata()["weight"], ShouldResemble, []float64{5, 6})
			So(norm.Metadata()["bias"], ShouldResemble, []float64{7, 8})
			So(up.Metadata()["weight"], ShouldResemble, []float64{9, 10})
			So(up.Metadata()["bias"], ShouldResemble, []float64{11})
		})
	})

	Convey("Given FLUX transformer safetensors names", test, func() {
		path := filepath.Join(test.TempDir(), "flux.safetensors")
		err := writeTestSafeTensors(path, []testTensor{
			{
				name:   "transformer_blocks.0.attn.to_q.weight",
				shape:  []int{2, 2},
				values: []float64{1, 2, 3, 4},
			},
			{
				name:   "transformer_blocks.0.ff.linear_in.weight",
				shape:  []int{4, 2},
				values: []float64{5, 6, 7, 8, 9, 10, 11, 12},
			},
			{
				name:  "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
				shape: []int{10, 2},
				values: []float64{
					13, 14, 15, 16,
					17, 18, 19, 20,
					21, 22, 23, 24,
					25, 26, 27, 28,
					29, 30, 31, 32,
				},
			},
			{
				name:  "single_transformer_blocks.0.attn.to_out.weight",
				shape: []int{2, 6},
				values: []float64{
					33, 34, 35, 36, 37, 38,
					39, 40, 41, 42, 43, 44,
				},
			},
		})
		So(err, ShouldBeNil)

		store, err := Open(path)
		So(err, ShouldBeNil)

		shape, err := tensor.NewShape([]int{1, 1, 2})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		affineFreeNorm := ir.NewNode(
			"transformer_blocks.0.norm1",
			ir.OpType("math.rmsnorm"),
			shape,
		)
		affineFreeNorm.SetOperationID("math.rmsnorm")
		affineFreeNorm.SetMetadata("affine", false)
		query := ir.NewNode(
			"transformer_blocks.0.attn.to_q",
			ir.OpType("projection.linear"),
			shape,
		)
		query.SetOperationID("projection.linear")
		query.SetMetadata("in_features", 2)
		query.SetMetadata("out_features", 2)
		feedForwardIn := ir.NewNode(
			"transformer_blocks.0.ff.linear_in",
			ir.OpType("projection.linear"),
			shape,
		)
		feedForwardIn.SetOperationID("projection.linear")
		feedForwardIn.SetMetadata("in_features", 2)
		feedForwardIn.SetMetadata("out_features", 4)
		singleQuery := ir.NewNode(
			"single_transformer_blocks.0.attn.to_q",
			ir.OpType("projection.linear"),
			shape,
		)
		singleQuery.SetOperationID("projection.linear")
		singleQuery.SetMetadata("in_features", 2)
		singleQuery.SetMetadata("out_features", 2)
		singleQuery.SetMetadata("from_safetensors", map[string]any{
			"weight":      "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
			"slice_axis":  "output",
			"slice_start": 0,
		})
		singleMLP := ir.NewNode(
			"single_transformer_blocks.0.proj_mlp",
			ir.OpType("projection.linear"),
			shape,
		)
		singleMLP.SetOperationID("projection.linear")
		singleMLP.SetMetadata("in_features", 2)
		singleMLP.SetMetadata("out_features", 4)
		singleMLP.SetMetadata("from_safetensors", map[string]any{
			"weight":      "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
			"slice_axis":  "output",
			"slice_start": 6,
		})
		singleAttentionOut := ir.NewNode(
			"single_transformer_blocks.0.attn.to_out.0",
			ir.OpType("projection.linear"),
			shape,
		)
		singleAttentionOut.SetOperationID("projection.linear")
		singleAttentionOut.SetMetadata("in_features", 2)
		singleAttentionOut.SetMetadata("out_features", 2)
		singleAttentionOut.SetMetadata("from_safetensors", map[string]any{
			"weight":      "single_transformer_blocks.0.attn.to_out.weight",
			"slice_axis":  "input",
			"slice_start": 0,
		})
		singleMLPOut := ir.NewNode(
			"single_transformer_blocks.0.proj_out",
			ir.OpType("projection.linear"),
			shape,
		)
		singleMLPOut.SetOperationID("projection.linear")
		singleMLPOut.SetMetadata("in_features", 4)
		singleMLPOut.SetMetadata("out_features", 2)
		singleMLPOut.SetMetadata("from_safetensors", map[string]any{
			"weight":      "single_transformer_blocks.0.attn.to_out.weight",
			"slice_axis":  "input",
			"slice_start": 2,
		})
		graph.AddNode(affineFreeNorm)
		graph.AddNode(query)
		graph.AddNode(feedForwardIn)
		graph.AddNode(singleQuery)
		graph.AddNode(singleMLP)
		graph.AddNode(singleAttentionOut)
		graph.AddNode(singleMLPOut)

		Convey("It should bind affine-free norms and packed FLUX projections", func() {
			err := BindIR(graph, store)
			So(err, ShouldBeNil)

			_, bound := affineFreeNorm.Metadata()["weight"]
			So(bound, ShouldBeFalse)
			So(query.Metadata()["weight"], ShouldResemble, []float64{1, 3, 2, 4})
			So(
				feedForwardIn.Metadata()["weight"],
				ShouldResemble,
				[]float64{5, 7, 9, 11, 6, 8, 10, 12},
			)
			So(singleQuery.Metadata()["weight"], ShouldResemble, []float64{13, 15, 14, 16})
			So(
				singleMLP.Metadata()["weight"],
				ShouldResemble,
				[]float64{25, 27, 29, 31, 26, 28, 30, 32},
			)
			So(
				singleAttentionOut.Metadata()["weight"],
				ShouldResemble,
				[]float64{33, 39, 34, 40},
			)
			So(
				singleMLPOut.Metadata()["weight"],
				ShouldResemble,
				[]float64{35, 41, 36, 42, 37, 43, 38, 44},
			)
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
		qProjection := ir.NewNode(
			"model.layers.0.self_attn.q_proj",
			ir.OpType("projection.linear"),
			shape,
		)
		qProjection.SetOperationID("projection.linear")
		qProjection.SetMetadata("in_features", 2)
		qProjection.SetMetadata("out_features", 3)
		oProjection := ir.NewNode(
			"model.layers.0.self_attn.o_proj",
			ir.OpType("projection.linear"),
			shape,
		)
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

func silenceWeightProgress() func() {
	previousPublish := qpool.Publish
	qpool.Publish = func(qpool.Event) {}

	return func() {
		qpool.Publish = previousPublish
	}
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
