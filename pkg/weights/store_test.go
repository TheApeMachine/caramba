package weights

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestStoreLookupTransposedReadsCheckpointBytes(testingObject *testing.T) {
	convey.Convey("Given a safetensors weight store with opaque resident tensors", testingObject, func() {
		archivePath := writeSafetensorsFixture(
			testingObject,
			"layers.0.weight",
			[]int{2, 3},
			[]float32{1, 2, 3, 4, 5, 6},
		)
		memory := &recordingWeightBackend{}

		store, err := New(memory, []string{archivePath})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should transpose before uploading to resident storage", func() {
			resident, err := store.LookupTransposed("layers.0.weight")

			convey.So(err, convey.ShouldBeNil)
			convey.So(resident, convey.ShouldNotBeNil)
			convey.So(memory.uploads, convey.ShouldHaveLength, 1)
			convey.So(memory.uploads[0].shape.Dims(), convey.ShouldResemble, []int{3, 2})
			convey.So(memory.uploads[0].dataType, convey.ShouldEqual, dtype.Float32)
			convey.So(float32BytesToValues(memory.uploads[0].payload), convey.ShouldResemble, []float32{
				1, 4,
				2, 5,
				3, 6,
			})
		})
	})
}

func writeSafetensorsFixture(
	testingObject *testing.T,
	name string,
	dimensions []int,
	values []float32,
) string {
	testingObject.Helper()

	data := float32ValuesToBytes(values)
	shape := make([]int64, len(dimensions))

	for index, dimension := range dimensions {
		shape[index] = int64(dimension)
	}

	header := map[string]any{
		name: map[string]any{
			"dtype":        "F32",
			"shape":        shape,
			"data_offsets": []int64{0, int64(len(data))},
		},
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		testingObject.Fatalf("writeSafetensorsFixture: header: %v", err)
	}

	buffer := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(buffer[:8], uint64(len(headerBytes)))
	copy(buffer[8:], headerBytes)
	copy(buffer[8+len(headerBytes):], data)

	path := filepath.Join(testingObject.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, buffer, 0600); err != nil {
		testingObject.Fatalf("writeSafetensorsFixture: write: %v", err)
	}

	return path
}

func float32BytesToValues(rawBytes []byte) []float32 {
	values := make([]float32, len(rawBytes)/4)

	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(rawBytes[index*4:]))
	}

	return values
}

type weightUpload struct {
	shape    tensor.Shape
	dataType dtype.DType
	payload  []byte
}

type recordingWeightBackend struct {
	uploads []weightUpload
}

func (backend *recordingWeightBackend) Location() tensor.Location {
	return tensor.Metal
}

func (backend *recordingWeightBackend) SupportedDTypes() []dtype.DType {
	return []dtype.DType{dtype.Float32}
}

func (backend *recordingWeightBackend) SupportedLayouts() []tensor.Layout {
	return []tensor.Layout{tensor.LayoutDense}
}

func (backend *recordingWeightBackend) Capabilities() tensor.Capabilities {
	return tensor.Capabilities{}
}

func (backend *recordingWeightBackend) Upload(
	shape tensor.Shape,
	sourceDType dtype.DType,
	rawBytes []byte,
) (tensor.Tensor, error) {
	backend.uploads = append(backend.uploads, weightUpload{
		shape:    shape,
		dataType: sourceDType,
		payload:  append([]byte(nil), rawBytes...),
	})

	resident, err := tensor.New(shape, sourceDType)
	if err != nil {
		return nil, err
	}

	return &opaqueWeightTensor{Tensor: resident}, nil
}

func (backend *recordingWeightBackend) UploadAsync(
	shape tensor.Shape,
	sourceDType dtype.DType,
	rawBytes []byte,
) (tensor.Tensor, error) {
	return backend.Upload(shape, sourceDType, rawBytes)
}

func (backend *recordingWeightBackend) UploadSparse(
	shape tensor.Shape,
	valueDType dtype.DType,
	layout tensor.Layout,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (backend *recordingWeightBackend) Download(input tensor.Tensor) (dtype.DType, []byte, error) {
	return dtype.Invalid, nil, tensor.ErrLayoutUnsupported
}

func (backend *recordingWeightBackend) Close() error {
	return nil
}

type opaqueWeightTensor struct {
	tensor.Tensor
}

func (resident *opaqueWeightTensor) Location() tensor.Location {
	return tensor.Metal
}

func (resident *opaqueWeightTensor) Float32Native() ([]float32, error) {
	return nil, tensor.ErrDTypeMismatch
}
