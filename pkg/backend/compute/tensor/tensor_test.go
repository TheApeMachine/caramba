package tensor

import (
	"testing"

	convey "github.com/smartystreets/goconvey/convey"
)

func TestDType_Size(t *testing.T) {
	convey.Convey("Given a supported dtype", t, func() {
		convey.Convey("It should return the scalar byte width", func() {
			size, err := Float64.Size()

			convey.So(err, convey.ShouldBeNil)
			convey.So(size, convey.ShouldEqual, 8)
		})
	})

	convey.Convey("Given an unsupported dtype", t, func() {
		convey.Convey("It should reject the dtype", func() {
			size, err := DType("complex128").Size()

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(size, convey.ShouldEqual, 0)
		})
	})
}

func TestNewShape(t *testing.T) {
	convey.Convey("Given valid tensor dimensions", t, func() {
		shape, err := NewShape([]int{2, 3, 4})

		convey.Convey("It should cache the element count", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Len(), convey.ShouldEqual, 24)
			convey.So(shape.Valid(), convey.ShouldBeTrue)
		})
	})

	convey.Convey("Given a scalar shape", t, func() {
		shape, err := NewShape(nil)

		convey.Convey("It should represent one scalar element", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Len(), convey.ShouldEqual, 1)
		})
	})

	convey.Convey("Given a zero dimension", t, func() {
		shape, err := NewShape([]int{4, 0, 8})

		convey.Convey("It should represent an empty tensor", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(shape.Len(), convey.ShouldEqual, 0)
		})
	})

	convey.Convey("Given a negative dimension", t, func() {
		shape, err := NewShape([]int{2, -1})

		convey.Convey("It should reject the shape", func() {
			convey.So(err, convey.ShouldNotBeNil)
			convey.So(shape.Valid(), convey.ShouldBeFalse)
		})
	})
}

func TestShape_Dims(t *testing.T) {
	convey.Convey("Given a shape", t, func() {
		shape, err := NewShape([]int{2, 3})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should return a defensive dimension copy", func() {
			dims := shape.Dims()
			dims[0] = 99

			convey.So(shape.Dims(), convey.ShouldResemble, []int{2, 3})
		})
	})
}

func TestShape_Bytes(t *testing.T) {
	convey.Convey("Given a valid shape and dtype", t, func() {
		shape, err := NewShape([]int{2, 3})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should calculate storage bytes", func() {
			bytes, err := shape.Bytes(Float64)

			convey.So(err, convey.ShouldBeNil)
			convey.So(bytes, convey.ShouldEqual, 48)
		})
	})

	convey.Convey("Given an invalid shape", t, func() {
		convey.Convey("It should reject byte calculation", func() {
			bytes, err := Shape{}.Bytes(Float64)

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(bytes, convey.ShouldEqual, 0)
		})
	})
}

func TestShape_Equal(t *testing.T) {
	convey.Convey("Given two equal shapes", t, func() {
		left, err := NewShape([]int{2, 3})
		convey.So(err, convey.ShouldBeNil)

		right, err := NewShape([]int{2, 3})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should report equality", func() {
			convey.So(left.Equal(right), convey.ShouldBeTrue)
		})
	})

	convey.Convey("Given shapes that differ by dimension order", t, func() {
		left, err := NewShape([]int{2, 3})
		convey.So(err, convey.ShouldBeNil)

		right, err := NewShape([]int{3, 2})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should report inequality", func() {
			convey.So(left.Equal(right), convey.ShouldBeFalse)
		})
	})

	convey.Convey("Given shapes that differ by rank", t, func() {
		left, err := NewShape([]int{2, 3})
		convey.So(err, convey.ShouldBeNil)

		right, err := NewShape([]int{2, 3, 1})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should report inequality", func() {
			convey.So(left.Equal(right), convey.ShouldBeFalse)
		})
	})
}

func TestFloat64From(t *testing.T) {
	convey.Convey("Given float64 values", t, func() {
		value, err := Float64From([]float64{1, 2, 3})

		convey.Convey("It should create a host-owned vector tensor", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(value.Location(), convey.ShouldEqual, Host)
			convey.So(value.Shape().Dims(), convey.ShouldResemble, []int{3})
			convey.So(MustCloneFloat64(value), convey.ShouldResemble, []float64{1, 2, 3})
		})
	})
}

func TestNewHostBackend(t *testing.T) {
	convey.Convey("Given a new host backend", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		convey.Convey("It should report host ownership", func() {
			convey.So(hostBackend.Location(), convey.ShouldEqual, Host)
		})
	})
}

func TestHostBackend_UploadFloat64(t *testing.T) {
	convey.Convey("Given a host backend and valid values", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{2, 2})
		convey.So(err, convey.ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2, 3, 4})

		convey.Convey("It should create persistent host storage", func() {
			convey.So(err, convey.ShouldBeNil)
			defer func() { convey.So(uploaded.Close(), convey.ShouldBeNil) }()

			convey.So(uploaded.Location(), convey.ShouldEqual, Host)
			convey.So(uploaded.DType(), convey.ShouldEqual, Float64)
			convey.So(uploaded.Len(), convey.ShouldEqual, 4)
			convey.So(uploaded.Bytes(), convey.ShouldEqual, 32)
		})
	})

	convey.Convey("Given mismatched shape and values", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{2, 3})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should reject the upload", func() {
			uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2, 3, 4})

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(uploaded, convey.ShouldBeNil)
		})
	})

	convey.Convey("Given an upload larger than the remaining arena", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{hostArenaFloat64Elements + 1})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should reject the upload instead of falling back to heap storage", func() {
			uploaded, err := hostBackend.UploadFloat64(shape, make([]float64, shape.Len()))

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "host arena exhausted")
			convey.So(uploaded, convey.ShouldBeNil)
		})
	})

	convey.Convey("Given an upload that exactly fills the arena", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{hostArenaFloat64Elements})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should use the arena without reporting exhaustion", func() {
			uploaded, err := hostBackend.UploadFloat64(shape, make([]float64, shape.Len()))

			convey.So(err, convey.ShouldBeNil)
			defer func() { convey.So(uploaded.Close(), convey.ShouldBeNil) }()

			convey.So(uploaded.Len(), convey.ShouldEqual, hostArenaFloat64Elements)
		})
	})

	convey.Convey("Given a closed arena tensor at the allocation tail", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		smallShape, err := NewShape([]int{4})
		convey.So(err, convey.ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(smallShape, []float64{1, 2, 3, 4})
		convey.So(err, convey.ShouldBeNil)
		convey.So(uploaded.Close(), convey.ShouldBeNil)

		fullShape, err := NewShape([]int{hostArenaFloat64Elements})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should rewind released tail capacity for the next upload", func() {
			reused, err := hostBackend.UploadFloat64(fullShape, make([]float64, fullShape.Len()))

			convey.So(err, convey.ShouldBeNil)
			defer func() { convey.So(reused.Close(), convey.ShouldBeNil) }()

			convey.So(reused.Len(), convey.ShouldEqual, hostArenaFloat64Elements)
		})
	})

	convey.Convey("Given a closed arena tensor before a live allocation", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		firstShape, err := NewShape([]int{4})
		convey.So(err, convey.ShouldBeNil)
		secondShape, err := NewShape([]int{6})
		convey.So(err, convey.ShouldBeNil)

		first, err := hostBackend.UploadFloat64(firstShape, []float64{1, 2, 3, 4})
		convey.So(err, convey.ShouldBeNil)
		second, err := hostBackend.UploadFloat64(secondShape, []float64{5, 6, 7, 8, 9, 10})
		convey.So(err, convey.ShouldBeNil)
		defer func() { convey.So(second.Close(), convey.ShouldBeNil) }()
		convey.So(first.Close(), convey.ShouldBeNil)

		convey.Convey("It should reuse the released span without disturbing live tensors", func() {
			reused, err := hostBackend.UploadFloat64(firstShape, []float64{11, 12, 13, 14})

			convey.So(err, convey.ShouldBeNil)
			defer func() { convey.So(reused.Close(), convey.ShouldBeNil) }()

			values, err := second.CloneFloat64()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float64{5, 6, 7, 8, 9, 10})

			values, err = reused.CloneFloat64()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float64{11, 12, 13, 14})
		})
	})

	convey.Convey("Given an empty tensor upload", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{0})
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should succeed without consuming arena capacity", func() {
			uploaded, err := hostBackend.UploadFloat64(shape, []float64{})

			convey.So(err, convey.ShouldBeNil)
			defer func() { convey.So(uploaded.Close(), convey.ShouldBeNil) }()

			convey.So(uploaded.Len(), convey.ShouldEqual, 0)
		})
	})
}

func TestHostBackend_AdoptFloat64(t *testing.T) {
	convey.Convey("Given fresh output values", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{2})
		convey.So(err, convey.ShouldBeNil)

		values := []float64{1, 2}
		uploaded, err := hostBackend.AdoptFloat64(shape, values)

		convey.Convey("It should take ownership without copying", func() {
			convey.So(err, convey.ShouldBeNil)
			defer func() { convey.So(uploaded.Close(), convey.ShouldBeNil) }()

			values[0] = 9
			cloned, err := uploaded.CloneFloat64()

			convey.So(err, convey.ShouldBeNil)
			convey.So(cloned, convey.ShouldResemble, []float64{9, 2})
		})
	})
}

func TestHostBackend_DownloadFloat64(t *testing.T) {
	convey.Convey("Given a host tensor", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{3})
		convey.So(err, convey.ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2, 3})
		convey.So(err, convey.ShouldBeNil)
		defer func() { convey.So(uploaded.Close(), convey.ShouldBeNil) }()

		convey.Convey("It should return a zero-copy host view", func() {
			values, err := hostBackend.DownloadFloat64(uploaded)
			convey.So(err, convey.ShouldBeNil)

			values[0] = 99
			again, err := hostBackend.DownloadFloat64(uploaded)

			convey.So(err, convey.ShouldBeNil)
			convey.So(again[0], convey.ShouldEqual, 99)
			convey.So(again, convey.ShouldResemble, []float64{99, 2, 3})

			cloned, err := uploaded.CloneFloat64()
			convey.So(err, convey.ShouldBeNil)
			convey.So(cloned, convey.ShouldResemble, []float64{99, 2, 3})

			cloned[0] = 1
			third, err := hostBackend.DownloadFloat64(uploaded)
			convey.So(err, convey.ShouldBeNil)
			convey.So(third[0], convey.ShouldEqual, 99)
		})
	})
}

func TestHostBackend_Close(t *testing.T) {
	convey.Convey("Given a closed host backend", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		err := hostBackend.Close()
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should reject future uploads", func() {
			shape, err := NewShape([]int{1})
			convey.So(err, convey.ShouldBeNil)

			uploaded, err := hostBackend.UploadFloat64(shape, []float64{1})

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(uploaded, convey.ShouldBeNil)
		})
	})
}

func TestHostBackend_Reset(t *testing.T) {
	convey.Convey("Given a host backend with a live arena tensor", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{2})
		convey.So(err, convey.ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2})
		convey.So(err, convey.ShouldBeNil)
		defer func() { convey.So(uploaded.Close(), convey.ShouldBeNil) }()

		convey.Convey("It should reject reset while the tensor can still alias arena memory", func() {
			err := hostBackend.Reset()

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err.Error(), convey.ShouldContainSubstring, "host arena still has live tensors")
		})
	})

	convey.Convey("Given all arena tensors are closed", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{2})
		convey.So(err, convey.ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2})
		convey.So(err, convey.ShouldBeNil)
		convey.So(uploaded.Close(), convey.ShouldBeNil)

		convey.Convey("It should reset arena capacity", func() {
			err := hostBackend.Reset()
			convey.So(err, convey.ShouldBeNil)

			reused, err := hostBackend.UploadFloat64(shape, []float64{3, 4})
			convey.So(err, convey.ShouldBeNil)
			defer func() { convey.So(reused.Close(), convey.ShouldBeNil) }()

			values, err := reused.CloneFloat64()
			convey.So(err, convey.ShouldBeNil)
			convey.So(values, convey.ShouldResemble, []float64{3, 4})
		})
	})

	convey.Convey("Given an adopted tensor", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{2})
		convey.So(err, convey.ShouldBeNil)

		adopted, err := hostBackend.AdoptFloat64(shape, []float64{1, 2})
		convey.So(err, convey.ShouldBeNil)
		defer func() { convey.So(adopted.Close(), convey.ShouldBeNil) }()

		convey.Convey("It should not block arena reset because it does not alias arena memory", func() {
			convey.So(hostBackend.Reset(), convey.ShouldBeNil)
		})
	})

	convey.Convey("Given a closed host backend", t, func() {
		hostBackend := NewHostBackend()
		convey.So(hostBackend.Close(), convey.ShouldBeNil)

		convey.Convey("It should reject reset", func() {
			err := hostBackend.Reset()

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(err, convey.ShouldEqual, errClosedBackend)
		})
	})
}

func TestHostTensor_Float64(t *testing.T) {
	convey.Convey("Given a host tensor", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{2})
		convey.So(err, convey.ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2})
		convey.So(err, convey.ShouldBeNil)
		defer func() { convey.So(uploaded.Close(), convey.ShouldBeNil) }()

		hostTensor, ok := uploaded.(*HostTensor)
		convey.So(ok, convey.ShouldBeTrue)

		convey.Convey("It should expose a zero-copy CPU view", func() {
			values, err := hostTensor.Float64()
			convey.So(err, convey.ShouldBeNil)

			values[0] = 9
			cloned, err := hostTensor.CloneFloat64()

			convey.So(err, convey.ShouldBeNil)
			convey.So(cloned, convey.ShouldResemble, []float64{9, 2})
		})
	})
}

func TestHostTensor_Close(t *testing.T) {
	convey.Convey("Given a closed host tensor", t, func() {
		hostBackend := NewHostBackend()
		defer func() { convey.So(hostBackend.Close(), convey.ShouldBeNil) }()

		shape, err := NewShape([]int{1})
		convey.So(err, convey.ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1})
		convey.So(err, convey.ShouldBeNil)

		err = uploaded.Close()
		convey.So(err, convey.ShouldBeNil)

		convey.Convey("It should reject host reads", func() {
			values, err := uploaded.CloneFloat64()

			convey.So(err, convey.ShouldNotBeNil)
			convey.So(values, convey.ShouldBeNil)
		})

		convey.Convey("Repeated Close should be harmless", func() {
			convey.So(uploaded.Close(), convey.ShouldBeNil)
		})
	})
}

func BenchmarkHostBackend_UploadFloat64(benchmark *testing.B) {
	hostBackend := NewHostBackend()
	defer func() { _ = hostBackend.Close() }()

	shape, err := NewShape([]int{1024})

	if err != nil {
		benchmark.Fatal(err)
	}

	values := make([]float64, shape.Len())

	benchmark.ReportAllocs()
	benchmark.SetBytes(int64(shape.Len() * 8))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		uploaded, err := hostBackend.UploadFloat64(shape, values)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := uploaded.Close(); err != nil {
			benchmark.Fatal(err)
		}

		if err := hostBackend.Reset(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkHostBackend_DownloadFloat64(benchmark *testing.B) {
	hostBackend := NewHostBackend()
	defer func() { _ = hostBackend.Close() }()

	shape, err := NewShape([]int{1024})

	if err != nil {
		benchmark.Fatal(err)
	}

	uploaded, err := hostBackend.UploadFloat64(shape, make([]float64, shape.Len()))

	if err != nil {
		benchmark.Fatal(err)
	}

	defer func() { _ = uploaded.Close() }()

	benchmark.ResetTimer()
	benchmark.ReportAllocs()
	benchmark.SetBytes(int64(shape.Len() * 8))

	for benchmark.Loop() {
		_, err := hostBackend.DownloadFloat64(uploaded)

		if err != nil {
			benchmark.Fatal(err)
		}
	}
}
