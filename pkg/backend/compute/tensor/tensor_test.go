package tensor

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestDType_Size(t *testing.T) {
	Convey("Given a supported dtype", t, func() {
		Convey("It should return the scalar byte width", func() {
			size, err := Float64.Size()

			So(err, ShouldBeNil)
			So(size, ShouldEqual, 8)
		})
	})

	Convey("Given an unsupported dtype", t, func() {
		Convey("It should reject the dtype", func() {
			size, err := DType("complex128").Size()

			So(err, ShouldNotBeNil)
			So(size, ShouldEqual, 0)
		})
	})
}

func TestNewShape(t *testing.T) {
	Convey("Given valid tensor dimensions", t, func() {
		shape, err := NewShape([]int{2, 3, 4})

		Convey("It should cache the element count", func() {
			So(err, ShouldBeNil)
			So(shape.Len(), ShouldEqual, 24)
			So(shape.Valid(), ShouldBeTrue)
		})
	})

	Convey("Given a scalar shape", t, func() {
		shape, err := NewShape(nil)

		Convey("It should represent one scalar element", func() {
			So(err, ShouldBeNil)
			So(shape.Len(), ShouldEqual, 1)
		})
	})

	Convey("Given a zero dimension", t, func() {
		shape, err := NewShape([]int{4, 0, 8})

		Convey("It should represent an empty tensor", func() {
			So(err, ShouldBeNil)
			So(shape.Len(), ShouldEqual, 0)
		})
	})

	Convey("Given a negative dimension", t, func() {
		shape, err := NewShape([]int{2, -1})

		Convey("It should reject the shape", func() {
			So(err, ShouldNotBeNil)
			So(shape.Valid(), ShouldBeFalse)
		})
	})
}

func TestShape_Dims(t *testing.T) {
	Convey("Given a shape", t, func() {
		shape, err := NewShape([]int{2, 3})
		So(err, ShouldBeNil)

		Convey("It should return a defensive dimension copy", func() {
			dims := shape.Dims()
			dims[0] = 99

			So(shape.Dims(), ShouldResemble, []int{2, 3})
		})
	})
}

func TestShape_Bytes(t *testing.T) {
	Convey("Given a valid shape and dtype", t, func() {
		shape, err := NewShape([]int{2, 3})
		So(err, ShouldBeNil)

		Convey("It should calculate storage bytes", func() {
			bytes, err := shape.Bytes(Float64)

			So(err, ShouldBeNil)
			So(bytes, ShouldEqual, 48)
		})
	})

	Convey("Given an invalid shape", t, func() {
		Convey("It should reject byte calculation", func() {
			bytes, err := Shape{}.Bytes(Float64)

			So(err, ShouldNotBeNil)
			So(bytes, ShouldEqual, 0)
		})
	})
}

func TestShape_Equal(t *testing.T) {
	Convey("Given two equal shapes", t, func() {
		left, err := NewShape([]int{2, 3})
		So(err, ShouldBeNil)

		right, err := NewShape([]int{2, 3})
		So(err, ShouldBeNil)

		Convey("It should report equality", func() {
			So(left.Equal(right), ShouldBeTrue)
		})
	})
}

func TestNewHostBackend(t *testing.T) {
	Convey("Given a new host backend", t, func() {
		hostBackend := NewHostBackend()

		Convey("It should report host ownership", func() {
			So(hostBackend.Location(), ShouldEqual, Host)
		})
	})
}

func TestHostBackend_UploadFloat64(t *testing.T) {
	Convey("Given a host backend and valid values", t, func() {
		hostBackend := NewHostBackend()
		shape, err := NewShape([]int{2, 2})
		So(err, ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2, 3, 4})

		Convey("It should create persistent host storage", func() {
			So(err, ShouldBeNil)
			So(uploaded.Location(), ShouldEqual, Host)
			So(uploaded.DType(), ShouldEqual, Float64)
			So(uploaded.Len(), ShouldEqual, 4)
			So(uploaded.Bytes(), ShouldEqual, 32)
		})
	})

	Convey("Given mismatched shape and values", t, func() {
		hostBackend := NewHostBackend()
		shape, err := NewShape([]int{2, 3})
		So(err, ShouldBeNil)

		Convey("It should reject the upload", func() {
			uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2, 3, 4})

			So(err, ShouldNotBeNil)
			So(uploaded, ShouldBeNil)
		})
	})
}

func TestHostBackend_AdoptFloat64(t *testing.T) {
	Convey("Given fresh output values", t, func() {
		hostBackend := NewHostBackend()
		shape, err := NewShape([]int{2})
		So(err, ShouldBeNil)

		values := []float64{1, 2}
		uploaded, err := hostBackend.AdoptFloat64(shape, values)

		Convey("It should take ownership without copying", func() {
			So(err, ShouldBeNil)

			values[0] = 9
			cloned, err := uploaded.CloneFloat64()

			So(err, ShouldBeNil)
			So(cloned, ShouldResemble, []float64{9, 2})
		})
	})
}

func TestHostBackend_DownloadFloat64(t *testing.T) {
	Convey("Given a host tensor", t, func() {
		hostBackend := NewHostBackend()
		shape, err := NewShape([]int{3})
		So(err, ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2, 3})
		So(err, ShouldBeNil)

		Convey("It should return a defensive host copy", func() {
			values, err := hostBackend.DownloadFloat64(uploaded)
			So(err, ShouldBeNil)

			values[0] = 99
			again, err := hostBackend.DownloadFloat64(uploaded)

			So(err, ShouldBeNil)
			So(again, ShouldResemble, []float64{1, 2, 3})
		})
	})
}

func TestHostBackend_Close(t *testing.T) {
	Convey("Given a closed host backend", t, func() {
		hostBackend := NewHostBackend()
		err := hostBackend.Close()
		So(err, ShouldBeNil)

		Convey("It should reject future uploads", func() {
			shape, err := NewShape([]int{1})
			So(err, ShouldBeNil)

			uploaded, err := hostBackend.UploadFloat64(shape, []float64{1})

			So(err, ShouldNotBeNil)
			So(uploaded, ShouldBeNil)
		})
	})
}

func TestHostTensor_Float64(t *testing.T) {
	Convey("Given a host tensor", t, func() {
		hostBackend := NewHostBackend()
		shape, err := NewShape([]int{2})
		So(err, ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1, 2})
		So(err, ShouldBeNil)

		hostTensor, ok := uploaded.(*HostTensor)
		So(ok, ShouldBeTrue)

		Convey("It should expose a zero-copy CPU view", func() {
			values, err := hostTensor.Float64()
			So(err, ShouldBeNil)

			values[0] = 9
			cloned, err := hostTensor.CloneFloat64()

			So(err, ShouldBeNil)
			So(cloned, ShouldResemble, []float64{9, 2})
		})
	})
}

func TestHostTensor_Close(t *testing.T) {
	Convey("Given a closed host tensor", t, func() {
		hostBackend := NewHostBackend()
		shape, err := NewShape([]int{1})
		So(err, ShouldBeNil)

		uploaded, err := hostBackend.UploadFloat64(shape, []float64{1})
		So(err, ShouldBeNil)

		err = uploaded.Close()
		So(err, ShouldBeNil)

		Convey("It should reject host reads", func() {
			values, err := uploaded.CloneFloat64()

			So(err, ShouldNotBeNil)
			So(values, ShouldBeNil)
		})
	})
}

func BenchmarkHostBackend_UploadFloat64(benchmark *testing.B) {
	hostBackend := NewHostBackend()
	shape, err := NewShape([]int{1024})

	if err != nil {
		benchmark.Fatal(err)
	}

	values := make([]float64, shape.Len())

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		uploaded, err := hostBackend.UploadFloat64(shape, values)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := uploaded.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkHostBackend_DownloadFloat64(benchmark *testing.B) {
	hostBackend := NewHostBackend()
	shape, err := NewShape([]int{1024})

	if err != nil {
		benchmark.Fatal(err)
	}

	uploaded, err := hostBackend.UploadFloat64(shape, make([]float64, shape.Len()))

	if err != nil {
		benchmark.Fatal(err)
	}

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		_, err := hostBackend.DownloadFloat64(uploaded)

		if err != nil {
			benchmark.Fatal(err)
		}
	}
}
