package datura

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestGetMetaValue(t *testing.T) {
	Convey("Given an artifact with metadata", t, func() {
		artifact := New(
			WithMediatype(MediaTypeTextPlain),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeError),
		)

		// Test string metadata
		artifact.SetMetaValue("string_key", "test_value")

		// Test int metadata
		artifact.SetMetaValue("int_key", 42)

		// Test float metadata
		artifact.SetMetaValue("float_key", 3.14)

		// Test bool metadata
		artifact.SetMetaValue("bool_key", true)

		Convey("When retrieving string metadata", func() {
			value := GetMetaValue[string](artifact, "string_key")
			Convey("Then it should return the correct value", func() {
				So(value, ShouldEqual, "test_value")
			})
		})

		Convey("When retrieving int metadata", func() {
			value := GetMetaValue[int](artifact, "int_key")
			Convey("Then it should return the correct value", func() {
				So(value, ShouldEqual, 42)
			})
		})

		Convey("When retrieving float metadata", func() {
			value := GetMetaValue[float64](artifact, "float_key")
			Convey("Then it should return the correct value", func() {
				So(value, ShouldEqual, 3.14)
			})
		})

		Convey("When retrieving bool metadata", func() {
			value := GetMetaValue[bool](artifact, "bool_key")
			Convey("Then it should return the correct value", func() {
				So(value, ShouldBeTrue)
			})
		})

		Convey("When retrieving non-existent metadata", func() {
			value := GetMetaValue[string](artifact, "non_existent")
			Convey("Then it should return zero value", func() {
				So(value, ShouldBeEmpty)
			})
		})
	})
}

func TestSetMetaValue(t *testing.T) {
	Convey("Given an artifact", t, func() {
		artifact := New(
			WithMediatype(MediaTypeTextPlain),
			WithRole(ArtifactRoleUser),
			WithScope(ArtifactScopeError),
		)

		Convey("When setting string metadata", func() {
			artifact.SetMetaValue("string_key", "test_value")
			Convey("Then it should succeed", func() {
				value := GetMetaValue[string](artifact, "string_key")
				So(value, ShouldEqual, "test_value")
			})
		})

		Convey("When setting int metadata", func() {
			artifact.SetMetaValue("int_key", 42)
			Convey("Then it should succeed", func() {
				value := GetMetaValue[int](artifact, "int_key")
				So(value, ShouldEqual, 42)
			})
		})

		Convey("When setting float metadata", func() {
			artifact.SetMetaValue("float_key", 3.14)
			Convey("Then it should succeed", func() {
				value := GetMetaValue[float64](artifact, "float_key")
				So(value, ShouldEqual, 3.14)
			})
		})

		Convey("When setting bool metadata", func() {
			artifact.SetMetaValue("bool_key", true)
			Convey("Then it should succeed", func() {
				value := GetMetaValue[bool](artifact, "bool_key")
				So(value, ShouldBeTrue)
			})
		})

		Convey("When setting binary metadata", func() {
			binaryData := []byte{1, 2, 3, 4}
			artifact.SetMetaValue("binary_key", binaryData)
			Convey("Then it should succeed", func() {
				mdList, err := artifact.Metadata()
				So(err, ShouldBeNil)

				var found bool
				for i := 0; i < mdList.Len(); i++ {
					item := mdList.At(i)
					key, err := item.Key()
					So(err, ShouldBeNil)
					if key == "binary_key" {
						found = true
						value, err := item.Value().BinaryValue()
						So(err, ShouldBeNil)
						So(value, ShouldResemble, binaryData)
					}
				}
				So(found, ShouldBeTrue)
			})
		})
	})
}
