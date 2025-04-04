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
			WithScope(ArtifactScopeContext),
		)

		// Test string metadata
		err := artifact.SetMetaValue("string_key", "test_value")
		So(err, ShouldBeNil)

		// Test int metadata
		err = artifact.SetMetaValue("int_key", 42)
		So(err, ShouldBeNil)

		// Test float metadata
		err = artifact.SetMetaValue("float_key", 3.14)
		So(err, ShouldBeNil)

		// Test bool metadata
		err = artifact.SetMetaValue("bool_key", true)
		So(err, ShouldBeNil)

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
			WithScope(ArtifactScopeContext),
		)

		Convey("When setting string metadata", func() {
			err := artifact.SetMetaValue("string_key", "test_value")
			Convey("Then it should succeed", func() {
				So(err, ShouldBeNil)
				value := GetMetaValue[string](artifact, "string_key")
				So(value, ShouldEqual, "test_value")
			})
		})

		Convey("When setting int metadata", func() {
			err := artifact.SetMetaValue("int_key", 42)
			Convey("Then it should succeed", func() {
				So(err, ShouldBeNil)
				value := GetMetaValue[int](artifact, "int_key")
				So(value, ShouldEqual, 42)
			})
		})

		Convey("When setting float metadata", func() {
			err := artifact.SetMetaValue("float_key", 3.14)
			Convey("Then it should succeed", func() {
				So(err, ShouldBeNil)
				value := GetMetaValue[float64](artifact, "float_key")
				So(value, ShouldEqual, 3.14)
			})
		})

		Convey("When setting bool metadata", func() {
			err := artifact.SetMetaValue("bool_key", true)
			Convey("Then it should succeed", func() {
				So(err, ShouldBeNil)
				value := GetMetaValue[bool](artifact, "bool_key")
				So(value, ShouldBeTrue)
			})
		})

		Convey("When setting binary metadata", func() {
			binaryData := []byte{1, 2, 3, 4}
			err := artifact.SetMetaValue("binary_key", binaryData)
			Convey("Then it should succeed", func() {
				So(err, ShouldBeNil)
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
