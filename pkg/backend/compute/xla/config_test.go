package xla

import (
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewPJRTConfig(test *testing.T) {
	Convey("Given PJRT environment variables", test, func() {
		clearPJRTEnv(test)

		includeDir := test.TempDir()
		pluginFile := filepath.Join(test.TempDir(), "pjrt_c_api_gpu_plugin.so")
		test.Setenv("CARAMBA_XLA_INCLUDE_DIR", includeDir)
		test.Setenv("CARAMBA_PJRT_GPU_PLUGIN", pluginFile)

		Convey("It should build a normalized GPU config", func() {
			config, err := NewPJRTConfig("cuda")

			So(err, ShouldBeNil)
			So(config.Platform, ShouldEqual, "gpu")
			So(config.IncludeDir, ShouldEqual, includeDir)
			So(config.PluginFile, ShouldEqual, pluginFile)
		})
	})
}

func TestNewPJRTConfig_UnsupportedPlatform(test *testing.T) {
	Convey("Given an unsupported platform token", test, func() {
		_, err := NewPJRTConfig("potato")

		Convey("It should reject the configuration", func() {
			So(err, ShouldNotBeNil)
		})
	})
}

func TestPJRTConfig_HeaderFile(test *testing.T) {
	Convey("Given an include directory", test, func() {
		config := PJRTConfig{IncludeDir: "/opt/xla"}

		Convey("It should point at the PJRT C API header", func() {
			So(config.HeaderFile(), ShouldEqual, "/opt/xla/xla/pjrt/c/pjrt_c_api.h")
		})
	})
}

func TestPJRTConfig_PluginName(test *testing.T) {
	Convey("Given a GPU platform", test, func() {
		config := PJRTConfig{Platform: "gpu"}

		Convey("It should select the GPU plugin name", func() {
			So(config.PluginName(), ShouldEqual, "pjrt_c_api_gpu_plugin.so")
		})
	})

	Convey("Given a CPU platform", test, func() {
		config := PJRTConfig{Platform: "cpu"}

		Convey("It should select the CPU plugin name", func() {
			So(config.PluginName(), ShouldEqual, "pjrt_c_api_cpu_plugin.so")
		})
	})

	Convey("Given plugin variants", test, func() {
		config := PJRTConfig{Platform: "cpu"}

		Convey("It should include shared library names used on Unix and Darwin", func() {
			So(config.PluginNames(), ShouldResemble, []string{
				"pjrt_c_api_cpu_plugin.so",
				"pjrt_c_api_cpu_plugin.dylib",
			})
		})
	})
}

func TestPJRTConfig_ResolvedPluginFile(test *testing.T) {
	Convey("Given an explicit plugin path", test, func() {
		pluginFile := filepath.Join(test.TempDir(), "pjrt_c_api_cpu_plugin.so")
		err := os.WriteFile(pluginFile, []byte{}, 0o600)
		So(err, ShouldBeNil)

		config := PJRTConfig{
			Platform:   "cpu",
			PluginFile: pluginFile,
		}

		Convey("It should resolve the explicit plugin", func() {
			So(config.ResolvedPluginFile(), ShouldEqual, pluginFile)
		})
	})

	Convey("Given a plugin on the library path", test, func() {
		libraryDir := test.TempDir()
		pluginFile := filepath.Join(libraryDir, "pjrt_c_api_gpu_plugin.so")
		err := os.WriteFile(pluginFile, []byte{}, 0o600)
		So(err, ShouldBeNil)

		config := PJRTConfig{
			Platform:    "gpu",
			LibraryDirs: []string{libraryDir},
		}

		Convey("It should discover the platform plugin", func() {
			So(config.ResolvedPluginFile(), ShouldEqual, pluginFile)
		})
	})

	Convey("Given a Darwin plugin on the library path", test, func() {
		libraryDir := test.TempDir()
		pluginFile := filepath.Join(libraryDir, "pjrt_c_api_cpu_plugin.dylib")
		err := os.WriteFile(pluginFile, []byte{}, 0o600)
		So(err, ShouldBeNil)

		config := PJRTConfig{
			Platform:    "cpu",
			LibraryDirs: []string{libraryDir},
		}

		Convey("It should discover the dylib plugin", func() {
			So(config.ResolvedPluginFile(), ShouldEqual, pluginFile)
		})
	})
}

func TestPJRTConfig_ValidateBuild(test *testing.T) {
	Convey("Given a valid PJRT include tree", test, func() {
		includeDir := test.TempDir()
		headerDir := filepath.Join(includeDir, "xla", "pjrt", "c")
		err := os.MkdirAll(headerDir, 0o755)
		So(err, ShouldBeNil)

		err = os.WriteFile(filepath.Join(headerDir, "pjrt_c_api.h"), []byte{}, 0o600)
		So(err, ShouldBeNil)

		config := PJRTConfig{IncludeDir: includeDir}

		Convey("It should accept the build environment", func() {
			So(config.ValidateBuild(), ShouldBeNil)
		})
	})

	Convey("Given a missing include directory", test, func() {
		config := PJRTConfig{}

		Convey("It should reject the build environment", func() {
			So(config.ValidateBuild(), ShouldNotBeNil)
		})
	})
}

func TestPJRTConfig_ValidateRuntime(test *testing.T) {
	Convey("Given a valid explicit plugin", test, func() {
		pluginFile := filepath.Join(test.TempDir(), "pjrt_c_api_cpu_plugin.so")
		err := os.WriteFile(pluginFile, []byte{}, 0o600)
		So(err, ShouldBeNil)

		config := PJRTConfig{
			Platform:   "cpu",
			PluginFile: pluginFile,
		}

		Convey("It should accept the runtime environment", func() {
			So(config.ValidateRuntime(), ShouldBeNil)
		})
	})

	Convey("Given a missing plugin", test, func() {
		config := PJRTConfig{Platform: "cpu"}

		Convey("It should reject the runtime environment", func() {
			So(config.ValidateRuntime(), ShouldNotBeNil)
		})
	})
}

func BenchmarkPJRTConfig_ValidateRuntime(benchmark *testing.B) {
	pluginFile := filepath.Join(benchmark.TempDir(), "pjrt_c_api_cpu_plugin.so")

	if err := os.WriteFile(pluginFile, []byte{}, 0o600); err != nil {
		benchmark.Fatal(err)
	}

	config := PJRTConfig{
		Platform:   "cpu",
		PluginFile: pluginFile,
	}

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		if err := config.ValidateRuntime(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkPJRTConfig_ResolvedPluginFile(benchmark *testing.B) {
	libraryDir := benchmark.TempDir()
	pluginFile := filepath.Join(libraryDir, "pjrt_c_api_cpu_plugin.so")

	if err := os.WriteFile(pluginFile, []byte{}, 0o600); err != nil {
		benchmark.Fatal(err)
	}

	config := PJRTConfig{
		Platform:    "cpu",
		LibraryDirs: []string{libraryDir},
	}

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		if resolved := config.ResolvedPluginFile(); resolved != pluginFile {
			benchmark.Fatalf("unexpected resolved path %q want %q", resolved, pluginFile)
		}
	}
}

func BenchmarkPJRTConfig_ValidateBuild(benchmark *testing.B) {
	includeDir := benchmark.TempDir()
	headerDir := filepath.Join(includeDir, "xla", "pjrt", "c")

	if err := os.MkdirAll(headerDir, 0o755); err != nil {
		benchmark.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(headerDir, "pjrt_c_api.h"), []byte{}, 0o600); err != nil {
		benchmark.Fatal(err)
	}

	config := PJRTConfig{IncludeDir: includeDir}

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		if err := config.ValidateBuild(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func clearPJRTEnv(test *testing.T) {
	for _, envName := range []string{
		"CARAMBA_XLA_INCLUDE_DIR",
		"XLA_INCLUDE_DIR",
		"XLA_INCLUDE",
		"CGO_CPPFLAGS",
		"CARAMBA_PJRT_CPU_PLUGIN",
		"CARAMBA_PJRT_GPU_PLUGIN",
		"CARAMBA_PJRT_PLUGIN",
		"PJRT_PLUGIN_PATH",
		"CARAMBA_PJRT_LIBRARY_DIR",
		"LD_LIBRARY_PATH",
		"DYLD_LIBRARY_PATH",
	} {
		test.Setenv(envName, "")
	}
}
