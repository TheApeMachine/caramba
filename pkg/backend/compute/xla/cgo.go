package xla

// Package-wide CGo flags for the XLA/PJRT backend.
//
// This file has no build tag so Go always recognises the package as
// cgo-enabled. Without the "xla" tag, xla_sources.cpp compiles as an empty
// translation unit (no __XLA_BUILD__ defined). With -tags "cgo xla" the full
// PJRT C++ sources are compiled.
//
// Required environment when building with -tags "cgo xla":
//   CGO_CPPFLAGS="-I/path/to/xla"
//   CGO_LDFLAGS="-ldl -lstdc++"

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ldl
import "C"
