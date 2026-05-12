package xla

// Package-wide CGo flags for the XLA/PJRT backend.
//
// This file has no build tag so Go always recognises the package as
// cgo-enabled. Without the "xla" tag, xla_sources.cpp compiles as an empty
// translation unit (no __XLA_BUILD__ defined). With -tags "cgo xla" the full
// PJRT C++ sources are compiled.
//
// PJRT runtime paths are loaded from compute.xla in cmd/asset/config.yml.

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ldl
import "C"
