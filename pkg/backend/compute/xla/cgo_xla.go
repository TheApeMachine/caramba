//go:build cgo && xla

package xla

// When building with -tags "cgo xla", define __XLA_BUILD__ so xla_sources.cpp
// includes the full PJRT translation units.

// #cgo CXXFLAGS: -D__XLA_BUILD__ -std=c++17
// #cgo LDFLAGS: -ldl -lstdc++
import "C"
