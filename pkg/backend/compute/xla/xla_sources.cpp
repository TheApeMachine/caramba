// xla_sources.cpp — amalgamation unit for all XLA/PJRT C++ translation units.
//
// The _*_xla.cpp files have an underscore prefix so the Go tool ignores them
// by default. This file (no underscore) is always compiled by cgo, but only
// includes the translation units when __XLA_BUILD__ is defined — which is set
// by cgo.go only when -tags xla is active.
//
// Without __XLA_BUILD__ this file is empty and adds zero overhead.

#ifdef __XLA_BUILD__

// PJRT client/API/static helpers live in _activation_xla.cpp (included first).
#include "_activation_xla.cpp"
#include "_tensor_xla.cpp"
#include "_shape_xla.cpp"
#include "_math_xla.cpp"
#include "_attention_xla.cpp"
#include "_convolution_xla.cpp"
#include "_embedding_xla.cpp"
#include "_masking_xla.cpp"
#include "_pooling_xla.cpp"
#include "_positional_xla.cpp"
#include "_projection_xla.cpp"
#include "_reference_xla_ops.cpp"

#endif // __XLA_BUILD__
