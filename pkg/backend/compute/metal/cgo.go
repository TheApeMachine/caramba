//go:build darwin && cgo

package metal

// Package-wide CGo flags.
//
// -Ddouble_t=double works around CarbonCore fp.h including double_t before the
// libc headers that normally define it when clang is invoked with cgo's
// -fPIC/-pthread/-O2 flag bundle.

// #cgo CFLAGS: -x objective-c -std=gnu17 -mmacosx-version-min=14.0 -U__STRICT_ANSI__ -Ddouble_t=double
// #cgo LDFLAGS: -framework Metal -framework Foundation
import "C"
