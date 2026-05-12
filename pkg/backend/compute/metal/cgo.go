//go:build darwin && cgo

package metal

// #cgo CFLAGS: -x objective-c -std=gnu17 -mmacosx-version-min=14.0 -U__STRICT_ANSI__
// #cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders
import "C"
