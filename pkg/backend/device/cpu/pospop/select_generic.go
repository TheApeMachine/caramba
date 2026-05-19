//go:build !amd64 && !arm64

package pospop

/*
generic variants only
*/
var Count8Funcs = []count8impl{{Count8Generic, "generic", true}}
var Count16Funcs = []count16impl{{Count16Generic, "generic", true}}
var Count32Funcs = []count32impl{{Count32Generic, "generic", true}}
var Count64Funcs = []count64impl{{Count64Generic, "generic", true}}
