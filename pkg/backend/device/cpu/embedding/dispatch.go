// Package embedding implements token embedding lookup and embedding-bag
// aggregation for float32, bfloat16, and float16 tables.
//
// Float32 lookup and bag row kernels follow the pick-at-init model via
// select_{amd64,other}.go on amd64 (AVX-512 when available).
package embedding
