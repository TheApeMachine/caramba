package math

var (
	uvnan      = uint64(0x7FF8000000000001)
	uvnan32    = uint32(0x7FC00001)
	uvinf      = uint64(0x7FF0000000000000)
	uvinf32    = uint32(0x7F800000)
	uvneginf   = uint64(0xFFF0000000000000)
	uvneginf32 = uint32(0xFF800000)
	uvone      = uint64(0x3FF0000000000000)
)

const (
	mask     = 0x7FF
	shift    = 64 - 11 - 1
	bias     = 1023
	signMask = 1 << 63
	fracMask = 1<<shift - 1

	GeluTanhAlpha = 0.7978845608028654
	GeluTanhBeta  = 0.044715
)
