package pospop

func Count8NEON(counts *[8]int, buf []uint8)
func Count16NEON(counts *[16]int, buf []uint16)
func Count32NEON(counts *[32]int, buf []uint32)
func Count64NEON(counts *[64]int, buf []uint64)

var Count8Funcs = []count8impl{
	{Count8NEON, "neon", true},
	{Count8Generic, "generic", true},
}

var Count16Funcs = []count16impl{
	{Count16NEON, "neon", true},
	{Count16Generic, "generic", true},
}

var Count32Funcs = []count32impl{
	{Count32NEON, "neon", true},
	{Count32Generic, "generic", true},
}

var Count64Funcs = []count64impl{
	{Count64NEON, "neon", true},
	{Count64Generic, "generic", true},
}
