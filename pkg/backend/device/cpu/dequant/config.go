package dequant

type Int8Config struct {
	Scale     float32
	ZeroPoint int8
}

type Int4Config struct {
	Scale     float32
	ZeroPoint int8
}

func DefaultInt8Config() Int8Config {
	return Int8Config{Scale: 1.0, ZeroPoint: 0}
}

func DefaultInt4Config() Int4Config {
	return Int4Config{Scale: 1.0, ZeroPoint: 0}
}
