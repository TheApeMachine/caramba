package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
)

func SwiGLUTensorsF32Generic(dst, gate, up *float32, count int) {
	destination := unsafe.Slice(dst, count)
	gateLane := unsafe.Slice(gate, count)
	upLane := unsafe.Slice(up, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSwiGLU32(gateLane[index], upLane[index])
	}
}

func LinGLUTensorsF32Generic(dst, gate, up *float32, count int) {
	destination := unsafe.Slice(dst, count)
	gateLane := unsafe.Slice(gate, count)
	upLane := unsafe.Slice(up, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastLinGLU32(gateLane[index], upLane[index])
	}
}

func ReGLUTensorsF32Generic(dst, gate, up *float32, count int) {
	destination := unsafe.Slice(dst, count)
	gateLane := unsafe.Slice(gate, count)
	upLane := unsafe.Slice(up, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastReGLU32(gateLane[index], upLane[index])
	}
}

func GLUTensorsF32Generic(dst, gate, up *float32, count int) {
	destination := unsafe.Slice(dst, count)
	gateLane := unsafe.Slice(gate, count)
	upLane := unsafe.Slice(up, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastGLU32(gateLane[index], upLane[index])
	}
}

func SiGLUTensorsF32Generic(dst, gate, up *float32, count int) {
	destination := unsafe.Slice(dst, count)
	gateLane := unsafe.Slice(gate, count)
	upLane := unsafe.Slice(up, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSiGLU32(gateLane[index], upLane[index])
	}
}

func SeGLUTensorsF32Generic(dst, gate, up *float32, count int) {
	destination := unsafe.Slice(dst, count)
	gateLane := unsafe.Slice(gate, count)
	upLane := unsafe.Slice(up, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSeGLU32(gateLane[index], upLane[index])
	}
}

func GeGLUTensorsF32Generic(dst, gate, up *float32, count int) {
	destination := unsafe.Slice(dst, count)
	gateLane := unsafe.Slice(gate, count)
	upLane := unsafe.Slice(up, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastGeGLU32(gateLane[index], upLane[index])
	}
}

func GeGLUTanhTensorsF32Generic(dst, gate, up *float32, count int) {
	destination := unsafe.Slice(dst, count)
	gateLane := unsafe.Slice(gate, count)
	upLane := unsafe.Slice(up, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastGeGLUTanh32(gateLane[index], upLane[index])
	}
}
