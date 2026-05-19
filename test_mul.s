#include "textflag.h"

TEXT ·TestMul(SB), NOSPLIT, $0
	MUL V1.S4, V2.S4, V3.S4
	RET
