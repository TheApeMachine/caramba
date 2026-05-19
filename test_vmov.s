#include "textflag.h"

TEXT ·TestVMov(SB), NOSPLIT, $0
	VMOV R8, V2.S[0]
	RET
