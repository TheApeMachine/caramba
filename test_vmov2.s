#include "textflag.h"

TEXT ·TestVMov2(SB), NOSPLIT, $0
	VMOV R8, V2.S[0]
	VMOV R8, V2.S[1]
	VMOV R8, V2.S[2]
	VMOV R8, V2.S[3]
	RET
