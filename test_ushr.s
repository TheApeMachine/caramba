#include "textflag.h"

TEXT ·TestUSHR(SB), NOSPLIT, $0
	VUSHR $8, V2.S4, V5.S4
	RET
