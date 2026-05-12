#include "metal_error.h"

const char *metal_error_string(int code) {
	switch (code) {
	case 0:
		return "ok";
	case -1:
		return "invalid_argument";
	case -2:
		return "not_initialized_or_graph_error";
	case -3:
		return "not_initialized_or_numeric_error";
	case -4:
		return "allocation_failure";
	case -5:
		return "numeric_or_graph_detail";
	default:
		return "unknown_error";
	}
}
