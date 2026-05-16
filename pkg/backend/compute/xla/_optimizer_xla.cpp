#include "optimizer.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

/*
Optimizer entry points compile optimizer steps to StableHLO and execute them
through PJRT. Host slices are the ABI boundary used by state.Dict; the optimizer
math itself is expressed in XLA IR so the selected PJRT plugin owns execution.
*/

struct OptimizerInput {
	const double* values;
	int count;
	const char* name;
};

struct OptimizerOutput {
	double* values;
	int count;
	const char* name;
};

static std::unordered_map<std::string, PJRT_LoadedExecutable*> g_optimizer_execs;
static std::mutex g_optimizer_execs_mutex;

static std::string optimizer_type(int count) {
	return "tensor<" + std::to_string(count) + "xf64>";
}

static std::string optimizer_bool_type(int count) {
	return "tensor<" + std::to_string(count) + "xi1>";
}

static std::string optimizer_scalar(double value) {
	char buffer[96];
	snprintf(buffer, sizeof(buffer), "%.17g", value);

	return std::string(buffer);
}

static std::string optimizer_constant(const char* name, double value, const std::string& type) {
	return
		"    %" + std::string(name) + " = stablehlo.constant dense<" +
		optimizer_scalar(value) + "> : " + type + "\n";
}

static std::string optimizer_scalar_constant(const char* name, double value) {
	return optimizer_constant(name, value, "tensor<f64>");
}

static std::string optimizer_broadcast(
	const char* name, const char* scalarName, const std::string& type
) {
	return
		"    %" + std::string(name) + " = stablehlo.broadcast_in_dim %" +
		scalarName + ", dims = [] : (tensor<f64>) -> " + type + "\n";
}

static std::string optimizer_broadcast_bool(
	const char* name, const char* scalarName, const std::string& type
) {
	return
		"    %" + std::string(name) + " = stablehlo.broadcast_in_dim %" +
		scalarName + ", dims = [] : (tensor<i1>) -> " + type + "\n";
}

static std::string optimizer_reduce_sum(
	const char* name, const char* value, const std::string& type
) {
	std::string zero_name = std::string(name) + "_zero";

	return
		"    %" + zero_name + " = stablehlo.constant dense<0.0> : tensor<f64>\n"
		"    %" + std::string(name) + " = stablehlo.reduce(%" + value +
		" init: %" + zero_name + ") applies stablehlo.add across dimensions = [0] : (" +
		type + ", tensor<f64>) -> tensor<f64>\n";
}

static PJRT_LoadedExecutable* optimizer_compile(
	const std::string& key, const std::string& mlir
) {
	std::lock_guard<std::mutex> guard(g_optimizer_execs_mutex);

	auto existing = g_optimizer_execs.find(key);

	if (existing != g_optimizer_execs.end()) {
		return existing->second;
	}

	PJRT_LoadedExecutable* executable = compile_stablehlo(mlir);

	if (executable) {
		g_optimizer_execs[key] = executable;
	}

	return executable;
}

static void optimizer_destroy_buffer(PJRT_Buffer* buffer) {
	if (!g_api || !buffer) return;

	PJRT_Buffer_Destroy_Args args{};
	args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
	args.buffer = buffer;
	g_api->PJRT_Buffer_Destroy(&args);
}

static PJRT_Buffer* optimizer_upload(const OptimizerInput& input) {
	if (input.count < 0) return nullptr;
	if (input.count > 0 && !input.values) return nullptr;

	int64_t dims[1] = { (int64_t)input.count };

	PJRT_Client_BufferFromHostBuffer_Args args{};
	args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
	args.client = g_client;
	args.data = input.count == 0 ? nullptr : const_cast<double*>(input.values);
	args.type = PJRT_Buffer_Type_F64;
	args.dims = dims;
	args.num_dims = 1;
	args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
	args.device = g_device;
	args.memory = g_memory;

	PJRT_Error* err = g_api->PJRT_Client_BufferFromHostBuffer(&args);

	if (!check(g_api, err, input.name)) {
		return nullptr;
	}

	if (!await_and_destroy_event(args.done_with_host_buffer, input.name)) {
		optimizer_destroy_buffer(args.buffer);
		return nullptr;
	}

	PJRT_Buffer_ReadyEvent_Args ready_args{};
	ready_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
	ready_args.buffer = args.buffer;
	err = g_api->PJRT_Buffer_ReadyEvent(&ready_args);

	if (!check(g_api, err, input.name)) {
		optimizer_destroy_buffer(args.buffer);
		return nullptr;
	}

	if (!await_and_destroy_event(ready_args.event, input.name)) {
		optimizer_destroy_buffer(args.buffer);
		return nullptr;
	}

	return args.buffer;
}

static int optimizer_download(PJRT_Buffer* buffer, const OptimizerOutput& output) {
	if (!buffer || output.count < 0) return -1;
	if (output.count > 0 && !output.values) return -1;

	PJRT_Buffer_ToHostBuffer_Args args{};
	args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
	args.src = buffer;
	args.dst = output.count == 0 ? nullptr : output.values;
	args.dst_size = (size_t)output.count * sizeof(double);

	PJRT_Error* err = g_api->PJRT_Buffer_ToHostBuffer(&args);

	if (!check(g_api, err, output.name)) {
		return -1;
	}

	return await_and_destroy_event(args.event, output.name) ? 0 : -1;
}

static int optimizer_execute(
	const std::string& key,
	const std::string& mlir,
	const std::vector<OptimizerInput>& inputs,
	const std::vector<OptimizerOutput>& outputs
) {
	if (!g_client || inputs.empty() || outputs.empty()) return -1;
	for (const OptimizerInput& input : inputs) {
		if (input.count < 0) return -1;
	}

	for (const OptimizerOutput& output : outputs) {
		if (output.count < 0) return -1;
	}

	PJRT_LoadedExecutable* executable = optimizer_compile(key, mlir);

	if (!executable) return -1;

	std::vector<PJRT_Buffer*> input_buffers(inputs.size(), nullptr);

	for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
		input_buffers[input_index] = optimizer_upload(inputs[input_index]);

		if (!input_buffers[input_index]) {
			for (PJRT_Buffer* buffer : input_buffers) optimizer_destroy_buffer(buffer);
			return -1;
		}
	}

	PJRT_Buffer** argument_lists[1] = { input_buffers.data() };
	std::vector<PJRT_Buffer*> output_buffers(outputs.size(), nullptr);
	PJRT_Buffer** output_lists[1] = { output_buffers.data() };

	PJRT_LoadedExecutable_Execute_Args args{};
	args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
	args.executable = executable;
	args.argument_lists = argument_lists;
	args.num_devices = 1;
	args.num_args = input_buffers.size();
	args.output_lists = output_lists;
	PJRT_ExecuteOptions options = single_device_execute_options();
	args.options = &options;

	PJRT_Error* err = g_api->PJRT_LoadedExecutable_Execute(&args);

	if (!check(g_api, err, "PJRT_LoadedExecutable_Execute(optimizer)")) {
		for (PJRT_Buffer* buffer : input_buffers) optimizer_destroy_buffer(buffer);
		for (PJRT_Buffer* buffer : output_buffers) optimizer_destroy_buffer(buffer);
		return -1;
	}

	int rc = 0;

	for (size_t output_index = 0; output_index < outputs.size(); output_index++) {
		if (optimizer_download(output_buffers[output_index], outputs[output_index]) != 0) {
			rc = -1;
		}
	}

	for (PJRT_Buffer* buffer : input_buffers) optimizer_destroy_buffer(buffer);
	for (PJRT_Buffer* buffer : output_buffers) optimizer_destroy_buffer(buffer);

	return rc;
}

static int optimizer_execute_same_size(
	const std::string& key,
	const std::string& mlir,
	const std::vector<OptimizerInput>& inputs,
	const std::vector<OptimizerOutput>& outputs
) {
	if (!g_client || inputs.empty() || outputs.empty()) return -1;

	int count = inputs[0].count;

	if (count == 0) return 0;

	for (const OptimizerInput& input : inputs) {
		if (input.count != count) return -1;
	}

	for (const OptimizerOutput& output : outputs) {
		if (output.count != count) return -1;
	}

	return optimizer_execute(key, mlir, inputs, outputs);
}

static std::string optimizer_module(
	const char* name,
	const std::vector<const char*>& args,
	const std::vector<const char*>& returns,
	int count,
	const std::string& body
) {
	std::string type = optimizer_type(count);
	std::string text = "module @" + std::string(name) + " {\n  func.func @main(";

	for (size_t arg_index = 0; arg_index < args.size(); arg_index++) {
		if (arg_index > 0) text += ", ";
		text += "%" + std::string(args[arg_index]) + ": " + type;
	}

	text += ") -> ";

	if (returns.size() == 1) {
		text += type;
	} else {
		text += "(";
		for (size_t return_index = 0; return_index < returns.size(); return_index++) {
			if (return_index > 0) text += ", ";
			text += type;
		}
		text += ")";
	}

	text += " {\n" + body + "    return ";

	for (size_t return_index = 0; return_index < returns.size(); return_index++) {
		if (return_index > 0) text += ", ";
		text += "%" + std::string(returns[return_index]);
	}

	text += " : ";

	for (size_t return_index = 0; return_index < returns.size(); return_index++) {
		if (return_index > 0) text += ", ";
		text += type;
	}

	text += "\n  }\n}\n";

	return text;
}

static std::string build_adam(
	int count, double beta1, double beta2, double learning_rate, double eps
) {
	std::string type = optimizer_type(count);
	std::string body;

	body += optimizer_constant("beta1", beta1, type);
	body += optimizer_constant("omb1", 1.0 - beta1, type);
	body += optimizer_constant("beta2", beta2, type);
	body += optimizer_constant("omb2", 1.0 - beta2, type);
	body += optimizer_constant("lr", learning_rate, type);
	body += optimizer_constant("eps", eps, type);
	body +=
		"    %m0 = stablehlo.multiply %moment, %beta1 : " + type + "\n"
		"    %m1 = stablehlo.multiply %grads, %omb1 : " + type + "\n"
		"    %next_moment = stablehlo.add %m0, %m1 : " + type + "\n"
		"    %grad_sq = stablehlo.multiply %grads, %grads : " + type + "\n"
		"    %v0 = stablehlo.multiply %variance, %beta2 : " + type + "\n"
		"    %v1 = stablehlo.multiply %grad_sq, %omb2 : " + type + "\n"
		"    %next_variance = stablehlo.add %v0, %v1 : " + type + "\n"
		"    %sqrt_v = stablehlo.sqrt %next_variance : " + type + "\n"
		"    %denom = stablehlo.add %sqrt_v, %eps : " + type + "\n"
		"    %step_ratio = stablehlo.divide %next_moment, %denom : " + type + "\n"
		"    %step = stablehlo.multiply %step_ratio, %lr : " + type + "\n"
		"    %out = stablehlo.subtract %params, %step : " + type + "\n";

	return optimizer_module(
		"xla_optimizer_adam", {"params", "grads", "moment", "variance"},
		{"out", "next_moment", "next_variance"}, count, body,
	);
}

static std::string build_adamw(
	int count, double beta1, double beta2, double learning_rate, double eps,
	double weight_decay_step
) {
	std::string type = optimizer_type(count);
	std::string body;

	body += optimizer_constant("wd_step", weight_decay_step, type);
	body +=
		"    %decay = stablehlo.multiply %params, %wd_step : " + type + "\n";
	body += optimizer_constant("beta1", beta1, type);
	body += optimizer_constant("omb1", 1.0 - beta1, type);
	body += optimizer_constant("beta2", beta2, type);
	body += optimizer_constant("omb2", 1.0 - beta2, type);
	body += optimizer_constant("lr", learning_rate, type);
	body += optimizer_constant("eps", eps, type);
	body +=
		"    %m0 = stablehlo.multiply %moment, %beta1 : " + type + "\n"
		"    %m1 = stablehlo.multiply %grads, %omb1 : " + type + "\n"
		"    %next_moment = stablehlo.add %m0, %m1 : " + type + "\n"
		"    %grad_sq = stablehlo.multiply %grads, %grads : " + type + "\n"
		"    %v0 = stablehlo.multiply %variance, %beta2 : " + type + "\n"
		"    %v1 = stablehlo.multiply %grad_sq, %omb2 : " + type + "\n"
		"    %next_variance = stablehlo.add %v0, %v1 : " + type + "\n"
		"    %sqrt_v = stablehlo.sqrt %next_variance : " + type + "\n"
		"    %denom = stablehlo.add %sqrt_v, %eps : " + type + "\n"
		"    %step_ratio = stablehlo.divide %next_moment, %denom : " + type + "\n"
		"    %adaptive = stablehlo.multiply %step_ratio, %lr : " + type + "\n"
		"    %after_decay = stablehlo.subtract %params, %decay : " + type + "\n"
		"    %out = stablehlo.subtract %after_decay, %adaptive : " + type + "\n";

	return optimizer_module(
		"xla_optimizer_adamw", {"params", "grads", "moment", "variance"},
		{"out", "next_moment", "next_variance"}, count, body,
	);
}

static std::string build_adamax(
	int count, double beta1, double beta2, double learning_rate, double eps
) {
	std::string type = optimizer_type(count);
	std::string body;

	body += optimizer_constant("beta1", beta1, type);
	body += optimizer_constant("omb1", 1.0 - beta1, type);
	body += optimizer_constant("beta2", beta2, type);
	body += optimizer_constant("lr", learning_rate, type);
	body += optimizer_constant("eps", eps, type);
	body +=
		"    %m0 = stablehlo.multiply %moment, %beta1 : " + type + "\n"
		"    %m1 = stablehlo.multiply %grads, %omb1 : " + type + "\n"
		"    %next_moment = stablehlo.add %m0, %m1 : " + type + "\n"
		"    %scaled_norm = stablehlo.multiply %infinity_norm, %beta2 : " + type + "\n"
		"    %abs_grad = stablehlo.abs %grads : " + type + "\n"
		"    %next_norm = stablehlo.maximum %scaled_norm, %abs_grad : " + type + "\n"
		"    %denom = stablehlo.add %next_norm, %eps : " + type + "\n"
		"    %step_ratio = stablehlo.divide %next_moment, %denom : " + type + "\n"
		"    %step = stablehlo.multiply %step_ratio, %lr : " + type + "\n"
		"    %out = stablehlo.subtract %params, %step : " + type + "\n";

	return optimizer_module(
		"xla_optimizer_adamax", {"params", "grads", "moment", "infinity_norm"},
		{"out", "next_moment", "next_norm"}, count, body,
	);
}

static std::string build_sgd(
	int count, double learning_rate, double weight_decay, double momentum, int nesterov
) {
	std::string type = optimizer_type(count);
	std::string body;

	body += optimizer_constant("lr", learning_rate, type);
	body += optimizer_constant("wd", weight_decay, type);
	body += optimizer_constant("momentum", momentum, type);
	body +=
		"    %decay = stablehlo.multiply %params, %wd : " + type + "\n"
		"    %grad = stablehlo.add %grads, %decay : " + type + "\n";

	if (momentum == 0.0) {
		body += optimizer_constant("zero", 0.0, type);
		body +=
			"    %step = stablehlo.multiply %grad, %lr : " + type + "\n"
			"    %out = stablehlo.subtract %params, %step : " + type + "\n"
			"    %next_velocity = stablehlo.add %velocity, %zero : " + type + "\n";

		return optimizer_module(
			"xla_optimizer_sgd", {"params", "grads", "velocity"},
			{"out", "next_velocity"}, count, body,
		);
	}

	body +=
		"    %scaled_velocity = stablehlo.multiply %velocity, %momentum : " + type + "\n"
		"    %next_velocity = stablehlo.add %scaled_velocity, %grad : " + type + "\n";

	if (nesterov != 0) {
		body +=
			"    %nesterov_delta = stablehlo.multiply %next_velocity, %momentum : " + type + "\n"
			"    %update = stablehlo.add %grad, %nesterov_delta : " + type + "\n";
	} else {
		body += optimizer_constant("zero", 0.0, type);
		body += "    %update = stablehlo.add %next_velocity, %zero : " + type + "\n";
	}

	body +=
		"    %step = stablehlo.multiply %update, %lr : " + type + "\n"
		"    %out = stablehlo.subtract %params, %step : " + type + "\n";

	return optimizer_module(
		"xla_optimizer_sgd", {"params", "grads", "velocity"},
		{"out", "next_velocity"}, count, body,
	);
}

static std::string build_lion(
	int count, double learning_rate, double beta1, double beta2, double weight_decay
) {
	std::string type = optimizer_type(count);
	std::string bool_type = optimizer_bool_type(count);
	std::string body;

	body += optimizer_constant("lr", learning_rate, type);
	body += optimizer_constant("beta1", beta1, type);
	body += optimizer_constant("omb1", 1.0 - beta1, type);
	body += optimizer_constant("beta2", beta2, type);
	body += optimizer_constant("omb2", 1.0 - beta2, type);
	body += optimizer_constant("wd", weight_decay, type);
	body += optimizer_constant("zero", 0.0, type);
	body += optimizer_constant("one", 1.0, type);
	body += optimizer_constant("neg_one", -1.0, type);
	body +=
		"    %blend_m = stablehlo.multiply %moment, %beta1 : " + type + "\n"
		"    %blend_g = stablehlo.multiply %grads, %omb1 : " + type + "\n"
		"    %blended = stablehlo.add %blend_m, %blend_g : " + type + "\n"
		"    %positive = stablehlo.compare GT, %blended, %zero, TOTALORDER : (" +
			type + ", " + type + ") -> " + bool_type + "\n"
		"    %negative = stablehlo.compare LT, %blended, %zero, TOTALORDER : (" +
			type + ", " + type + ") -> " + bool_type + "\n"
		"    %positive_sign = stablehlo.select %positive, %one, %zero : (" +
			bool_type + ", " + type + ", " + type + ") -> " + type + "\n"
		"    %sign = stablehlo.select %negative, %neg_one, %positive_sign : (" +
			bool_type + ", " + type + ", " + type + ") -> " + type + "\n"
		"    %decay = stablehlo.multiply %params, %wd : " + type + "\n"
		"    %direction = stablehlo.add %sign, %decay : " + type + "\n"
		"    %step = stablehlo.multiply %direction, %lr : " + type + "\n"
		"    %out = stablehlo.subtract %params, %step : " + type + "\n"
		"    %m0 = stablehlo.multiply %moment, %beta2 : " + type + "\n"
		"    %m1 = stablehlo.multiply %grads, %omb2 : " + type + "\n"
		"    %next_moment = stablehlo.add %m0, %m1 : " + type + "\n";

	return optimizer_module(
		"xla_optimizer_lion", {"params", "grads", "moment"},
		{"out", "next_moment"}, count, body,
	);
}

static std::string build_rmsprop(
	int count, double learning_rate, double alpha, double eps,
	double momentum, double weight_decay, int centered
) {
	std::string type = optimizer_type(count);
	std::string body;

	body += optimizer_constant("lr", learning_rate, type);
	body += optimizer_constant("alpha", alpha, type);
	body += optimizer_constant("one_minus_alpha", 1.0 - alpha, type);
	body += optimizer_constant("eps", eps, type);
	body += optimizer_constant("momentum", momentum, type);
	body += optimizer_constant("wd", weight_decay, type);
	body +=
		"    %decay = stablehlo.multiply %params, %wd : " + type + "\n"
		"    %grad = stablehlo.add %grads, %decay : " + type + "\n"
		"    %grad_sq = stablehlo.multiply %grad, %grad : " + type + "\n"
		"    %square0 = stablehlo.multiply %square_average, %alpha : " + type + "\n"
		"    %square1 = stablehlo.multiply %grad_sq, %one_minus_alpha : " + type + "\n"
		"    %next_square = stablehlo.add %square0, %square1 : " + type + "\n";

	if (centered != 0) {
		body +=
			"    %ga0 = stablehlo.multiply %grad_average, %alpha : " + type + "\n"
			"    %ga1 = stablehlo.multiply %grad, %one_minus_alpha : " + type + "\n"
			"    %next_grad_average = stablehlo.add %ga0, %ga1 : " + type + "\n"
			"    %ga_sq = stablehlo.multiply %next_grad_average, %next_grad_average : " + type + "\n"
			"    %average = stablehlo.subtract %next_square, %ga_sq : " + type + "\n";
	} else {
		body += optimizer_constant("zero", 0.0, type);
		body +=
			"    %next_grad_average = stablehlo.add %grad_average, %zero : " + type + "\n"
			"    %average = stablehlo.add %next_square, %zero : " + type + "\n";
	}

	body +=
		"    %sqrt_average = stablehlo.sqrt %average : " + type + "\n"
		"    %denom = stablehlo.add %sqrt_average, %eps : " + type + "\n"
		"    %base_update = stablehlo.divide %grad, %denom : " + type + "\n";

	if (momentum != 0.0) {
		body += optimizer_constant("zero2", 0.0, type);
		body +=
			"    %buf0 = stablehlo.multiply %momentum_buffer, %momentum : " + type + "\n"
			"    %next_buffer = stablehlo.add %buf0, %base_update : " + type + "\n"
			"    %update = stablehlo.add %next_buffer, %zero2 : " + type + "\n";
	} else {
		body += optimizer_constant("zero2", 0.0, type);
		body +=
			"    %next_buffer = stablehlo.add %momentum_buffer, %zero2 : " + type + "\n"
			"    %update = stablehlo.add %base_update, %zero2 : " + type + "\n";
	}

	body +=
		"    %step = stablehlo.multiply %update, %lr : " + type + "\n"
		"    %out = stablehlo.subtract %params, %step : " + type + "\n";

	return optimizer_module(
		"xla_optimizer_rmsprop",
		{"params", "grads", "square_average", "momentum_buffer", "grad_average"},
		{"out", "next_square", "next_buffer", "next_grad_average"}, count, body,
	);
}

static std::string build_hebbian(int count, double learning_rate, double max_norm) {
	std::string type = optimizer_type(count);
	std::string bool_type = optimizer_bool_type(count);
	std::string body;

	body += optimizer_constant("lr", learning_rate, type);
	body +=
		"    %step = stablehlo.multiply %grads, %lr : " + type + "\n"
		"    %candidate = stablehlo.add %params, %step : " + type + "\n";

	if (max_norm <= 0.0) {
		body += optimizer_constant("zero", 0.0, type);
		body += "    %out = stablehlo.add %candidate, %zero : " + type + "\n";

		return optimizer_module(
			"xla_optimizer_hebbian", {"params", "grads"}, {"out"}, count, body,
		);
	}

	body += optimizer_scalar_constant("max_norm", max_norm);
	body += optimizer_scalar_constant("zero_scalar", 0.0);
	body +=
		"    %candidate_sq = stablehlo.multiply %candidate, %candidate : " + type + "\n";
	body += optimizer_reduce_sum("norm_sq", "candidate_sq", type);
	body +=
		"    %norm = stablehlo.sqrt %norm_sq : tensor<f64>\n"
		"    %scale = stablehlo.divide %max_norm, %norm : tensor<f64>\n"
		"    %scaled_scale = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f64>) -> " + type + "\n"
		"    %scaled = stablehlo.multiply %candidate, %scaled_scale : " + type + "\n"
		"    %gt_norm = stablehlo.compare GT, %norm, %max_norm, TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>\n"
		"    %gt_zero = stablehlo.compare GT, %max_norm, %zero_scalar, TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>\n"
		"    %need_scale = stablehlo.and %gt_norm, %gt_zero : tensor<i1>\n";
	body += optimizer_broadcast_bool("need_scale_v", "need_scale", bool_type);
	body +=
		"    %out = stablehlo.select %need_scale_v, %scaled, %candidate : (" +
			bool_type + ", " + type + ", " + type + ") -> " + type + "\n";

	return optimizer_module(
		"xla_optimizer_hebbian", {"params", "grads"}, {"out"}, count, body,
	);
}

static std::string build_lars(
	int count, double learning_rate, double eta, double momentum,
	double weight_decay, double eps
) {
	std::string type = optimizer_type(count);
	std::string bool_type = optimizer_bool_type(count);
	std::string body;

	body += optimizer_scalar_constant("zero_scalar", 0.0);
	body += optimizer_scalar_constant("lr_scalar", learning_rate);
	body += optimizer_scalar_constant("eta_scalar", eta);
	body += optimizer_scalar_constant("wd_scalar", weight_decay);
	body += optimizer_scalar_constant("eps_scalar", eps);
	body += optimizer_constant("wd", weight_decay, type);
	body += optimizer_constant("momentum", momentum, type);
	body +=
		"    %param_sq = stablehlo.multiply %params, %params : " + type + "\n"
		"    %grad_sq = stablehlo.multiply %grads, %grads : " + type + "\n";
	body += optimizer_reduce_sum("param_norm_sq", "param_sq", type);
	body +=
		"    %param_norm = stablehlo.sqrt %param_norm_sq : tensor<f64>\n";
	body += optimizer_reduce_sum("grad_norm_sq", "grad_sq", type);
	body +=
		"    %grad_norm = stablehlo.sqrt %grad_norm_sq : tensor<f64>\n"
		"    %eta_param = stablehlo.multiply %eta_scalar, %param_norm : tensor<f64>\n"
		"    %wd_param = stablehlo.multiply %wd_scalar, %param_norm : tensor<f64>\n"
		"    %denom0 = stablehlo.add %grad_norm, %wd_param : tensor<f64>\n"
		"    %denom = stablehlo.add %denom0, %eps_scalar : tensor<f64>\n"
		"    %trust = stablehlo.divide %eta_param, %denom : tensor<f64>\n"
		"    %param_positive = stablehlo.compare GT, %param_norm, %zero_scalar, TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>\n"
		"    %grad_positive = stablehlo.compare GT, %grad_norm, %zero_scalar, TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>\n"
		"    %valid = stablehlo.and %param_positive, %grad_positive : tensor<i1>\n"
		"    %local_lr_s = stablehlo.select %valid, %trust, %lr_scalar : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>\n";
	body += optimizer_broadcast("local_lr", "local_lr_s", type);
	body +=
		"    %decay = stablehlo.multiply %params, %wd : " + type + "\n"
		"    %grad = stablehlo.add %grads, %decay : " + type + "\n"
		"    %scaled_grad = stablehlo.multiply %grad, %local_lr : " + type + "\n"
		"    %velocity0 = stablehlo.multiply %velocity, %momentum : " + type + "\n"
		"    %next_velocity = stablehlo.add %velocity0, %scaled_grad : " + type + "\n"
		"    %out = stablehlo.subtract %params, %next_velocity : " + type + "\n";
	(void)bool_type;

	return optimizer_module(
		"xla_optimizer_lars", {"params", "grads", "velocity"},
		{"out", "next_velocity"}, count, body,
	);
}

static std::string build_lamb(
	int count, double learning_rate, double beta1, double beta2, double eps,
	double weight_decay, double bias_correction1_inv, double bias_correction2_inv
) {
	std::string type = optimizer_type(count);
	std::string body;

	body += optimizer_scalar_constant("zero_scalar", 0.0);
	body += optimizer_scalar_constant("lr_scalar", learning_rate);
	body += optimizer_constant("beta1", beta1, type);
	body += optimizer_constant("omb1", 1.0 - beta1, type);
	body += optimizer_constant("beta2", beta2, type);
	body += optimizer_constant("omb2", 1.0 - beta2, type);
	body += optimizer_constant("eps", eps, type);
	body += optimizer_constant("wd", weight_decay, type);
	body += optimizer_constant("bc1_inv", bias_correction1_inv, type);
	body += optimizer_constant("bc2_inv", bias_correction2_inv, type);
	body +=
		"    %m0 = stablehlo.multiply %moment, %beta1 : " + type + "\n"
		"    %m1 = stablehlo.multiply %grads, %omb1 : " + type + "\n"
		"    %next_moment = stablehlo.add %m0, %m1 : " + type + "\n"
		"    %grad_sq = stablehlo.multiply %grads, %grads : " + type + "\n"
		"    %v0 = stablehlo.multiply %variance, %beta2 : " + type + "\n"
		"    %v1 = stablehlo.multiply %grad_sq, %omb2 : " + type + "\n"
		"    %next_variance = stablehlo.add %v0, %v1 : " + type + "\n"
		"    %m_hat = stablehlo.multiply %next_moment, %bc1_inv : " + type + "\n"
		"    %v_hat = stablehlo.multiply %next_variance, %bc2_inv : " + type + "\n"
		"    %sqrt_v = stablehlo.sqrt %v_hat : " + type + "\n"
		"    %denom = stablehlo.add %sqrt_v, %eps : " + type + "\n"
		"    %adaptive = stablehlo.divide %m_hat, %denom : " + type + "\n"
		"    %decay = stablehlo.multiply %params, %wd : " + type + "\n"
		"    %update = stablehlo.add %adaptive, %decay : " + type + "\n"
		"    %param_sq = stablehlo.multiply %params, %params : " + type + "\n"
		"    %update_sq = stablehlo.multiply %update, %update : " + type + "\n";
	body += optimizer_reduce_sum("param_norm_sq", "param_sq", type);
	body += optimizer_reduce_sum("update_norm_sq", "update_sq", type);
	body +=
		"    %param_norm = stablehlo.sqrt %param_norm_sq : tensor<f64>\n"
		"    %update_norm = stablehlo.sqrt %update_norm_sq : tensor<f64>\n"
		"    %trust = stablehlo.divide %param_norm, %update_norm : tensor<f64>\n"
		"    %scaled_trust = stablehlo.multiply %lr_scalar, %trust : tensor<f64>\n"
		"    %param_positive = stablehlo.compare GT, %param_norm_sq, %zero_scalar, TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>\n"
		"    %update_positive = stablehlo.compare GT, %update_norm_sq, %zero_scalar, TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>\n"
		"    %valid = stablehlo.and %param_positive, %update_positive : tensor<i1>\n"
		"    %ratio_s = stablehlo.select %valid, %scaled_trust, %lr_scalar : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>\n";
	body += optimizer_broadcast("ratio", "ratio_s", type);
	body +=
		"    %step = stablehlo.multiply %update, %ratio : " + type + "\n"
		"    %out = stablehlo.subtract %params, %step : " + type + "\n";

	return optimizer_module(
		"xla_optimizer_lamb", {"params", "grads", "moment", "variance"},
		{"out", "next_moment", "next_variance"}, count, body,
	);
}

static std::string build_adagrad(
	int count, double learning_rate, double eps, double weight_decay
) {
	std::string type = optimizer_type(count);
	std::string body;

	body += optimizer_constant("lr", learning_rate, type);
	body += optimizer_constant("eps", eps, type);
	body += optimizer_constant("wd", weight_decay, type);
	body +=
		"    %decay = stablehlo.multiply %params, %wd : " + type + "\n"
		"    %grad = stablehlo.add %grads, %decay : " + type + "\n"
		"    %grad_sq = stablehlo.multiply %grad, %grad : " + type + "\n"
		"    %next_accumulator = stablehlo.add %accumulator, %grad_sq : " + type + "\n"
		"    %sqrt_accumulator = stablehlo.sqrt %next_accumulator : " + type + "\n"
		"    %denom = stablehlo.add %sqrt_accumulator, %eps : " + type + "\n"
		"    %ratio = stablehlo.divide %grad, %denom : " + type + "\n"
		"    %step = stablehlo.multiply %ratio, %lr : " + type + "\n"
		"    %out = stablehlo.subtract %params, %step : " + type + "\n";

	return optimizer_module(
		"xla_optimizer_adagrad", {"params", "grads", "accumulator"},
		{"out", "next_accumulator"}, count, body,
	);
}

static std::string build_adadelta(
	int count, double rho, double eps, double weight_decay
) {
	std::string type = optimizer_type(count);
	std::string body;

	body += optimizer_constant("rho", rho, type);
	body += optimizer_constant("one_minus_rho", 1.0 - rho, type);
	body += optimizer_constant("eps", eps, type);
	body += optimizer_constant("wd", weight_decay, type);
	body +=
		"    %decay = stablehlo.multiply %params, %wd : " + type + "\n"
		"    %grad = stablehlo.add %grads, %decay : " + type + "\n"
		"    %grad_sq = stablehlo.multiply %grad, %grad : " + type + "\n"
		"    %ga0 = stablehlo.multiply %grad_average, %rho : " + type + "\n"
		"    %ga1 = stablehlo.multiply %grad_sq, %one_minus_rho : " + type + "\n"
		"    %next_grad_average = stablehlo.add %ga0, %ga1 : " + type + "\n"
		"    %delta_eps = stablehlo.add %delta_average, %eps : " + type + "\n"
		"    %grad_eps = stablehlo.add %next_grad_average, %eps : " + type + "\n"
		"    %delta_root = stablehlo.sqrt %delta_eps : " + type + "\n"
		"    %grad_root = stablehlo.sqrt %grad_eps : " + type + "\n"
		"    %scale = stablehlo.divide %delta_root, %grad_root : " + type + "\n"
		"    %update = stablehlo.multiply %scale, %grad : " + type + "\n"
		"    %out = stablehlo.subtract %params, %update : " + type + "\n"
		"    %update_sq = stablehlo.multiply %update, %update : " + type + "\n"
		"    %da0 = stablehlo.multiply %delta_average, %rho : " + type + "\n"
		"    %da1 = stablehlo.multiply %update_sq, %one_minus_rho : " + type + "\n"
		"    %next_delta_average = stablehlo.add %da0, %da1 : " + type + "\n";

	return optimizer_module(
		"xla_optimizer_adadelta", {"params", "grads", "grad_average", "delta_average"},
		{"out", "next_grad_average", "next_delta_average"}, count, body,
	);
}

static std::vector<OptimizerInput> optimizer_inputs(
	const double* params, const double* grads, int count
) {
	return {
		{params, count, "xla optimizer upload params"},
		{grads, count, "xla optimizer upload grads"},
	};
}

static std::string optimizer_key(const char* name, int count, const std::vector<double>& values) {
	std::string key = std::string(name) + ":" + std::to_string(count);

	for (double value : values) {
		key += ":" + optimizer_scalar(value);
	}

	return key;
}

static std::string lbfgs_history_type(int history_size, int count) {
	return optimizer_type(history_size * count);
}

static std::string lbfgs_rho_type(int history_size) {
	return optimizer_type(history_size);
}

static std::string lbfgs_slice(
	const char* name,
	const char* source,
	int start,
	int end,
	const std::string& source_type,
	const std::string& output_type
) {
	return
		"    %" + std::string(name) + " = stablehlo.slice %" + source +
		" [" + std::to_string(start) + ":" + std::to_string(end) + "] : (" +
		source_type + ") -> " + output_type + "\n";
}

static std::string lbfgs_replace_segment(
	std::string& body,
	const std::string& prefix,
	const char* source,
	const char* update,
	int total,
	int start,
	int length,
	const std::string& source_type,
	const std::string& update_type
) {
	if (total == length) {
		return std::string(update);
	}

	std::vector<std::pair<std::string, std::string>> segments;

	if (start > 0) {
		std::string name = prefix + "_before";
		std::string type = optimizer_type(start);
		body += lbfgs_slice(name.c_str(), source, 0, start, source_type, type);
		segments.push_back({name, type});
	}

	segments.push_back({std::string(update), update_type});

	int after_start = start + length;

	if (after_start < total) {
		std::string name = prefix + "_after";
		std::string type = optimizer_type(total - after_start);
		body += lbfgs_slice(name.c_str(), source, after_start, total, source_type, type);
		segments.push_back({name, type});
	}

	std::string current_name = segments[0].first;
	std::string current_type = segments[0].second;

	for (size_t segment_index = 1; segment_index < segments.size(); segment_index++) {
		std::string next_name = prefix + "_cat" + std::to_string(segment_index);
		int next_length = 0;

		for (size_t count_index = 0; count_index <= segment_index; count_index++) {
			std::string segment_type = segments[count_index].second;
			size_t start_index = segment_type.find('<') + 1;
			size_t end_index = segment_type.find('x');
			next_length += std::stoi(segment_type.substr(start_index, end_index - start_index));
		}

		std::string next_type = optimizer_type(next_length);
		body +=
			"    %" + next_name + " = stablehlo.concatenate %" + current_name +
			", %" + segments[segment_index].first + ", dim = 0 : (" +
			current_type + ", " + segments[segment_index].second + ") -> " +
			next_type + "\n";
		current_name = next_name;
		current_type = next_type;
	}

	return current_name;
}

static std::string lbfgs_slot_vector(
	std::string& body,
	const std::string& name,
	const char* source,
	int slot,
	int count,
	const std::string& history_type
) {
	std::string type = optimizer_type(count);
	body += lbfgs_slice(
		name.c_str(), source, slot * count, (slot + 1) * count, history_type, type
	);

	return name;
}

static std::string lbfgs_rho_scalar(
	std::string& body,
	const std::string& name,
	int slot,
	const std::string& rho_type
) {
	std::string vector_name = name + "_vec";
	body += lbfgs_slice(
		vector_name.c_str(), "rho_history", slot, slot + 1, rho_type, optimizer_type(1)
	);
	body +=
		"    %" + name + " = stablehlo.reshape %" + vector_name +
		" : (tensor<1xf64>) -> tensor<f64>\n";

	return name;
}

static std::string lbfgs_reduce_dot(
	std::string& body,
	const std::string& name,
	const std::string& left,
	const std::string& right,
	const std::string& type
) {
	std::string product = name + "_product";
	body +=
		"    %" + product + " = stablehlo.multiply %" + left + ", %" + right +
		" : " + type + "\n";
	body += optimizer_reduce_sum(name.c_str(), product.c_str(), type);

	return name;
}

static std::string lbfgs_effective_lr(
	std::string& body,
	const std::string& direction,
	const std::string& type,
	double learning_rate,
	int line_search,
	double c1
) {
	if (line_search == 0) {
		body += optimizer_scalar_constant("effective_lr", learning_rate);

		return "effective_lr";
	}

	body +=
		"    %grad_sq = stablehlo.multiply %grads, %grads : " + type + "\n";
	body += optimizer_reduce_sum("f0", "grad_sq", type);
	lbfgs_reduce_dot(body, "slope_pos", "grads", direction, type);
	body += optimizer_scalar_constant("neg_one_s", -1.0);
	body += optimizer_scalar_constant("c1_s", c1 == 0.0 ? 1e-4 : c1);
	body += optimizer_scalar_constant("half_s", 0.5);
	body += optimizer_scalar_constant("min_lr_s", 1e-10);
	body += optimizer_scalar_constant("zero_s", 0.0);
	body +=
		"    %slope = stablehlo.multiply %slope_pos, %neg_one_s : tensor<f64>\n";
	body += optimizer_scalar_constant("lr_0", learning_rate);

	std::string current_lr = "lr_0";

	for (int search_index = 0; search_index < 50; search_index++) {
		std::string c1_lr = "ls_c1_lr_" + std::to_string(search_index);
		std::string armijo = "ls_armijo_" + std::to_string(search_index);
		std::string decrease = "ls_decrease_" + std::to_string(search_index);
		std::string accepted = "ls_accepted_" + std::to_string(search_index);
		std::string half_lr = "ls_half_lr_" + std::to_string(search_index);
		std::string next_lr = "lr_" + std::to_string(search_index + 1);

		body +=
			"    %" + c1_lr + " = stablehlo.multiply %c1_s, %" + current_lr +
			" : tensor<f64>\n"
			"    %" + armijo + " = stablehlo.multiply %" + c1_lr +
			", %slope : tensor<f64>\n"
			"    %" + decrease + " = stablehlo.subtract %f0, %" + armijo +
			" : tensor<f64>\n"
			"    %" + accepted + " = stablehlo.compare GT, %" + decrease +
			", %zero_s, TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>\n"
			"    %" + half_lr + " = stablehlo.multiply %" + current_lr +
			", %half_s : tensor<f64>\n"
			"    %" + next_lr + " = stablehlo.select %" + accepted +
			", %" + current_lr + ", %" + half_lr +
			" : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>\n";
		current_lr = next_lr;
	}

	body +=
		"    %effective_lr = stablehlo.maximum %" + current_lr +
		", %min_lr_s : tensor<f64>\n";

	return "effective_lr";
}

static std::string build_lbfgs_history(
	int count,
	int history_size,
	int head,
	int history_count,
	int has_previous
) {
	std::string type = optimizer_type(count);
	std::string history_type = lbfgs_history_type(history_size, count);
	std::string rho_type = lbfgs_rho_type(history_size);
	std::string one_type = optimizer_type(1);
	std::string bool_type = optimizer_bool_type(count);

	std::string body;
	body +=
		"module @xla_optimizer_lbfgs_history {\n"
		"  func.func @main(%params: " + type + ", %grads: " + type +
		", %previous_params: " + type + ", %previous_grads: " + type +
		", %s_history: " + history_type + ", %y_history: " + history_type +
		", %rho_history: " + rho_type + ") -> (" + history_type + ", " +
		history_type + ", " + rho_type + ", " + one_type + ") {\n";

	if (has_previous == 0 || history_size == 0 || history_count < 0) {
		body +=
			"    %accept_s = stablehlo.constant dense<0.0> : tensor<f64>\n"
			"    %accept = stablehlo.reshape %accept_s : (tensor<f64>) -> " + one_type + "\n"
			"    return %s_history, %y_history, %rho_history, %accept : " +
			history_type + ", " + history_type + ", " + rho_type + ", " + one_type + "\n"
			"  }\n"
			"}\n";

		return body;
	}

	int slot = head % history_size;
	body +=
		"    %s_candidate = stablehlo.subtract %params, %previous_params : " + type + "\n"
		"    %y_candidate = stablehlo.subtract %grads, %previous_grads : " + type + "\n";
	lbfgs_reduce_dot(body, "curvature", "y_candidate", "s_candidate", type);
	body += optimizer_scalar_constant("min_curvature", 1e-10);
	body += optimizer_scalar_constant("one_s", 1.0);
	body += optimizer_scalar_constant("zero_s", 0.0);
	body +=
		"    %accept_i1 = stablehlo.compare GT, %curvature, %min_curvature, TOTALORDER : (tensor<f64>, tensor<f64>) -> tensor<i1>\n";
	body += optimizer_broadcast_bool("accept_v", "accept_i1", bool_type);
	lbfgs_slot_vector(body, "old_s_slot", "s_history", slot, count, history_type);
	lbfgs_slot_vector(body, "old_y_slot", "y_history", slot, count, history_type);
	body +=
		"    %next_s_slot = stablehlo.select %accept_v, %s_candidate, %old_s_slot : (" +
			bool_type + ", " + type + ", " + type + ") -> " + type + "\n"
		"    %next_y_slot = stablehlo.select %accept_v, %y_candidate, %old_y_slot : (" +
			bool_type + ", " + type + ", " + type + ") -> " + type + "\n"
		"    %rho_candidate = stablehlo.divide %one_s, %curvature : tensor<f64>\n";
	lbfgs_rho_scalar(body, "old_rho_slot", slot, rho_type);
	body +=
		"    %next_rho_slot = stablehlo.select %accept_i1, %rho_candidate, %old_rho_slot : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>\n"
		"    %next_rho_vec = stablehlo.reshape %next_rho_slot : (tensor<f64>) -> " + one_type + "\n"
		"    %accept_s = stablehlo.select %accept_i1, %one_s, %zero_s : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>\n"
		"    %accept = stablehlo.reshape %accept_s : (tensor<f64>) -> " + one_type + "\n";

	std::string next_s = lbfgs_replace_segment(
		body, "s_history_next", "s_history", "next_s_slot",
		history_size * count, slot * count, count, history_type, type
	);
	std::string next_y = lbfgs_replace_segment(
		body, "y_history_next", "y_history", "next_y_slot",
		history_size * count, slot * count, count, history_type, type
	);
	std::string next_rho = lbfgs_replace_segment(
		body, "rho_history_next", "rho_history", "next_rho_vec",
		history_size, slot, 1, rho_type, one_type
	);

	body +=
		"    return %" + next_s + ", %" + next_y + ", %" + next_rho +
		", %accept : " + history_type + ", " + history_type + ", " +
		rho_type + ", " + one_type + "\n"
		"  }\n"
		"}\n";

	return body;
}

static std::string build_lbfgs_step(
	int count,
	int history_size,
	int head,
	int history_count,
	double learning_rate,
	int line_search,
	double c1
) {
	std::string type = optimizer_type(count);
	std::string history_type = lbfgs_history_type(history_size, count);
	std::string rho_type = lbfgs_rho_type(history_size);
	std::string body;
	body +=
		"module @xla_optimizer_lbfgs_step {\n"
		"  func.func @main(%params: " + type + ", %grads: " + type +
		", %s_history: " + history_type + ", %y_history: " + history_type +
		", %rho_history: " + rho_type + ") -> " + type + " {\n";
	body += optimizer_constant("zero_v", 0.0, type);
	body +=
		"    %direction_0 = stablehlo.add %grads, %zero_v : " + type + "\n";

	std::string direction = "direction_0";
	std::vector<std::string> alphas((size_t)history_count);

	for (int history_index = history_count - 1; history_index >= 0; history_index--) {
		int slot = (head - 1 - history_index + history_size * 2) % history_size;
		std::string suffix = std::to_string(history_index);
		std::string s_slot = lbfgs_slot_vector(
			body, "s_rev_" + suffix, "s_history", slot, count, history_type
		);
		std::string y_slot = lbfgs_slot_vector(
			body, "y_rev_" + suffix, "y_history", slot, count, history_type
		);
		std::string dot = lbfgs_reduce_dot(body, "dot_rev_" + suffix, s_slot, direction, type);
		std::string rho = lbfgs_rho_scalar(body, "rho_rev_" + suffix, slot, rho_type);
		std::string alpha = "alpha_" + suffix;
		body +=
			"    %" + alpha + " = stablehlo.multiply %" + rho + ", %" + dot +
			" : tensor<f64>\n";
		body += optimizer_broadcast(("alpha_v_" + suffix).c_str(), alpha.c_str(), type);
		std::string scaled_y = "scaled_y_" + suffix;
		std::string next_direction = "direction_rev_" + suffix;
		body +=
			"    %" + scaled_y + " = stablehlo.multiply %" + y_slot +
			", %alpha_v_" + suffix + " : " + type + "\n"
			"    %" + next_direction + " = stablehlo.subtract %" + direction +
			", %" + scaled_y + " : " + type + "\n";
		direction = next_direction;
		alphas[(size_t)history_index] = alpha;
	}

	if (history_count > 0) {
		int slot = (head - 1 + history_size * 2) % history_size;
		std::string y_slot = lbfgs_slot_vector(
			body, "y_gamma", "y_history", slot, count, history_type
		);
		std::string s_slot = lbfgs_slot_vector(
			body, "s_gamma", "s_history", slot, count, history_type
		);
		std::string yy = lbfgs_reduce_dot(body, "gamma_yy", y_slot, y_slot, type);
		std::string ys = lbfgs_reduce_dot(body, "gamma_ys", y_slot, s_slot, type);
		body +=
			"    %gamma = stablehlo.divide %" + ys + ", %" + yy + " : tensor<f64>\n";
		body += optimizer_broadcast("gamma_v", "gamma", type);
		body +=
			"    %direction_gamma = stablehlo.multiply %" + direction +
			", %gamma_v : " + type + "\n";
		direction = "direction_gamma";
	}

	for (int history_index = 0; history_index < history_count; history_index++) {
		int slot = (head - history_count + history_index + history_size * 2) % history_size;
		std::string suffix = std::to_string(history_index);
		std::string s_slot = lbfgs_slot_vector(
			body, "s_fwd_" + suffix, "s_history", slot, count, history_type
		);
		std::string y_slot = lbfgs_slot_vector(
			body, "y_fwd_" + suffix, "y_history", slot, count, history_type
		);
		std::string dot = lbfgs_reduce_dot(body, "dot_fwd_" + suffix, y_slot, direction, type);
		std::string rho = lbfgs_rho_scalar(body, "rho_fwd_" + suffix, slot, rho_type);
		std::string beta = "beta_" + suffix;
		std::string coeff = "coeff_" + suffix;
		body +=
			"    %" + beta + " = stablehlo.multiply %" + rho + ", %" + dot +
			" : tensor<f64>\n"
			"    %" + coeff + " = stablehlo.subtract %" + alphas[(size_t)history_index] +
			", %" + beta + " : tensor<f64>\n";
		body += optimizer_broadcast(("coeff_v_" + suffix).c_str(), coeff.c_str(), type);
		std::string scaled_s = "scaled_s_" + suffix;
		std::string next_direction = "direction_fwd_" + suffix;
		body +=
			"    %" + scaled_s + " = stablehlo.multiply %" + s_slot +
			", %coeff_v_" + suffix + " : " + type + "\n"
			"    %" + next_direction + " = stablehlo.add %" + direction +
			", %" + scaled_s + " : " + type + "\n";
		direction = next_direction;
	}

	std::string effective_lr = lbfgs_effective_lr(
		body, direction, type, learning_rate, line_search, c1
	);
	body += optimizer_broadcast("effective_lr_v", effective_lr.c_str(), type);
	body +=
		"    %step = stablehlo.multiply %" + direction +
		", %effective_lr_v : " + type + "\n"
		"    %out = stablehlo.subtract %params, %step : " + type + "\n"
		"    return %out : " + type + "\n"
		"  }\n"
		"}\n";

	return body;
}

extern "C" {

int xla_optimizer_init(const char* platform) {
	return xla_init(platform);
}

void xla_optimizer_shutdown(void) {
	std::lock_guard<std::mutex> guard(g_optimizer_execs_mutex);

	if (!g_api) {
		g_optimizer_execs.clear();
		return;
	}

	for (auto& item : g_optimizer_execs) {
		PJRT_LoadedExecutable_Destroy_Args args{};
		args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
		args.executable = item.second;
		g_api->PJRT_LoadedExecutable_Destroy(&args);
	}

	g_optimizer_execs.clear();
}

int xla_optimizer_adam(
	double *out, double *moment, double *variance,
	const double *params, const double *grads, int count,
	double beta1, double beta2, double learning_rate, double eps
) {
	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({moment, count, "xla optimizer upload moment"});
	inputs.push_back({variance, count, "xla optimizer upload variance"});

	return optimizer_execute_same_size(
		optimizer_key("adam", count, {beta1, beta2, learning_rate, eps}),
		build_adam(count, beta1, beta2, learning_rate, eps),
		inputs,
		{
			{out, count, "xla optimizer download adam out"},
			{moment, count, "xla optimizer download adam moment"},
			{variance, count, "xla optimizer download adam variance"},
		}
	);
}

int xla_optimizer_adamw(
	double *out, double *moment, double *variance,
	const double *params, const double *grads, int count,
	double beta1, double beta2, double learning_rate, double eps,
	double weight_decay_step
) {
	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({moment, count, "xla optimizer upload moment"});
	inputs.push_back({variance, count, "xla optimizer upload variance"});

	return optimizer_execute_same_size(
		optimizer_key("adamw", count, {beta1, beta2, learning_rate, eps, weight_decay_step}),
		build_adamw(count, beta1, beta2, learning_rate, eps, weight_decay_step),
		inputs,
		{
			{out, count, "xla optimizer download adamw out"},
			{moment, count, "xla optimizer download adamw moment"},
			{variance, count, "xla optimizer download adamw variance"},
		}
	);
}

int xla_optimizer_adamax(
	double *out, double *moment, double *infinity_norm,
	const double *params, const double *grads, int count,
	double beta1, double beta2, double learning_rate, double eps
) {
	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({moment, count, "xla optimizer upload moment"});
	inputs.push_back({infinity_norm, count, "xla optimizer upload infinity norm"});

	return optimizer_execute_same_size(
		optimizer_key("adamax", count, {beta1, beta2, learning_rate, eps}),
		build_adamax(count, beta1, beta2, learning_rate, eps),
		inputs,
		{
			{out, count, "xla optimizer download adamax out"},
			{moment, count, "xla optimizer download adamax moment"},
			{infinity_norm, count, "xla optimizer download adamax norm"},
		}
	);
}

int xla_optimizer_sgd(
	double *out, double *velocity,
	const double *params, const double *grads, int count,
	double learning_rate, double weight_decay, double momentum, int nesterov
) {
	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({velocity, count, "xla optimizer upload velocity"});

	return optimizer_execute_same_size(
		optimizer_key(
			"sgd", count, {learning_rate, weight_decay, momentum, (double)nesterov}
		),
		build_sgd(count, learning_rate, weight_decay, momentum, nesterov),
		inputs,
		{
			{out, count, "xla optimizer download sgd out"},
			{velocity, count, "xla optimizer download sgd velocity"},
		}
	);
}

int xla_optimizer_lion(
	double *out, double *moment,
	const double *params, const double *grads, int count,
	double learning_rate, double beta1, double beta2, double weight_decay
) {
	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({moment, count, "xla optimizer upload moment"});

	return optimizer_execute_same_size(
		optimizer_key("lion", count, {learning_rate, beta1, beta2, weight_decay}),
		build_lion(count, learning_rate, beta1, beta2, weight_decay),
		inputs,
		{
			{out, count, "xla optimizer download lion out"},
			{moment, count, "xla optimizer download lion moment"},
		}
	);
}

int xla_optimizer_rmsprop(
	double *out, double *square_average, double *momentum_buffer,
	double *grad_average, const double *params, const double *grads,
	int count, double learning_rate, double alpha, double eps,
	double momentum, double weight_decay, int centered
) {
	if (count < 0) return -1;

	std::vector<double> grad_average_storage;
	double* grad_average_values = grad_average;

	if (!grad_average_values) {
		grad_average_storage.assign((size_t)count, 0.0);
		grad_average_values = grad_average_storage.data();
	}

	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({square_average, count, "xla optimizer upload square average"});
	inputs.push_back({momentum_buffer, count, "xla optimizer upload momentum buffer"});
	inputs.push_back({grad_average_values, count, "xla optimizer upload grad average"});

	return optimizer_execute_same_size(
		optimizer_key(
			"rmsprop", count,
			{learning_rate, alpha, eps, momentum, weight_decay, (double)centered}
		),
		build_rmsprop(count, learning_rate, alpha, eps, momentum, weight_decay, centered),
		inputs,
		{
			{out, count, "xla optimizer download rmsprop out"},
			{square_average, count, "xla optimizer download rmsprop square average"},
			{momentum_buffer, count, "xla optimizer download rmsprop momentum buffer"},
			{grad_average_values, count, "xla optimizer download rmsprop grad average"},
		}
	);
}

int xla_optimizer_hebbian(
	double *out, const double *params, const double *grads, int count,
	double learning_rate, double max_norm
) {
	return optimizer_execute_same_size(
		optimizer_key("hebbian", count, {learning_rate, max_norm}),
		build_hebbian(count, learning_rate, max_norm),
		optimizer_inputs(params, grads, count),
		{{out, count, "xla optimizer download hebbian out"}}
	);
}

int xla_optimizer_lars(
	double *out, double *velocity,
	const double *params, const double *grads, int count,
	double learning_rate, double eta, double momentum,
	double weight_decay, double eps
) {
	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({velocity, count, "xla optimizer upload velocity"});

	return optimizer_execute_same_size(
		optimizer_key("lars", count, {learning_rate, eta, momentum, weight_decay, eps}),
		build_lars(count, learning_rate, eta, momentum, weight_decay, eps),
		inputs,
		{
			{out, count, "xla optimizer download lars out"},
			{velocity, count, "xla optimizer download lars velocity"},
		}
	);
}

int xla_optimizer_lamb(
	double *out, double *moment, double *variance,
	const double *params, const double *grads, int count,
	double learning_rate, double beta1, double beta2, double eps,
	double weight_decay, double bias_correction1_inv,
	double bias_correction2_inv
) {
	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({moment, count, "xla optimizer upload moment"});
	inputs.push_back({variance, count, "xla optimizer upload variance"});

	return optimizer_execute_same_size(
		optimizer_key(
			"lamb", count,
			{
				learning_rate, beta1, beta2, eps, weight_decay,
				bias_correction1_inv, bias_correction2_inv,
			}
		),
		build_lamb(
			count, learning_rate, beta1, beta2, eps, weight_decay,
			bias_correction1_inv, bias_correction2_inv
		),
		inputs,
		{
			{out, count, "xla optimizer download lamb out"},
			{moment, count, "xla optimizer download lamb moment"},
			{variance, count, "xla optimizer download lamb variance"},
		}
	);
}

int xla_optimizer_adagrad(
	double *out, double *accumulator,
	const double *params, const double *grads, int count,
	double learning_rate, double eps, double weight_decay
) {
	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({accumulator, count, "xla optimizer upload accumulator"});

	return optimizer_execute_same_size(
		optimizer_key("adagrad", count, {learning_rate, eps, weight_decay}),
		build_adagrad(count, learning_rate, eps, weight_decay),
		inputs,
		{
			{out, count, "xla optimizer download adagrad out"},
			{accumulator, count, "xla optimizer download adagrad accumulator"},
		}
	);
}

int xla_optimizer_adadelta(
	double *out, double *grad_average, double *delta_average,
	const double *params, const double *grads, int count,
	double rho, double eps, double weight_decay
) {
	auto inputs = optimizer_inputs(params, grads, count);
	inputs.push_back({grad_average, count, "xla optimizer upload grad average"});
	inputs.push_back({delta_average, count, "xla optimizer upload delta average"});

	return optimizer_execute_same_size(
		optimizer_key("adadelta", count, {rho, eps, weight_decay}),
		build_adadelta(count, rho, eps, weight_decay),
		inputs,
		{
			{out, count, "xla optimizer download adadelta out"},
			{grad_average, count, "xla optimizer download adadelta grad average"},
			{delta_average, count, "xla optimizer download adadelta delta average"},
		}
	);
}

int xla_optimizer_lbfgs(
	double *out, double *s_history, double *y_history, double *rho_history,
	int *head, int *history_count, const double *params,
	const double *grads, const double *previous_params,
	const double *previous_grads, int has_previous, int count,
	int history_size, double learning_rate, int line_search, double c1
) {
	if (!out || !params || !grads || !head || !history_count || count < 0 || history_size < 0) {
		return -1;
	}

	if (count == 0) return 0;
	if (*head < 0 || *history_count < 0 || *history_count > history_size) return -1;

	if (has_previous != 0 && (!previous_params || !previous_grads)) return -1;
	if (history_size > 0 && (!s_history || !y_history || !rho_history)) return -1;

	int history_values = history_size * count;
	std::vector<double> accept(1, 0.0);

	int rc = optimizer_execute(
		optimizer_key(
			"lbfgs_history", count,
			{(double)history_size, (double)*head, (double)*history_count, (double)has_previous}
		),
		build_lbfgs_history(count, history_size, *head, *history_count, has_previous),
		{
			{params, count, "xla optimizer upload lbfgs params"},
			{grads, count, "xla optimizer upload lbfgs grads"},
			{previous_params, count, "xla optimizer upload lbfgs previous params"},
			{previous_grads, count, "xla optimizer upload lbfgs previous grads"},
			{s_history, history_values, "xla optimizer upload lbfgs s history"},
			{y_history, history_values, "xla optimizer upload lbfgs y history"},
			{rho_history, history_size, "xla optimizer upload lbfgs rho history"},
		},
		{
			{s_history, history_values, "xla optimizer download lbfgs s history"},
			{y_history, history_values, "xla optimizer download lbfgs y history"},
			{rho_history, history_size, "xla optimizer download lbfgs rho history"},
			{accept.data(), 1, "xla optimizer download lbfgs accept"},
		}
	);

	if (rc != 0) {
		return rc;
	}

	if (accept[0] != 0.0 && history_size > 0) {
		*head += 1;

		if (*history_count < history_size) {
			*history_count += 1;
		}
	}

	return optimizer_execute(
		optimizer_key(
			"lbfgs_step", count,
			{
				(double)history_size, (double)*head, (double)*history_count,
				learning_rate, (double)line_search, c1,
			}
		),
		build_lbfgs_step(
			count, history_size, *head, *history_count, learning_rate, line_search, c1
		),
		{
			{params, count, "xla optimizer upload lbfgs step params"},
			{grads, count, "xla optimizer upload lbfgs step grads"},
			{s_history, history_values, "xla optimizer upload lbfgs step s history"},
			{y_history, history_values, "xla optimizer upload lbfgs step y history"},
			{rho_history, history_size, "xla optimizer upload lbfgs step rho history"},
		},
		{{out, count, "xla optimizer download lbfgs out"}}
	);
}

} // extern "C"
