// XLA resident tensor backend — PJRT C API implementation.
//
// This unit keeps PJRT_Buffer ownership explicit so tensor operations can run
// buffer-to-buffer without staging through host memory between kernels.

#include "tensor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

struct XLA_Tensor {
    PJRT_Buffer* buffer;
    std::vector<int64_t> dims;
    int64_t elements;
};

static std::unordered_map<std::string, PJRT_LoadedExecutable*> g_tensor_execs;
static std::mutex                                       g_tensor_execs_mutex;

/*
tensor_count_elements returns the product of dims, or 0 if any dimension is 0.
Returns -1 if the product overflows int64_t (callers must abort / fail the operation).
*/
static int64_t tensor_count_elements(const int64_t* dims, int rank) {
    int64_t elements = 1;

    for (int dimension_index = 0; dimension_index < rank; dimension_index++) {
        int64_t dimension = dims[dimension_index];

        if (dimension == 0) {
            return 0;
        }

        if (dimension != 0 && elements > INT64_MAX / dimension) {
            return -1;
        }

        elements *= dimension;
    }

    return elements;
}

static std::string tensor_dims_key(const std::vector<int64_t>& dims) {
    if (dims.empty()) return "scalar";

    std::string key;

    for (size_t dimension_index = 0; dimension_index < dims.size(); dimension_index++) {
        if (dimension_index > 0) key += "x";
        key += std::to_string(dims[dimension_index]);
    }

    return key;
}

static std::string tensor_type(const std::vector<int64_t>& dims, const char* scalar_type) {
    if (dims.empty()) {
        return std::string("tensor<") + scalar_type + ">";
    }

    std::string result = "tensor<";

    for (int64_t dimension : dims) {
        result += std::to_string(dimension);
        result += "x";
    }

    result += scalar_type;
    result += ">";

    return result;
}

static std::string tensor_flat_type(int64_t elements) {
    return "tensor<" + std::to_string(elements) + "xf64>";
}

static bool tensor_same_shape(const XLA_Tensor* left, const XLA_Tensor* right) {
    return left && right && left->dims == right->dims;
}

static XLA_Tensor* tensor_wrap(PJRT_Buffer* buffer, const std::vector<int64_t>& dims) {
    if (!buffer) return nullptr;

    int64_t elements = tensor_count_elements(dims.data(), (int)dims.size());

    if (elements < 0) {
        tensor_destroy_buffer(buffer);

        return nullptr;
    }

    auto* tensor = new XLA_Tensor{};
    tensor->buffer = buffer;
    tensor->dims = dims;
    tensor->elements = elements;

    return tensor;
}

static int tensor_destroy_buffer(PJRT_Buffer* buffer) {
    if (!buffer) return 0;
    if (!g_api) return -1;

    PJRT_Buffer_Destroy_Args args{};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.buffer = buffer;

    PJRT_Error* err = g_api->PJRT_Buffer_Destroy(&args);
    return check(g_api, err, "PJRT_Buffer_Destroy(tensor)") ? 0 : -1;
}

static bool tensor_await_and_destroy(PJRT_Event* event, const char* context) {
    if (!g_api) return false;

    if (!event) return true;

    PJRT_Event_Await_Args await_args{};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.event = event;

    PJRT_Error* err = g_api->PJRT_Event_Await(&await_args);
    bool ok = check(g_api, err, context);

    PJRT_Event_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    destroy_args.event = event;
    g_api->PJRT_Event_Destroy(&destroy_args);

    return ok;
}

static bool tensor_await_ready(PJRT_Buffer* buffer, const char* context) {
    if (!g_api || !buffer) return false;

    PJRT_Buffer_ReadyEvent_Args ready_args{};
    ready_args.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    ready_args.buffer = buffer;

    PJRT_Error* err = g_api->PJRT_Buffer_ReadyEvent(&ready_args);
    if (!check(g_api, err, context)) return false;

    return tensor_await_and_destroy(ready_args.event, context);
}

static PJRT_LoadedExecutable* tensor_compile(
    const std::string& key, const std::string& mlir_text
) {
    std::lock_guard<std::mutex> guard(g_tensor_execs_mutex);

    auto existing = g_tensor_execs.find(key);

    if (existing != g_tensor_execs.end()) {
        return existing->second;
    }

    PJRT_LoadedExecutable* executable = compile_stablehlo(mlir_text);

    if (executable) {
        g_tensor_execs[key] = executable;
    }

    return executable;
}

static XLA_Tensor* tensor_execute(
    PJRT_LoadedExecutable* executable,
    const std::vector<const XLA_Tensor*>& inputs,
    const std::vector<int64_t>& output_dims
) {
    if (!executable || inputs.empty()) return nullptr;

    std::vector<PJRT_Buffer*> input_buffers(inputs.size());

    for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
        if (!inputs[input_index] || !inputs[input_index]->buffer) return nullptr;
        input_buffers[input_index] = inputs[input_index]->buffer;
    }

    PJRT_Buffer** argument_lists[1] = { input_buffers.data() };
    PJRT_Buffer* output_buffer = nullptr;
    PJRT_Buffer** output_lists[1] = { &output_buffer };

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

    if (!check(g_api, err, "PJRT_LoadedExecutable_Execute(tensor)")) {
        return nullptr;
    }

    return tensor_wrap(output_buffer, output_dims);
}

static std::string tensor_build_unary(
    const char* module_name,
    const std::vector<int64_t>& dims,
    const std::string& body
) {
    std::string t = tensor_type(dims, "f64");

    return
        "module @" + std::string(module_name) + " {\n"
        "  func.func @main(%arg0: " + t + ") -> " + t + " {\n" +
        body +
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string tensor_build_relu(const std::vector<int64_t>& dims) {
    std::string t = tensor_type(dims, "f64");

    return tensor_build_unary(
        "tensor_relu",
        dims,
        "    %zero = stablehlo.constant dense<0.0> : " + t + "\n"
        "    %out = stablehlo.maximum %arg0, %zero : " + t + "\n"
    );
}

static std::string tensor_build_leaky_relu(
    const std::vector<int64_t>& dims, const char* alpha
) {
    std::string t = tensor_type(dims, "f64");
    std::string mask_type = tensor_type(dims, "i1");

    return tensor_build_unary(
        "tensor_leaky_relu",
        dims,
        "    %zero = stablehlo.constant dense<0.0> : " + t + "\n"
        "    %alpha = stablehlo.constant dense<" + std::string(alpha) + "> : " + t + "\n"
        "    %scaled = stablehlo.multiply %arg0, %alpha : " + t + "\n"
        "    %mask = stablehlo.compare GT, %arg0, %zero, TOTALORDER : (" + t + ", " + t + ") -> " + mask_type + "\n"
        "    %out = stablehlo.select %mask, %arg0, %scaled : (" + mask_type + ", " + t + ", " + t + ") -> " + t + "\n"
    );
}

static std::string tensor_build_gelu(const std::vector<int64_t>& dims) {
    std::string t = tensor_type(dims, "f64");

    return tensor_build_unary(
        "tensor_gelu",
        dims,
        "    %half = stablehlo.constant dense<5.000000e-01> : " + t + "\n"
        "    %one = stablehlo.constant dense<1.0> : " + t + "\n"
        "    %rsqrt2 = stablehlo.constant dense<0.7071067811865476> : " + t + "\n"
        "    %scaled = stablehlo.multiply %arg0, %rsqrt2 : " + t + "\n"
        "    %erf = stablehlo.erf %scaled : " + t + "\n"
        "    %gate = stablehlo.add %one, %erf : " + t + "\n"
        "    %weighted = stablehlo.multiply %arg0, %gate : " + t + "\n"
        "    %out = stablehlo.multiply %half, %weighted : " + t + "\n"
    );
}

static std::string tensor_build_simple_unary(
    const char* module_name, const char* stablehlo_op, const std::vector<int64_t>& dims
) {
    std::string t = tensor_type(dims, "f64");

    return tensor_build_unary(
        module_name,
        dims,
        "    %out = " + std::string(stablehlo_op) + " %arg0 : " + t + "\n"
    );
}

static std::string tensor_build_swish(const std::vector<int64_t>& dims) {
    std::string t = tensor_type(dims, "f64");

    return tensor_build_unary(
        "tensor_swish",
        dims,
        "    %sigmoid = stablehlo.logistic %arg0 : " + t + "\n"
        "    %out = stablehlo.multiply %arg0, %sigmoid : " + t + "\n"
    );
}

static std::string tensor_build_selu(const std::vector<int64_t>& dims) {
    std::string t = tensor_type(dims, "f64");
    std::string mask_type = tensor_type(dims, "i1");

    return tensor_build_unary(
        "tensor_selu",
        dims,
        "    %zero = stablehlo.constant dense<0.0> : " + t + "\n"
        "    %one = stablehlo.constant dense<1.0> : " + t + "\n"
        "    %scale = stablehlo.constant dense<1.0507009873554805> : " + t + "\n"
        "    %scale_alpha = stablehlo.constant dense<1.7580993408473766> : " + t + "\n"
        "    %positive = stablehlo.multiply %arg0, %scale : " + t + "\n"
        "    %exp = stablehlo.exponential %arg0 : " + t + "\n"
        "    %minus_one = stablehlo.subtract %exp, %one : " + t + "\n"
        "    %negative = stablehlo.multiply %minus_one, %scale_alpha : " + t + "\n"
        "    %mask = stablehlo.compare GT, %arg0, %zero, TOTALORDER : (" + t + ", " + t + ") -> " + mask_type + "\n"
        "    %out = stablehlo.select %mask, %positive, %negative : (" + mask_type + ", " + t + ", " + t + ") -> " + t + "\n"
    );
}

static std::string tensor_build_swiglu(const XLA_Tensor* input) {
    std::vector<int64_t> output_dims = input->dims;
    int last = (int)output_dims.size() - 1;
    output_dims[last] /= 2;

    int64_t output_elements = tensor_count_elements(output_dims.data(), (int)output_dims.size());

    if (output_elements < 0 || input->elements < 0) {
        return "";
    }

    std::string input_type = tensor_type(input->dims, "f64");
    std::string output_type = tensor_type(output_dims, "f64");
    std::string input_flat_type = tensor_flat_type(input->elements);
    std::string output_flat_type = tensor_flat_type(output_elements);

    return
        "module @tensor_swiglu {\n"
        "  func.func @main(%arg0: " + input_type + ") -> " + output_type + " {\n"
        "    %flat = stablehlo.reshape %arg0 : (" + input_type + ") -> " + input_flat_type + "\n"
        "    %gates = stablehlo.slice %flat [0:" + std::to_string(output_elements) + "] : (" + input_flat_type + ") -> " + output_flat_type + "\n"
        "    %values = stablehlo.slice %flat [" + std::to_string(output_elements) + ":" + std::to_string(input->elements) + "] : (" + input_flat_type + ") -> " + output_flat_type + "\n"
        "    %sigmoid = stablehlo.logistic %gates : " + output_flat_type + "\n"
        "    %flat_out = stablehlo.multiply %sigmoid, %values : " + output_flat_type + "\n"
        "    %out = stablehlo.reshape %flat_out : (" + output_flat_type + ") -> " + output_type + "\n"
        "    return %out : " + output_type + "\n"
        "  }\n"
        "}\n";
}

static std::string tensor_build_binary(
    const char* module_name,
    const char* stablehlo_op,
    const std::vector<int64_t>& dims
) {
    std::string t = tensor_type(dims, "f64");

    return
        "module @" + std::string(module_name) + " {\n"
        "  func.func @main(%left: " + t + ", %right: " + t + ") -> " + t + " {\n"
        "    %out = " + std::string(stablehlo_op) + " %left, %right : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string tensor_build_matmul(const XLA_Tensor* left, const XLA_Tensor* right) {
    std::vector<int64_t> output_dims = { left->dims[0], right->dims[1] };
    std::string left_type = tensor_type(left->dims, "f64");
    std::string right_type = tensor_type(right->dims, "f64");
    std::string output_type = tensor_type(output_dims, "f64");

    return
        "module @tensor_matmul {\n"
        "  func.func @main(%left: " + left_type + ", %right: " + right_type + ") -> " + output_type + " {\n"
        "    %out = stablehlo.dot_general %left, %right, contracting_dims = [1] x [0] : (" + left_type + ", " + right_type + ") -> " + output_type + "\n"
        "    return %out : " + output_type + "\n"
        "  }\n"
        "}\n";
}

static std::string tensor_gelu_body(const std::string& t, const char* input_name) {
    return
        "    %half = stablehlo.constant dense<5.000000e-01> : " + t + "\n"
        "    %one = stablehlo.constant dense<1.0> : " + t + "\n"
        "    %rsqrt2 = stablehlo.constant dense<0.7071067811865476> : " + t + "\n"
        "    %scaled = stablehlo.multiply " + input_name + ", %rsqrt2 : " + t + "\n"
        "    %erf = stablehlo.erf %scaled : " + t + "\n"
        "    %gate = stablehlo.add %one, %erf : " + t + "\n"
        "    %weighted = stablehlo.multiply " + input_name + ", %gate : " + t + "\n"
        "    %out = stablehlo.multiply %half, %weighted : " + t + "\n";
}

static std::string tensor_build_matmul_add(
    const XLA_Tensor* left,
    const XLA_Tensor* right,
    const XLA_Tensor* bias,
    bool apply_gelu
) {
    std::vector<int64_t> output_dims = { left->dims[0], right->dims[1] };
    std::string left_type = tensor_type(left->dims, "f64");
    std::string right_type = tensor_type(right->dims, "f64");
    std::string bias_type = tensor_type(bias->dims, "f64");
    std::string output_type = tensor_type(output_dims, "f64");

    std::string body =
        "    %product = stablehlo.dot_general %left, %right, contracting_dims = [1] x [0] : (" + left_type + ", " + right_type + ") -> " + output_type + "\n";

    if (bias->elements == right->dims[1]) {
        body +=
            "    %bias2d = stablehlo.broadcast_in_dim %bias, dims = [1] : (" + bias_type + ") -> " + output_type + "\n"
            "    %biased = stablehlo.add %product, %bias2d : " + output_type + "\n";
    } else {
        body += "    %biased = stablehlo.add %product, %bias : " + output_type + "\n";
    }

    std::string return_ssa;

    if (apply_gelu) {
        body += tensor_gelu_body(output_type, "%biased");
        return_ssa = "%out";
    } else {
        return_ssa = "%biased";
    }

    return
        "module @tensor_matmul_add {\n"
        "  func.func @main(%left: " + left_type + ", %right: " + right_type + ", %bias: " + bias_type + ") -> " + output_type + " {\n" +
        body +
        "    return " + return_ssa + " : " + output_type + "\n"
        "  }\n"
        "}\n";
}

static int tensor_finish_output(XLA_Tensor* result, XLA_Tensor** output) {
    if (!result) return -1;

    *output = result;
    return 0;
}

static int tensor_unary(
    const XLA_Tensor* input,
    XLA_Tensor** output,
    const std::string& op_key,
    const std::string& mlir_text
) {
    if (!input || !output) return -1;

    std::string key = op_key + ":" + tensor_dims_key(input->dims);
    PJRT_LoadedExecutable* executable = tensor_compile(key, mlir_text);
    XLA_Tensor* result = tensor_execute(executable, { input }, input->dims);

    return tensor_finish_output(result, output);
}

static int tensor_binary(
    const XLA_Tensor* left,
    const XLA_Tensor* right,
    XLA_Tensor** output,
    const std::string& op_key,
    const std::string& mlir_text
) {
    if (!tensor_same_shape(left, right) || !output) return -1;

    std::string key = op_key + ":" + tensor_dims_key(left->dims);
    PJRT_LoadedExecutable* executable = tensor_compile(key, mlir_text);
    XLA_Tensor* result = tensor_execute(executable, { left, right }, left->dims);

    return tensor_finish_output(result, output);
}

extern "C" {

int xla_tensor_init(const char* platform) {
    return xla_init(platform);
}

void xla_tensor_shutdown(void) {
    std::lock_guard<std::mutex> guard(g_tensor_execs_mutex);

    if (!g_api) {
        g_tensor_execs.clear();
        return;
    }

    for (auto& item : g_tensor_execs) {
        PJRT_LoadedExecutable_Destroy_Args args{};
        args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        args.executable = item.second;
        g_api->PJRT_LoadedExecutable_Destroy(&args);
    }

    g_tensor_execs.clear();
}

int xla_tensor_upload_f64(const double* src, const int64_t* dims, int rank, XLA_Tensor** out) {
    if (!out) return -1;

    *out = nullptr;

    if (!g_client || rank < 0) return -1;

    std::vector<int64_t> owned_dims;

    for (int dimension_index = 0; dimension_index < rank; dimension_index++) {
        owned_dims.push_back(dims[dimension_index]);
    }

    int64_t elements = tensor_count_elements(owned_dims.data(), (int)owned_dims.size());

    if (elements < 0) return -1;

    std::vector<double> host_values;

    if (elements > 0) {
        if (!src) return -1;

        host_values.assign(src, src + elements);
    }

    PJRT_Client_BufferFromHostBuffer_Args args{};
    args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    args.client = g_client;
    args.data = elements == 0 ? nullptr : host_values.data();
    args.type = PJRT_Buffer_Type_F64;
    args.dims = owned_dims.empty() ? nullptr : owned_dims.data();
    args.num_dims = owned_dims.size();
    args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    args.device = g_device;
    args.memory = g_memory;

    PJRT_Error* err = g_api->PJRT_Client_BufferFromHostBuffer(&args);

    if (!check(g_api, err, "PJRT_Client_BufferFromHostBuffer(tensor)")) {
        return -1;
    }

    if (!tensor_await_and_destroy(args.done_with_host_buffer, "Event_Await(upload host buffer)")) {
        tensor_destroy_buffer(args.buffer);
        return -1;
    }

    if (!tensor_await_ready(args.buffer, "ReadyEvent(upload tensor)")) {
        tensor_destroy_buffer(args.buffer);
        return -1;
    }

    XLA_Tensor* wrapped = tensor_wrap(args.buffer, owned_dims);

    if (!wrapped) return -1;

    *out = wrapped;

    return 0;
}

int xla_tensor_download_f64(const XLA_Tensor* tensor, double* dst, int64_t n_elements) {
    if (!tensor || !tensor->buffer || n_elements < 0 || tensor->elements != n_elements) return -1;

    PJRT_Buffer_ToHostBuffer_Args args{};
    args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    args.src = tensor->buffer;
    args.dst = dst;
    args.dst_size = (size_t)n_elements * sizeof(double);

    PJRT_Error* err = g_api->PJRT_Buffer_ToHostBuffer(&args);

    if (!check(g_api, err, "PJRT_Buffer_ToHostBuffer(tensor)")) {
        return -1;
    }

    return tensor_await_and_destroy(args.event, "Event_Await(download tensor)") ? 0 : -1;
}

int xla_tensor_free(XLA_Tensor* tensor) {
    if (!tensor) return 0;

    int rc = tensor_destroy_buffer(tensor->buffer);
    tensor->buffer = nullptr;
    delete tensor;

    return rc;
}

int xla_tensor_relu(const XLA_Tensor* input, XLA_Tensor** output) {
    if (!input || !output) return -1;

    return tensor_unary(input, output, "relu", tensor_build_relu(input->dims));
}

int xla_tensor_leaky_relu(const XLA_Tensor* input, double alpha, XLA_Tensor** output) {
    if (!input || !output) return -1;

    char alpha_text[64];
    snprintf(alpha_text, sizeof(alpha_text), "%.17g", alpha);
    std::string key = "leaky_relu:" + tensor_dims_key(input->dims) + ":" + alpha_text;
    PJRT_LoadedExecutable* executable = tensor_compile(
        key, tensor_build_leaky_relu(input->dims, alpha_text)
    );
    XLA_Tensor* result = tensor_execute(executable, { input }, input->dims);

    return tensor_finish_output(result, output);
}

int xla_tensor_gelu(const XLA_Tensor* input, XLA_Tensor** output) {
    if (!input || !output) return -1;

    return tensor_unary(input, output, "gelu", tensor_build_gelu(input->dims));
}

int xla_tensor_tanh(const XLA_Tensor* input, XLA_Tensor** output) {
    if (!input || !output) return -1;

    return tensor_unary(
        input, output, "tanh", tensor_build_simple_unary("tensor_tanh", "stablehlo.tanh", input->dims)
    );
}

int xla_tensor_sigmoid(const XLA_Tensor* input, XLA_Tensor** output) {
    if (!input || !output) return -1;

    return tensor_unary(
        input, output, "sigmoid", tensor_build_simple_unary("tensor_sigmoid", "stablehlo.logistic", input->dims)
    );
}

int xla_tensor_swish(const XLA_Tensor* input, XLA_Tensor** output) {
    if (!input || !output) return -1;

    return tensor_unary(input, output, "swish", tensor_build_swish(input->dims));
}

int xla_tensor_selu(const XLA_Tensor* input, XLA_Tensor** output) {
    if (!input || !output) return -1;

    return tensor_unary(input, output, "selu", tensor_build_selu(input->dims));
}

int xla_tensor_swiglu(const XLA_Tensor* input, XLA_Tensor** output) {
    if (!input || !output || input->dims.empty()) return -1;

    int last = (int)input->dims.size() - 1;

    if (input->dims[last] % 2 != 0) return -1;

    std::vector<int64_t> output_dims = input->dims;
    output_dims[last] /= 2;

    std::string key = "swiglu:" + tensor_dims_key(input->dims);
    PJRT_LoadedExecutable* executable = tensor_compile(key, tensor_build_swiglu(input));
    XLA_Tensor* result = tensor_execute(executable, { input }, output_dims);

    return tensor_finish_output(result, output);
}

int xla_tensor_add(const XLA_Tensor* left, const XLA_Tensor* right, XLA_Tensor** output) {
    if (!left || !right || !output) return -1;

    return tensor_binary(
        left,
        right,
        output,
        "add",
        tensor_build_binary("tensor_add", "stablehlo.add", left->dims)
    );
}

int xla_tensor_mul(const XLA_Tensor* left, const XLA_Tensor* right, XLA_Tensor** output) {
    if (!left || !right || !output) return -1;

    return tensor_binary(
        left,
        right,
        output,
        "mul",
        tensor_build_binary("tensor_mul", "stablehlo.multiply", left->dims)
    );
}

int xla_tensor_matmul(const XLA_Tensor* left, const XLA_Tensor* right, XLA_Tensor** output) {
    if (!left || !right || !output || left->dims.size() != 2 || right->dims.size() != 2) {
        return -1;
    }

    if (left->dims[1] != right->dims[0]) return -1;

    std::vector<int64_t> output_dims = { left->dims[0], right->dims[1] };
    std::string key = "matmul:" + tensor_dims_key(left->dims) + ":" + tensor_dims_key(right->dims);
    PJRT_LoadedExecutable* executable = tensor_compile(key, tensor_build_matmul(left, right));
    XLA_Tensor* result = tensor_execute(executable, { left, right }, output_dims);

    return tensor_finish_output(result, output);
}

int xla_tensor_matmul_add(
    const XLA_Tensor* left,
    const XLA_Tensor* right,
    const XLA_Tensor* bias,
    XLA_Tensor** output,
    bool apply_gelu
) {
    if (!left || !right || !bias || !output || left->dims.size() != 2 || right->dims.size() != 2) {
        return -1;
    }

    if (left->dims[1] != right->dims[0]) return -1;

    int64_t m = left->dims[0];
    int64_t n = right->dims[1];

    if (bias->elements != n && bias->elements != m * n) return -1;

    std::vector<int64_t> output_dims = { m, n };
    std::string key =
        "matmul_add:" +
        tensor_dims_key(left->dims) + ":" +
        tensor_dims_key(right->dims) + ":" +
        tensor_dims_key(bias->dims) + ":" +
        std::to_string((int)apply_gelu);
    PJRT_LoadedExecutable* executable = tensor_compile(
        key, tensor_build_matmul_add(left, right, bias, apply_gelu)
    );
    XLA_Tensor* result = tensor_execute(executable, { left, right, bias }, output_dims);

    return tensor_finish_output(result, output);
}

} // extern "C"
