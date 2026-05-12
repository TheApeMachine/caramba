// XLA math backend — PJRT C API implementation.
//
// Each math operation is expressed as a StableHLO text module, compiled once
// via PJRT_Client_Compile and cached for reuse.
//
// Compile requirements:
//   - openxla/xla headers on the include path
//   - Link against the PJRT plugin shared library for your platform.

#include "xla_math.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#include "xla/pjrt/c/pjrt_c_api.h"

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static const PJRT_Api*    gm_api    = nullptr;
static PJRT_Client*       gm_client = nullptr;

// When true, gm_client/gm_api point at activation's g_api/g_client — do not destroy on shutdown.
static bool               gm_borrowed_activation_client = false;

static std::unordered_map<std::string, PJRT_LoadedExecutable*> gm_execs;

// ---------------------------------------------------------------------------
// Error helpers (mirror activation_xla.cc style)
// ---------------------------------------------------------------------------

static void mfree_error(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
}

static bool mcheck(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return true;
    mfree_error(api, err);
    return false;
}

// ---------------------------------------------------------------------------
// StableHLO helpers
// ---------------------------------------------------------------------------

// Run a precompiled executable with host double* in/out.
// in_ptrs: array of pointers to input buffers, out_ptr: output buffer.
int xla_math_xla_math_run_exec(PJRT_LoadedExecutable* exec,
                    const double** in_ptrs, int num_in, size_t* in_sizes,
                    double* out_ptr, size_t out_size)
{
    // Minimal synchronous host-buffer execution via PJRT.
    // Buffer transfer: create PJRT_Buffer for each input from host memory,
    // execute, copy output back.

    // We use PJRT_Client_BufferFromHostBuffer for each input.
    std::vector<PJRT_Buffer*> input_bufs(num_in);
    for (int i = 0; i < num_in; i++) {
        PJRT_Client_BufferFromHostBuffer_Args ba{};
        ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        ba.client   = gm_client;
        ba.data     = in_ptrs[i];
        ba.type     = PJRT_Buffer_Type_F64;
        ba.num_dims = 1;
        size_t dims[1] = { in_sizes[i] / sizeof(double) };
        ba.dims    = (int64_t*)dims;
        ba.byte_strides = nullptr;
        ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        ba.device_layout = nullptr;
        ba.device = nullptr;  // default device

        auto* err = gm_api->PJRT_Client_BufferFromHostBuffer(&ba);
        if (!mcheck(gm_api, err)) return -1;
        // Wait for transfer
        PJRT_Buffer_ReadyEvent_Args re{};
        re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        re.buffer = ba.buffer;
        gm_api->PJRT_Buffer_ReadyEvent(&re);
        // simple poll-wait: use PJRT_Event_Await
        PJRT_Event_Await_Args ea{};
        ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ea.event = re.event;
        gm_api->PJRT_Event_Await(&ea);
        PJRT_Event_Destroy_Args ed{};
        ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        ed.event = re.event;
        gm_api->PJRT_Event_Destroy(&ed);
        input_bufs[i] = ba.buffer;
    }

    // Execute
    PJRT_LoadedExecutable_Execute_Args xa{};
    xa.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    xa.executable  = exec;
    PJRT_ExecuteOptions options = single_device_execute_options();
    xa.options     = &options;
    PJRT_Buffer* const* arg_lists[1] = { input_bufs.data() };
    xa.argument_lists      = arg_lists;
    xa.num_devices         = 1;
    xa.num_args            = num_in;
    PJRT_Buffer* out_bufs[1] = {};
    PJRT_Buffer** out_lists[1] = { out_bufs };
    xa.output_lists        = out_lists;
    xa.device_complete_events = nullptr;

    auto* xerr = gm_api->PJRT_LoadedExecutable_Execute(&xa);
    if (!mcheck(gm_api, xerr)) {
        for (auto* b : input_bufs) {
            PJRT_Buffer_Destroy_Args bd{};
            bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
            bd.buffer = b;
            gm_api->PJRT_Buffer_Destroy(&bd);
        }
        return -1;
    }

    // Copy output back
    PJRT_Buffer_ToHostBuffer_Args ca{};
    ca.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    ca.src = out_bufs[0];
    ca.dst = out_ptr;
    ca.dst_size = out_size;
    ca.host_layout = nullptr;
    auto* cerr = gm_api->PJRT_Buffer_ToHostBuffer(&ca);
    if (!mcheck(gm_api, cerr)) return -1;
    // await copy event
    PJRT_Event_Await_Args ea2{};
    ea2.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea2.event = ca.event;
    gm_api->PJRT_Event_Await(&ea2);
    PJRT_Event_Destroy_Args ed2{};
    ed2.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed2.event = ca.event;
    gm_api->PJRT_Event_Destroy(&ed2);

    // cleanup buffers
    for (auto* b : input_bufs) {
        PJRT_Buffer_Destroy_Args bd{};
        bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        bd.buffer = b;
        gm_api->PJRT_Buffer_Destroy(&bd);
    }
    {
        PJRT_Buffer_Destroy_Args bd{};
        bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        bd.buffer = out_bufs[0];
        gm_api->PJRT_Buffer_Destroy(&bd);
    }
    return 0;
}

// Compile a StableHLO module and cache under key.
PJRT_LoadedExecutable* xla_math_xla_math_compile_module(const std::string& key,
                                              const std::string& mlir)
{
    auto it = gm_execs.find(key);
    if (it != gm_execs.end()) return it->second;

    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = const_cast<char*>(mlir.c_str());
    prog.code_size   = mlir.size();
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size   = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client        = gm_client;
    ca.program       = &prog;
    set_single_device_compile_options(&ca);

    auto* err = gm_api->PJRT_Client_Compile(&ca);
    if (!mcheck(gm_api, err)) return nullptr;

    gm_execs[key] = ca.executable;
    return ca.executable;
}

static std::string matmul_module(int M, int K, int N) {
    std::string tA = "tensor<" + std::to_string(M) + "x" + std::to_string(K) + "xf64>";
    std::string tB = "tensor<" + std::to_string(K) + "x" + std::to_string(N) + "xf64>";
    std::string tC = "tensor<" + std::to_string(M) + "x" + std::to_string(N) + "xf64>";
    return
        "module @mm {\n"
        "  func.func @main(%a: " + tA + ", %b: " + tB + ") -> " + tC + " {\n"
        "    %c = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : ("
        + tA + ", " + tB + ") -> " + tC + "\n"
        "    return %c : " + tC + "\n"
        "  }\n"
        "}\n";
}

static std::string reduce_sum_module(int n) {
    std::string tv = "tensor<" + std::to_string(n) + "xf64>";
    std::string ts = "tensor<f64>";
    return
        "module @redsum {\n"
        "  func.func @main(%x: " + tv + ") -> " + ts + " {\n"
        "    %zero = stablehlo.constant dense<0.0> : " + ts + "\n"
        "    %r = stablehlo.reduce(%x init: %zero)"
        " applies stablehlo.add across dimensions = [0] : (" + tv + ", " + ts + ") -> " + ts + "\n"
        "    return %r : " + ts + "\n"
        "  }\n"
        "}\n";
}

static std::string subtract_module(int n) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    return
        "module @sub {\n"
        "  func.func @main(%a: " + t + ", %b: " + t + ") -> " + t + " {\n"
        "    %r = stablehlo.subtract %a, %b : " + t + "\n"
        "    return %r : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string add_scalar_broadcast_module(int n, double scalar) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    std::ostringstream oss;
    oss << std::setprecision(17) << std::defaultfloat << scalar;
    return
        "module @adds {\n"
        "  func.func @main(%x: " + t + ") -> " + t + " {\n"
        "    %s = stablehlo.constant dense<" + oss.str() + "> : tensor<f64>\n"
        "    %b = stablehlo.broadcast_in_dim %s, dims = [] : (tensor<f64>) -> " + t + "\n"
        "    %r = stablehlo.add %x, %b : " + t + "\n"
        "    return %r : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string softmax_module(int num_rows, int dim_size) {
    std::string tr = "tensor<" + std::to_string(num_rows) + "xf64>";
    std::string tm = "tensor<" + std::to_string(num_rows) + "x" + std::to_string(dim_size) + "xf64>";
    return
        "module @softmax {\n"
        "  func.func @main(%x: " + tm + ") -> " + tm + " {\n"
        "    %neg_inf = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>\n"
        "    %max = stablehlo.reduce(%x init: %neg_inf) applies stablehlo.maximum across dimensions = [1] : (" + tm + ", tensor<f64>) -> " + tr + "\n"
        "    %max_b = stablehlo.broadcast_in_dim %max, dims = [0] : (" + tr + ") -> " + tm + "\n"
        "    %shifted = stablehlo.subtract %x, %max_b : " + tm + "\n"
        "    %exp = stablehlo.exponential %shifted : " + tm + "\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sum = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : (" + tm + ", tensor<f64>) -> " + tr + "\n"
        "    %sum_b = stablehlo.broadcast_in_dim %sum, dims = [0] : (" + tr + ") -> " + tm + "\n"
        "    %out = stablehlo.divide %exp, %sum_b : " + tm + "\n"
        "    return %out : " + tm + "\n"
        "  }\n"
        "}\n";
}

static std::string layernorm_module(int num_rows, int dim_size, double eps) {
    std::ostringstream eps_ss;
    eps_ss << std::setprecision(17) << std::defaultfloat << eps;
    std::string eps_s = eps_ss.str();

    std::string tr = "tensor<" + std::to_string(num_rows) + "xf64>";
    std::string tm = "tensor<" + std::to_string(num_rows) + "x" + std::to_string(dim_size) + "xf64>";
    std::string tw = "tensor<" + std::to_string(dim_size) + "xf64>";
    std::string td = std::to_string(dim_size);
    std::ostringstream ds;
    ds << std::setprecision(17) << std::defaultfloat << (double)dim_size;
    std::string ds_s = ds.str();

    return
        "module @layernorm {\n"
        "  func.func @main(%x: " + tm + ", %w: " + tw + ", %b: " + tw + ") -> " + tm + " {\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sum = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [1] : (" + tm + ", tensor<f64>) -> " + tr + "\n"
        "    %dim_f = stablehlo.constant dense<" + ds_s + "> : tensor<f64>\n"
        "    %dim_b = stablehlo.broadcast_in_dim %dim_f, dims = [] : (tensor<f64>) -> " + tr + "\n"
        "    %mean = stablehlo.divide %sum, %dim_b : " + tr + "\n"
        "    %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0] : (" + tr + ") -> " + tm + "\n"
        "    %diff = stablehlo.subtract %x, %mean_b : " + tm + "\n"
        "    %sq = stablehlo.multiply %diff, %diff : " + tm + "\n"
        "    %var_sum = stablehlo.reduce(%sq init: %zero) applies stablehlo.add across dimensions = [1] : (" + tm + ", tensor<f64>) -> " + tr + "\n"
        "    %var = stablehlo.divide %var_sum, %dim_b : " + tr + "\n"
        "    %eps_val = stablehlo.constant dense<" + eps_s + "> : tensor<f64>\n"
        "    %eps_b = stablehlo.broadcast_in_dim %eps_val, dims = [] : (tensor<f64>) -> " + tr + "\n"
        "    %var_eps = stablehlo.add %var, %eps_b : " + tr + "\n"
        "    %std = stablehlo.sqrt %var_eps : " + tr + "\n"
        "    %std_b = stablehlo.broadcast_in_dim %std, dims = [0] : (" + tr + ") -> " + tm + "\n"
        "    %norm = stablehlo.divide %diff, %std_b : " + tm + "\n"
        "    %w_b = stablehlo.broadcast_in_dim %w, dims = [1] : (" + tw + ") -> " + tm + "\n"
        "    %b_b = stablehlo.broadcast_in_dim %b, dims = [1] : (" + tw + ") -> " + tm + "\n"
        "    %scaled = stablehlo.multiply %norm, %w_b : " + tm + "\n"
        "    %out = stablehlo.add %scaled, %b_b : " + tm + "\n"
        "    return %out : " + tm + "\n"
        "  }\n"
        "}\n";
}

static std::string rmsnorm_module(int num_rows, int dim_size, double eps) {
    std::ostringstream eps_ss;
    eps_ss << std::setprecision(17) << std::defaultfloat << eps;
    std::string eps_s = eps_ss.str();

    std::string tr = "tensor<" + std::to_string(num_rows) + "xf64>";
    std::string tm = "tensor<" + std::to_string(num_rows) + "x" + std::to_string(dim_size) + "xf64>";
    std::string tw = "tensor<" + std::to_string(dim_size) + "xf64>";
    std::ostringstream ds;
    ds << std::setprecision(17) << std::defaultfloat << (double)dim_size;
    std::string ds_s = ds.str();

    return
        "module @rmsnorm {\n"
        "  func.func @main(%x: " + tm + ", %w: " + tw + ") -> " + tm + " {\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sq = stablehlo.multiply %x, %x : " + tm + "\n"
        "    %ss = stablehlo.reduce(%sq init: %zero) applies stablehlo.add across dimensions = [1] : (" + tm + ", tensor<f64>) -> " + tr + "\n"
        "    %dim_f = stablehlo.constant dense<" + ds_s + "> : tensor<f64>\n"
        "    %dim_b = stablehlo.broadcast_in_dim %dim_f, dims = [] : (tensor<f64>) -> " + tr + "\n"
        "    %mean_ss = stablehlo.divide %ss, %dim_b : " + tr + "\n"
        "    %eps_val = stablehlo.constant dense<" + eps_s + "> : tensor<f64>\n"
        "    %eps_b = stablehlo.broadcast_in_dim %eps_val, dims = [] : (tensor<f64>) -> " + tr + "\n"
        "    %var_eps = stablehlo.add %mean_ss, %eps_b : " + tr + "\n"
        "    %rms = stablehlo.sqrt %var_eps : " + tr + "\n"
        "    %rms_b = stablehlo.broadcast_in_dim %rms, dims = [0] : (" + tr + ") -> " + tm + "\n"
        "    %norm = stablehlo.divide %x, %rms_b : " + tm + "\n"
        "    %w_b = stablehlo.broadcast_in_dim %w, dims = [1] : (" + tw + ") -> " + tm + "\n"
        "    %out = stablehlo.multiply %norm, %w_b : " + tm + "\n"
        "    return %out : " + tm + "\n"
        "  }\n"
        "}\n";
}

// ---------------------------------------------------------------------------
// StableHLO module templates
// ---------------------------------------------------------------------------

static std::string elementwise_module(const std::string& op, int n) {
    // op: stablehlo op name like "stablehlo.add", "stablehlo.multiply", etc.
    std::string t = "tensor<" + std::to_string(n) + "xf64>";

    return
        "module @m {"
        "  func.func @main(%a: " + t + ", %b: " + t + ") -> " + t + " {"
        "    %r = " + op + " %a, %b : " + t +
        "    return %r : " + t +
        "  }"
        "}";
}

static std::string unary_module(const std::string& op, int n) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";

    return
        "module @m {"
        "  func.func @main(%x: " + t + ") -> " + t + " {"
        "    %r = " + op + " %x : " + t +
        "    return %r : " + t +
        "  }"
        "}";
}

static std::string scale_module(int n, double scale) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    std::ostringstream scale_stream;
    scale_stream << std::setprecision(17) << std::defaultfloat << scale;
    std::string scale_literal = scale_stream.str();

    return
        "module @m {"
        "  func.func @main(%x: " + t + ") -> " + t + " {"
        "    %s = stablehlo.constant dense<" + scale_literal + "> : tensor<f64>"
        "    %b = stablehlo.broadcast_in_dim %s, dims=[] : (tensor<f64>) -> " + t +
        "    %r = stablehlo.multiply %x, %b : " + t +
        "    return %r : " + t +
        "  }"
        "}";
}

// ---------------------------------------------------------------------------
// C API implementation
// ---------------------------------------------------------------------------

extern "C" {

int xla_math_init(const char* platform) {
    if (g_client) {
        gm_api    = g_api;
        gm_client = g_client;
        gm_borrowed_activation_client = true;
        return 0;
    }

    gm_borrowed_activation_client = false;

    // Stand-alone math client (only when xla_init was not used).
    std::string plugin = pjrt_plugin_path(platform);

#ifdef __linux__
    void* handle = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) return -1;
    typedef const PJRT_Api* (*GetPJRTApiFn)();
    auto* get_api = (GetPJRTApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) return -1;
    gm_api = get_api();
#else
    (void)plugin;
    return -1;  // not supported on this platform
#endif

    if (!gm_api) return -1;

    PJRT_Client_Create_Args cca{};
    cca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    cca.create_options = nullptr;
    cca.num_options = 0;
    auto* err = gm_api->PJRT_Client_Create(&cca);
    if (!mcheck(gm_api, err)) return -1;
    gm_client = cca.client;
    return 0;
}

void xla_math_shutdown(void) {
    if (gm_api) {
        for (auto& kv : gm_execs) {
            PJRT_LoadedExecutable_Destroy_Args da{};
            da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
            da.executable  = kv.second;
            gm_api->PJRT_LoadedExecutable_Destroy(&da);
        }
    }
    gm_execs.clear();

    if (gm_borrowed_activation_client) {
        gm_api    = nullptr;
        gm_client = nullptr;
        gm_borrowed_activation_client = false;
        return;
    }

    if (gm_client) {
        PJRT_Client_Destroy_Args da{};
        da.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        da.client = gm_client;
        gm_api->PJRT_Client_Destroy(&da);
        gm_client = nullptr;
    }
}

int xla_add(const double* a, const double* b, double* out, int n) {
    std::string key = "add_" + std::to_string(n);
    auto* exec = xla_math_compile_module(key, elementwise_module("stablehlo.add", n));
    if (!exec) return -1;
    const double* ins[2] = {a, b};
    size_t sizes[2] = {(size_t)n*8, (size_t)n*8};
    return xla_math_run_exec(exec, ins, 2, sizes, out, (size_t)n*8);
}

int xla_mul(const double* a, const double* b, double* out, int n) {
    std::string key = "mul_" + std::to_string(n);
    auto* exec = xla_math_compile_module(key, elementwise_module("stablehlo.multiply", n));
    if (!exec) return -1;
    const double* ins[2] = {a, b};
    size_t sizes[2] = {(size_t)n*8, (size_t)n*8};
    return xla_math_run_exec(exec, ins, 2, sizes, out, (size_t)n*8);
}

int xla_exp(const double* src, double* dst, int n) {
    std::string key = "exp_" + std::to_string(n);
    auto* exec = xla_math_compile_module(key, unary_module("stablehlo.exponential", n));
    if (!exec) return -1;
    const double* ins[1] = {src};
    size_t sizes[1] = {(size_t)n*8};
    return xla_math_run_exec(exec, ins, 1, sizes, dst, (size_t)n*8);
}

int xla_log(const double* src, double* dst, int n) {
    std::string key = "log_" + std::to_string(n);
    auto* exec = xla_math_compile_module(key, unary_module("stablehlo.log", n));
    if (!exec) return -1;
    const double* ins[1] = {src};
    size_t sizes[1] = {(size_t)n*8};
    return xla_math_run_exec(exec, ins, 1, sizes, dst, (size_t)n*8);
}

int xla_inv_sqrt_dim_scale(const double* src, double* dst, int n, int dim) {
    if (dim <= 0) return -1;

    double scale = 1.0 / std::sqrt((double)dim);
    std::string key = "isdscale_" + std::to_string(n) + "_" + std::to_string(dim);
    auto* exec = xla_math_compile_module(key, scale_module(n, scale));
    if (!exec) return -1;
    const double* ins[1] = {src};
    size_t sizes[1] = {(size_t)n*8};
    return xla_math_run_exec(exec, ins, 1, sizes, dst, (size_t)n*8);
}

// Softmax, layernorm, and rmsnorm still use host loops; matmul/diff/reductions use StableHLO.

int xla_matmul(const double* A, const double* B, double* C, int M, int K, int N) {
    if (M <= 0 || K <= 0 || N <= 0) return -1;
    if (!gm_client) return -1;

    std::string key = "mm_" + std::to_string(M) + "_" + std::to_string(K) + "_" + std::to_string(N);
    PJRT_LoadedExecutable* exec = xla_math_compile_module(key, matmul_module(M, K, N));
    if (!exec) return -1;

    const double* ins[2] = {A, B};
    size_t sizes[2] = {
        (size_t)M * (size_t)K * sizeof(double),
        (size_t)K * (size_t)N * sizeof(double),
    };

    return xla_math_run_exec(exec, ins, 2, sizes, C, (size_t)M * (size_t)N * sizeof(double));
}

int xla_reduce_sum(const double* x, double* out_scalar, int n) {
    if (n <= 0 || !out_scalar) return -1;
    if (!gm_client) return -1;

    std::string key = "redsum_" + std::to_string(n);
    PJRT_LoadedExecutable* exec = xla_math_compile_module(key, reduce_sum_module(n));
    if (!exec) return -1;

    const double* ins[1] = {x};
    size_t sizes[1] = {(size_t)n * sizeof(double)};

    return xla_math_run_exec(exec, ins, 1, sizes, out_scalar, sizeof(double));
}

int xla_sub(const double* a, const double* b, double* out, int n) {
    if (n <= 0 || !gm_client) return -1;

    std::string key = "sub_" + std::to_string(n);
    PJRT_LoadedExecutable* exec = xla_math_compile_module(key, subtract_module(n));
    if (!exec) return -1;

    const double* ins[2] = {a, b};
    size_t sizes[2] = {(size_t)n * sizeof(double), (size_t)n * sizeof(double)};

    return xla_math_run_exec(exec, ins, 2, sizes, out, (size_t)n * sizeof(double));
}

int xla_softmax(const double* src, double* dst, int num_rows, int dim_size) {
    if (num_rows <= 0 || dim_size <= 0) return -1;
    if (!gm_client) return -1;

    std::string key = "smax_" + std::to_string(num_rows) + "_" + std::to_string(dim_size);
    PJRT_LoadedExecutable* exec = xla_math_compile_module(key, softmax_module(num_rows, dim_size));
    if (!exec) return -1;

    const double* ins[1] = {src};
    size_t sizes[1] = {(size_t)num_rows * (size_t)dim_size * sizeof(double)};
    return xla_math_run_exec(exec, ins, 1, sizes, dst, sizes[0]);
}

int xla_layernorm(const double* src, double* dst,
                  const double* weight, const double* bias,
                  int num_rows, int d_model, double eps) {
    if (num_rows <= 0 || d_model <= 0) return -1;
    if (!gm_client) return -1;

    std::ostringstream eps_ss;
    eps_ss << std::setprecision(17) << std::defaultfloat << eps;
    std::string key = "lnorm_" + std::to_string(num_rows) + "_" + std::to_string(d_model) + "_" + eps_ss.str();
    PJRT_LoadedExecutable* exec = xla_math_compile_module(key, layernorm_module(num_rows, d_model, eps));
    if (!exec) return -1;

    const double* ins[3] = {src, weight, bias};
    size_t sizes[3] = {
        (size_t)num_rows * (size_t)d_model * sizeof(double),
        (size_t)d_model * sizeof(double),
        (size_t)d_model * sizeof(double)
    };
    return xla_math_run_exec(exec, ins, 3, sizes, dst, sizes[0]);
}

int xla_rmsnorm(const double* src, double* dst,
                const double* weight,
                int num_rows, int d_model, double eps) {
    if (num_rows <= 0 || d_model <= 0) return -1;
    if (!gm_client) return -1;

    std::ostringstream eps_ss;
    eps_ss << std::setprecision(17) << std::defaultfloat << eps;
    std::string key = "rmsnorm_" + std::to_string(num_rows) + "_" + std::to_string(d_model) + "_" + eps_ss.str();
    PJRT_LoadedExecutable* exec = xla_math_compile_module(key, rmsnorm_module(num_rows, d_model, eps));
    if (!exec) return -1;

    const double* ins[2] = {src, weight};
    size_t sizes[2] = {
        (size_t)num_rows * (size_t)d_model * sizeof(double),
        (size_t)d_model * sizeof(double)
    };
    return xla_math_run_exec(exec, ins, 2, sizes, dst, sizes[0]);
}

// ---------------------------------------------------------------------------
// Optimizer primitives — scalar fallbacks; StableHLO variants follow same
// pattern as unary/elementwise_module above.
// ---------------------------------------------------------------------------

int xla_axpy(double* dst, const double* src, double scale, int n) {
    for (int i = 0; i < n; i++) dst[i] += scale * src[i];
    return 0;
}

int xla_scale(double* dst, double s, int n) {
    for (int i = 0; i < n; i++) dst[i] *= s;
    return 0;
}

int xla_sqrt_vec(const double* src, double* dst, int n) {
    char key[64];
    snprintf(key, sizeof(key), "sqrt_%d", n);
    auto* exec = xla_math_compile_module(key, unary_module("stablehlo.sqrt", n));
    if (!exec) {
        for (int i = 0; i < n; i++) dst[i] = std::sqrt(src[i]);
        return 0;
    }
    return xla_math_run_exec(exec, (const double**)&src, 1, (size_t[]){(size_t)n*sizeof(double)}, dst, (size_t)n*sizeof(double));
}

int xla_add_scalar(double* dst, double scalar, int n) {
    if (n <= 0) return -1;

    if (!gm_client) {
        for (int i = 0; i < n; i++) {
            dst[i] += scalar;
        }
        return 0;
    }

    std::ostringstream scalar_key;
    scalar_key << std::setprecision(17) << std::defaultfloat << scalar;

    std::string key = "adds_ip_" + std::to_string(n) + "_" + scalar_key.str();
    PJRT_LoadedExecutable* exec = xla_math_compile_module(key, add_scalar_broadcast_module(n, scalar));
    if (!exec) return -1;

    std::vector<double> tmp((size_t)n);
    std::memcpy(tmp.data(), dst, (size_t)n * sizeof(double));

    const double* ins[1] = {tmp.data()};
    size_t sizes[1] = {(size_t)n * sizeof(double)};

    return xla_math_run_exec(exec, ins, 1, sizes, dst, (size_t)n * sizeof(double));
}

int xla_div_vec(const double* a, const double* b, double* dst, int n) {
    char key[64];
    snprintf(key, sizeof(key), "div_%d", n);
    auto* exec = xla_math_compile_module(key, elementwise_module("stablehlo.divide", n));
    if (!exec) {
        for (int i = 0; i < n; i++) dst[i] = a[i] / b[i];
        return 0;
    }
    const double* in[2] = {a, b};
    size_t in_sz[2] = {(size_t)n*sizeof(double), (size_t)n*sizeof(double)};
    return xla_math_run_exec(exec, in, 2, in_sz, dst, (size_t)n*sizeof(double));
}

int xla_clamp_vec(double* dst, double lo, double hi, int n) {
    if (lo > hi) return -1;

    for (int i = 0; i < n; i++) {
        if (dst[i] < lo) dst[i] = lo;
        else if (dst[i] > hi) dst[i] = hi;
    }
    return 0;
}

int xla_sign(const double* src, double* dst, int n) {
    // StableHLO: stablehlo.sign is the standard op for elementwise sign.
    char key[64];
    snprintf(key, sizeof(key), "sign_%d", n);
    auto* exec = xla_math_compile_module(key, unary_module("stablehlo.sign", n));
    if (!exec) {
        // scalar fallback
        for (int i = 0; i < n; i++) {
            double v = src[i];
            dst[i] = (v > 0.0) ? 1.0 : (v < 0.0) ? -1.0 : 0.0;
        }
        return 0;
    }
    return xla_math_run_exec(exec, (const double**)&src, 1, (size_t[]){(size_t)n*sizeof(double)}, dst, (size_t)n*sizeof(double));
}

int xla_outer(const double* a, const double* b, double* dst, int M, int N) {
    // StableHLO: broadcast a to [M,N] and b to [M,N], then multiply.
    char key[64];
    snprintf(key, sizeof(key), "outer_%d_%d", M, N);
    char buf[2048];
    std::string a_type = "tensor<" + std::to_string(M) + "xf64>";
    std::string b_type = "tensor<" + std::to_string(N) + "xf64>";
    std::string out_type = "tensor<" + std::to_string(M) + "x" + std::to_string(N) + "xf64>";
    snprintf(buf, sizeof(buf),
        "module @m {"
        "  func.func @main(%%a: %s, %%b: %s) -> %s {"
        "    %%ab = stablehlo.broadcast_in_dim %%a, dims=[0] : (%s) -> %s"
        "    %%bb = stablehlo.broadcast_in_dim %%b, dims=[1] : (%s) -> %s"
        "    %%r  = stablehlo.multiply %%ab, %%bb : %s"
        "    return %%r : %s"
        "  }"
        "}",
        a_type.c_str(), b_type.c_str(), out_type.c_str(),
        a_type.c_str(), out_type.c_str(),
        b_type.c_str(), out_type.c_str(),
        out_type.c_str(), out_type.c_str());
    auto* exec = xla_math_compile_module(key, std::string(buf));
    if (!exec) {
        // scalar fallback
        for (int row = 0; row < M; row++)
            for (int col = 0; col < N; col++)
                dst[row*N+col] = a[row] * b[col];
        return 0;
    }
    const double* in[2] = {a, b};
    size_t in_sz[2] = {(size_t)M*sizeof(double), (size_t)N*sizeof(double)};
    return xla_math_run_exec(exec, in, 2, in_sz, dst, (size_t)M*N*sizeof(double));
}

} // extern "C"
