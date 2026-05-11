// XLA VSA backend — PJRT C API implementation.
//
// Implements bind (elementwise multiply), bundle (sum + L2-normalise),
// and similarity (dot product) for FHRR-style VSA hypervectors.
//
// Each operation is expressed as a StableHLO text module, compiled once via
// PJRT_Client_Compile and cached for reuse.

#include "xla_vsa.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>

#include "xla/pjrt/c/pjrt_c_api.h"

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static const PJRT_Api*    gv_api    = nullptr;
static PJRT_Client*       gv_client = nullptr;

static std::unordered_map<std::string, PJRT_LoadedExecutable*> gv_execs;

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

static void vfree_error(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
}

static bool vcheck(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return true;
    vfree_error(api, err);
    return false;
}

// ---------------------------------------------------------------------------
// PJRT plugin path helper (matches _math_xla.cpp convention)
// ---------------------------------------------------------------------------

static std::string pjrt_plugin_path(const char* platform) {
    std::string p(platform);
    if (p == "cpu") return "pjrt_c_api_cpu_plugin.so";
    if (p == "gpu") return "pjrt_c_api_gpu_plugin.so";
    return std::string("pjrt_c_api_") + p + "_plugin.so";
}

// ---------------------------------------------------------------------------
// Execute options helpers
// ---------------------------------------------------------------------------

static PJRT_ExecuteOptions single_device_execute_options() {
    PJRT_ExecuteOptions opts{};
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    return opts;
}

static void set_single_device_compile_options(PJRT_Client_Compile_Args* ca) {
    ca->compile_options       = nullptr;
    ca->compile_options_size  = 0;
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

static PJRT_Buffer* host_to_device(const double* ptr, size_t n) {
    PJRT_Client_BufferFromHostBuffer_Args ba{};
    ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client   = gv_client;
    ba.data     = ptr;
    ba.type     = PJRT_Buffer_Type_F64;
    ba.num_dims = 1;
    int64_t dims[1] = { (int64_t)n };
    ba.dims    = dims;
    ba.byte_strides = nullptr;
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    ba.device_layout = nullptr;
    ba.device = nullptr;

    auto* err = gv_api->PJRT_Client_BufferFromHostBuffer(&ba);
    if (!vcheck(gv_api, err)) return nullptr;

    PJRT_Buffer_ReadyEvent_Args re{};
    re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    re.buffer = ba.buffer;
    gv_api->PJRT_Buffer_ReadyEvent(&re);

    PJRT_Event_Await_Args ea{};
    ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea.event = re.event;
    gv_api->PJRT_Event_Await(&ea);

    PJRT_Event_Destroy_Args ed{};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = re.event;
    gv_api->PJRT_Event_Destroy(&ed);

    return ba.buffer;
}

static int device_to_host(PJRT_Buffer* buf, double* out, size_t bytes) {
    PJRT_Buffer_ToHostBuffer_Args ca{};
    ca.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    ca.src = buf;
    ca.dst = out;
    ca.dst_size = bytes;
    ca.host_layout = nullptr;

    auto* err = gv_api->PJRT_Buffer_ToHostBuffer(&ca);
    if (!vcheck(gv_api, err)) return -1;

    PJRT_Event_Await_Args ea{};
    ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea.event = ca.event;
    gv_api->PJRT_Event_Await(&ea);

    PJRT_Event_Destroy_Args ed{};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = ca.event;
    gv_api->PJRT_Event_Destroy(&ed);

    return 0;
}

static void destroy_buf(PJRT_Buffer* buf) {
    if (!buf) return;
    PJRT_Buffer_Destroy_Args bd{};
    bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    bd.buffer = buf;
    gv_api->PJRT_Buffer_Destroy(&bd);
}

// ---------------------------------------------------------------------------
// Module compile + cache
// ---------------------------------------------------------------------------

static PJRT_LoadedExecutable* compile_module(
    const std::string& key, const std::string& mlir)
{
    auto it = gv_execs.find(key);
    if (it != gv_execs.end()) return it->second;

    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = const_cast<char*>(mlir.c_str());
    prog.code_size   = mlir.size();
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client      = gv_client;
    ca.program     = &prog;
    set_single_device_compile_options(&ca);

    auto* err = gv_api->PJRT_Client_Compile(&ca);
    if (!vcheck(gv_api, err)) return nullptr;

    gv_execs[key] = ca.executable;
    return ca.executable;
}

// ---------------------------------------------------------------------------
// StableHLO module builders
// ---------------------------------------------------------------------------

static std::string elementwise_module(const std::string& op, int n) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    return
        "module @m {"
        "  func.func @main(%a: " + t + ", %b: " + t + ") -> " + t + " {"
        "    %r = " + op + " %a, %b : " + t +
        "    return %r : " + t +
        "  }"
        "}";
}

static std::string reduce_sum_module(int n) {
    // Reduces a 1-D tensor of length n to a scalar via stablehlo.reduce.
    std::string tv = "tensor<" + std::to_string(n) + "xf64>";
    std::string ts = "tensor<f64>";
    return
        "module @m {"
        "  func.func @main(%x: " + tv + ") -> " + ts + " {"
        "    %zero = stablehlo.constant dense<0.0> : " + ts +
        "    %r = stablehlo.reduce(%x init: %zero)"
        "      applies stablehlo.add across dimensions = [0] : (" + tv + ", " + ts + ") -> " + ts +
        "    return %r : " + ts +
        "  }"
        "}";
}

static std::string scale_module(int n, double scale) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    std::ostringstream oss;
    oss << std::setprecision(17) << scale;
    std::string sv = oss.str();
    return
        "module @m {"
        "  func.func @main(%x: " + t + ") -> " + t + " {"
        "    %s = stablehlo.constant dense<" + sv + "> : tensor<f64>"
        "    %b = stablehlo.broadcast_in_dim %s, dims=[] : (tensor<f64>) -> " + t +
        "    %r = stablehlo.multiply %x, %b : " + t +
        "    return %r : " + t +
        "  }"
        "}";
}

// ---------------------------------------------------------------------------
// run_binary: execute a 2-input → 1-output compiled executable
// ---------------------------------------------------------------------------

static int run_binary(
    PJRT_LoadedExecutable* exec,
    const double* a_ptr, size_t a_n,
    const double* b_ptr, size_t b_n,
    double* out_ptr, size_t out_bytes)
{
    PJRT_Buffer* ba = host_to_device(a_ptr, a_n);
    if (!ba) return -1;
    PJRT_Buffer* bb = host_to_device(b_ptr, b_n);
    if (!bb) { destroy_buf(ba); return -1; }

    PJRT_Buffer* args[2] = { ba, bb };
    PJRT_Buffer* const* arg_lists[1] = { args };
    PJRT_Buffer* out_buf = nullptr;
    PJRT_Buffer** out_lists[1] = { &out_buf };

    PJRT_LoadedExecutable_Execute_Args xa{};
    xa.struct_size    = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    xa.executable     = exec;
    PJRT_ExecuteOptions opts = single_device_execute_options();
    xa.options        = &opts;
    xa.argument_lists = arg_lists;
    xa.num_devices    = 1;
    xa.num_args       = 2;
    xa.output_lists   = out_lists;
    xa.device_complete_events = nullptr;

    auto* xerr = gv_api->PJRT_LoadedExecutable_Execute(&xa);
    destroy_buf(ba);
    destroy_buf(bb);
    if (!vcheck(gv_api, xerr)) return -1;

    int rc = device_to_host(out_buf, out_ptr, out_bytes);
    destroy_buf(out_buf);
    return rc;
}

// ---------------------------------------------------------------------------
// run_unary: execute a 1-input → 1-output compiled executable
// ---------------------------------------------------------------------------

static int run_unary(
    PJRT_LoadedExecutable* exec,
    const double* in_ptr, size_t in_n,
    double* out_ptr, size_t out_bytes)
{
    PJRT_Buffer* bin = host_to_device(in_ptr, in_n);
    if (!bin) return -1;

    PJRT_Buffer* const* arg_lists[1] = { &bin };
    PJRT_Buffer* out_buf = nullptr;
    PJRT_Buffer** out_lists[1] = { &out_buf };

    PJRT_LoadedExecutable_Execute_Args xa{};
    xa.struct_size    = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    xa.executable     = exec;
    PJRT_ExecuteOptions opts = single_device_execute_options();
    xa.options        = &opts;
    xa.argument_lists = arg_lists;
    xa.num_devices    = 1;
    xa.num_args       = 1;
    xa.output_lists   = out_lists;
    xa.device_complete_events = nullptr;

    auto* xerr = gv_api->PJRT_LoadedExecutable_Execute(&xa);
    destroy_buf(bin);
    if (!vcheck(gv_api, xerr)) return -1;

    int rc = device_to_host(out_buf, out_ptr, out_bytes);
    destroy_buf(out_buf);
    return rc;
}

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_vsa_init(const char* platform) {
    std::string plugin = pjrt_plugin_path(platform);

#ifdef __linux__
    void* handle = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) return -1;
    typedef const PJRT_Api* (*GetPJRTApiFn)();
    auto* get_api = (GetPJRTApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) return -1;
    gv_api = get_api();
#else
    (void)plugin;
    return -1;
#endif

    if (!gv_api) return -1;

    PJRT_Client_Create_Args cca{};
    cca.struct_size    = PJRT_Client_Create_Args_STRUCT_SIZE;
    cca.create_options = nullptr;
    cca.num_options    = 0;

    auto* err = gv_api->PJRT_Client_Create(&cca);
    if (!vcheck(gv_api, err)) return -1;

    gv_client = cca.client;
    return 0;
}

void xla_vsa_shutdown(void) {
    for (auto& kv : gv_execs) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = kv.second;
        gv_api->PJRT_LoadedExecutable_Destroy(&da);
    }
    gv_execs.clear();

    if (gv_client) {
        PJRT_Client_Destroy_Args da{};
        da.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        da.client      = gv_client;
        gv_api->PJRT_Client_Destroy(&da);
        gv_client = nullptr;
    }
}

int xla_vsa_bind(const double* a, const double* b, double* out, int n) {
    if (n <= 0 || !gv_client) return -1;

    std::string key = "vsa_bind_" + std::to_string(n);
    auto* exec = compile_module(key, elementwise_module("stablehlo.multiply", n));
    if (!exec) return -1;

    return run_binary(exec, a, (size_t)n, b, (size_t)n, out, (size_t)n * sizeof(double));
}

int xla_vsa_bundle(const double** vecs, int num_vecs, double* out, int n) {
    if (n <= 0 || num_vecs <= 0 || !gv_client) return -1;

    // Sum all vectors using repeated elementwise add
    std::vector<double> acc(n, 0.0);
    std::string add_key = "vsa_add_" + std::to_string(n);
    auto* add_exec = compile_module(add_key, elementwise_module("stablehlo.add", n));
    if (!add_exec) return -1;

    // First vector
    std::copy(vecs[0], vecs[0] + n, acc.data());

    // Accumulate remaining
    for (int v = 1; v < num_vecs; v++) {
        std::vector<double> tmp(n);
        if (run_binary(add_exec, acc.data(), (size_t)n, vecs[v], (size_t)n,
                       tmp.data(), (size_t)n * sizeof(double)) != 0) return -1;
        acc = std::move(tmp);
    }

    // Compute norm: dot(acc, acc) via multiply then reduce
    std::vector<double> sq(n);
    std::string mul_key = "vsa_mul_" + std::to_string(n);
    auto* mul_exec = compile_module(mul_key, elementwise_module("stablehlo.multiply", n));
    if (!mul_exec) return -1;
    if (run_binary(mul_exec, acc.data(), (size_t)n, acc.data(), (size_t)n,
                   sq.data(), (size_t)n * sizeof(double)) != 0) return -1;

    std::string red_key = "vsa_reduce_sum_" + std::to_string(n);
    auto* red_exec = compile_module(red_key, reduce_sum_module(n));
    if (!red_exec) return -1;

    double sumsq = 0.0;
    if (run_unary(red_exec, sq.data(), (size_t)n, &sumsq, sizeof(double)) != 0) return -1;

    double norm = std::sqrt(sumsq);

    if (norm > 1e-12) {
        double inv_norm = 1.0 / norm;
        std::string sc_key = "vsa_scale_" + std::to_string(n) + "_" +
                             std::to_string(inv_norm);
        auto* sc_exec = compile_module(sc_key, scale_module(n, inv_norm));
        if (!sc_exec) return -1;
        if (run_unary(sc_exec, acc.data(), (size_t)n, out, (size_t)n * sizeof(double)) != 0) return -1;
    } else {
        std::copy(acc.begin(), acc.end(), out);
    }

    return 0;
}

int xla_vsa_similarity(const double* a, const double* b, double* out, int n) {
    if (n <= 0 || !gv_client) return -1;

    // Elementwise multiply then reduce-sum
    std::string mul_key = "vsa_mul_" + std::to_string(n);
    auto* mul_exec = compile_module(mul_key, elementwise_module("stablehlo.multiply", n));
    if (!mul_exec) return -1;

    std::vector<double> prod(n);
    if (run_binary(mul_exec, a, (size_t)n, b, (size_t)n,
                   prod.data(), (size_t)n * sizeof(double)) != 0) return -1;

    std::string red_key = "vsa_reduce_sum_" + std::to_string(n);
    auto* red_exec = compile_module(red_key, reduce_sum_module(n));
    if (!red_exec) return -1;

    return run_unary(red_exec, prod.data(), (size_t)n, out, sizeof(double));
}

} // extern "C"
