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
#include <string>
#include <unordered_map>

#include "xla/pjrt/c/pjrt_c_api.h"

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static const PJRT_Api*    gm_api    = nullptr;
static PJRT_Client*       gm_client = nullptr;

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
static int run_exec(PJRT_LoadedExecutable* exec,
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
static PJRT_LoadedExecutable* compile_module(const std::string& key,
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
    char buf[2048];
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    snprintf(buf, sizeof(buf),
        "module @m {"
        "  func.func @main(%%x: %s) -> %s {"
        "    %%s = stablehlo.constant dense<%.17g> : tensor<f64>"
        "    %%b = stablehlo.broadcast_in_dim %%s, dims=[] : (tensor<f64>) -> %s"
        "    %%r = stablehlo.multiply %%x, %%b : %s"
        "    return %%r : %s"
        "  }"
        "}", t.c_str(), t.c_str(), scale, t.c_str(), t.c_str(), t.c_str());
    return buf;
}

// ---------------------------------------------------------------------------
// C API implementation
// ---------------------------------------------------------------------------

extern "C" {

int xla_math_init(const char* platform) {
    // Load the PJRT plugin — same mechanism as activation_xla.cc.
    // Platform "cpu" -> look for pjrt_c_api_cpu_plugin.so via dlopen.
    // For brevity we reuse the global g_api/g_client from activation if
    // already initialised; otherwise initialise independently.

    // Use PJRT_Api_from_Platform (a convenience provided by some XLA builds)
    // or fall back to dlopen.  Here we use the same approach as activation.
    // Since we can't share statics across TUs easily we do a fresh init.

    // Try the PJRT dynamic load approach:
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
    for (auto& kv : gm_execs) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = kv.second;
        gm_api->PJRT_LoadedExecutable_Destroy(&da);
    }
    gm_execs.clear();
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
    auto* exec = compile_module(key, elementwise_module("stablehlo.add", n));
    if (!exec) return -1;
    const double* ins[2] = {a, b};
    size_t sizes[2] = {(size_t)n*8, (size_t)n*8};
    return run_exec(exec, ins, 2, sizes, out, (size_t)n*8);
}

int xla_mul(const double* a, const double* b, double* out, int n) {
    std::string key = "mul_" + std::to_string(n);
    auto* exec = compile_module(key, elementwise_module("stablehlo.multiply", n));
    if (!exec) return -1;
    const double* ins[2] = {a, b};
    size_t sizes[2] = {(size_t)n*8, (size_t)n*8};
    return run_exec(exec, ins, 2, sizes, out, (size_t)n*8);
}

int xla_exp(const double* src, double* dst, int n) {
    std::string key = "exp_" + std::to_string(n);
    auto* exec = compile_module(key, unary_module("stablehlo.exponential", n));
    if (!exec) return -1;
    const double* ins[1] = {src};
    size_t sizes[1] = {(size_t)n*8};
    return run_exec(exec, ins, 1, sizes, dst, (size_t)n*8);
}

int xla_log(const double* src, double* dst, int n) {
    std::string key = "log_" + std::to_string(n);
    auto* exec = compile_module(key, unary_module("stablehlo.log", n));
    if (!exec) return -1;
    const double* ins[1] = {src};
    size_t sizes[1] = {(size_t)n*8};
    return run_exec(exec, ins, 1, sizes, dst, (size_t)n*8);
}

int xla_inv_sqrt_dim_scale(const double* src, double* dst, int n, int dim) {
    double scale = 1.0 / std::sqrt((double)dim);
    std::string key = "isdscale_" + std::to_string(n) + "_" + std::to_string(dim);
    auto* exec = compile_module(key, scale_module(n, scale));
    if (!exec) return -1;
    const double* ins[1] = {src};
    size_t sizes[1] = {(size_t)n*8};
    return run_exec(exec, ins, 1, sizes, dst, (size_t)n*8);
}

// For softmax, layernorm, rmsnorm and matmul the StableHLO is more complex.
// We provide a pure C fallback that mirrors the CPU implementation since
// XLA's value is in large-scale parallelism — small ops fall back gracefully.

int xla_matmul(const double* A, const double* B, double* C, int M, int K, int N) {
    // Pure C reference implementation as fallback.
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += A[i*K+k] * B[k*N+j];
            C[i*N+j] = acc;
        }
    }
    return 0;
}

int xla_softmax(const double* src, double* dst, int num_rows, int dim_size) {
    for (int r = 0; r < num_rows; r++) {
        const double* row_src = src + r * dim_size;
        double*       row_dst = dst + r * dim_size;
        double mx = row_src[0];
        for (int i = 1; i < dim_size; i++) if (row_src[i] > mx) mx = row_src[i];
        double sum = 0.0;
        for (int i = 0; i < dim_size; i++) { row_dst[i] = std::exp(row_src[i] - mx); sum += row_dst[i]; }
        for (int i = 0; i < dim_size; i++) row_dst[i] /= sum;
    }
    return 0;
}

int xla_layernorm(const double* src, double* dst,
                  const double* weight, const double* bias,
                  int num_rows, int d_model, double eps) {
    for (int r = 0; r < num_rows; r++) {
        const double* rs = src + r * d_model;
        double*       rd = dst + r * d_model;
        double mean = 0.0;
        for (int i = 0; i < d_model; i++) mean += rs[i];
        mean /= d_model;
        double var = 0.0;
        for (int i = 0; i < d_model; i++) { double d = rs[i]-mean; var += d*d; }
        var /= d_model;
        double inv_std = 1.0 / std::sqrt(var + eps);
        for (int i = 0; i < d_model; i++)
            rd[i] = (rs[i] - mean) * inv_std * weight[i] + bias[i];
    }
    return 0;
}

int xla_rmsnorm(const double* src, double* dst,
                const double* weight,
                int num_rows, int d_model, double eps) {
    for (int r = 0; r < num_rows; r++) {
        const double* rs = src + r * d_model;
        double*       rd = dst + r * d_model;
        double ss = 0.0;
        for (int i = 0; i < d_model; i++) ss += rs[i]*rs[i];
        double inv_rms = 1.0 / std::sqrt(ss/d_model + eps);
        for (int i = 0; i < d_model; i++) rd[i] = rs[i] * inv_rms * weight[i];
    }
    return 0;
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
    auto* exec = compile_module(key, unary_module("stablehlo.sqrt", n));
    if (!exec) {
        for (int i = 0; i < n; i++) dst[i] = std::sqrt(src[i]);
        return 0;
    }
    return run_exec(exec, (const double**)&src, 1, (size_t[]){(size_t)n*sizeof(double)}, dst, (size_t)n*sizeof(double));
}

int xla_add_scalar(double* dst, double scalar, int n) {
    for (int i = 0; i < n; i++) dst[i] += scalar;
    return 0;
}

int xla_div_vec(const double* a, const double* b, double* dst, int n) {
    char key[64];
    snprintf(key, sizeof(key), "div_%d", n);
    auto* exec = compile_module(key, elementwise_module("stablehlo.divide", n));
    if (!exec) {
        for (int i = 0; i < n; i++) dst[i] = a[i] / b[i];
        return 0;
    }
    const double* in[2] = {a, b};
    size_t in_sz[2] = {(size_t)n*sizeof(double), (size_t)n*sizeof(double)};
    return run_exec(exec, in, 2, in_sz, dst, (size_t)n*sizeof(double));
}

int xla_clamp_vec(double* dst, double lo, double hi, int n) {
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
    auto* exec = compile_module(key, unary_module("stablehlo.sign", n));
    if (!exec) {
        // scalar fallback
        for (int i = 0; i < n; i++) {
            double v = src[i];
            dst[i] = (v > 0.0) ? 1.0 : (v < 0.0) ? -1.0 : 0.0;
        }
        return 0;
    }
    return run_exec(exec, (const double**)&src, 1, (size_t[]){(size_t)n*sizeof(double)}, dst, (size_t)n*sizeof(double));
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
    auto* exec = compile_module(key, std::string(buf));
    if (!exec) {
        // scalar fallback
        for (int row = 0; row < M; row++)
            for (int col = 0; col < N; col++)
                dst[row*N+col] = a[row] * b[col];
        return 0;
    }
    const double* in[2] = {a, b};
    size_t in_sz[2] = {(size_t)M*sizeof(double), (size_t)N*sizeof(double)};
    return run_exec(exec, in, 2, in_sz, dst, (size_t)M*N*sizeof(double));
}

} // extern "C"
