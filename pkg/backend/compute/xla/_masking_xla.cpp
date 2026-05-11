// XLA masking backend — PJRT C API implementation.
//
// CausalMask: uses stablehlo.iota + stablehlo.compare to build a triangular
//             boolean mask, then select between 0.0 and -Inf.
// ApplyMask:  stablehlo.add (elementwise).

#include "masking.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <string>

#include "xla/pjrt/c/pjrt_c_api.h"
#include <dlfcn.h>

// ---------------------------------------------------------------------------
// Globals (separate from activation.cc to allow standalone use)
// ---------------------------------------------------------------------------

static const PJRT_Api*  g_mask_api    = nullptr;
static PJRT_Client*     g_mask_client = nullptr;

static std::unordered_map<std::string, PJRT_LoadedExecutable*> g_mask_execs;

// ---------------------------------------------------------------------------
// Helpers (duplicated from activation_xla.cc to keep files independent)
// ---------------------------------------------------------------------------

static void gm_free_error(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
}

static bool gm_check(const PJRT_Api* api, PJRT_Error* err, const char* ctx) {
    if (!err) return true;
    PJRT_Error_Message_Args ma{};
    ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    ma.error = err;
    api->PJRT_Error_Message(&ma);
    fprintf(stderr, "XLA masking PJRT error in %s: %.*s\n",
            ctx, (int)ma.message_size, ma.message);
    gm_free_error(api, err);
    return false;
}

typedef const PJRT_Api* (*GetPjrtApiFn)();

static const PJRT_Api* gm_load_plugin(const char* platform) {
    std::string path = pjrt_plugin_path(platform);
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "XLA masking: failed to dlopen %s: %s\n", path.c_str(), dlerror());
        return nullptr;
    }
    auto get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) return nullptr;
    return get_api();
}

static PJRT_LoadedExecutable* gm_compile(const std::string& mlir_text) {
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = const_cast<char*>(mlir_text.c_str());
    prog.code_size   = mlir_text.size();
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client      = g_mask_client;
    ca.program      = &prog;
    set_single_device_compile_options(&ca);
    PJRT_Error* err = g_mask_api->PJRT_Client_Compile(&ca);
    if (!gm_check(g_mask_api, err, "gm_compile")) return nullptr;
    return ca.executable;
}

static int gm_run(
    PJRT_LoadedExecutable* exec,
    const double* src, int src_n,
    double* dst,       int dst_n)
{
    PJRT_Client_BufferFromHostBuffer_Args ba{};
    ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client      = g_mask_client;
    ba.data        = src;
    ba.type        = PJRT_Buffer_Type_F64;
    int64_t dims[1] = { (int64_t)src_n };
    ba.dims        = dims;
    ba.num_dims    = 1;
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;

    PJRT_Error* err = g_mask_api->PJRT_Client_BufferFromHostBuffer(&ba);
    if (!gm_check(g_mask_api, err, "BufferFromHostBuffer")) return -1;
    PJRT_Buffer* in_buf = ba.buffer;

    {
        PJRT_Buffer_ReadyEvent_Args re{};
        re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        re.buffer = in_buf;
        err = g_mask_api->PJRT_Buffer_ReadyEvent(&re);
        if (!gm_check(g_mask_api, err, "ReadyEvent(in)")) return -1;
        PJRT_Event_Await_Args ea{};
        ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ea.event = re.event;
        err = g_mask_api->PJRT_Event_Await(&ea);
        gm_check(g_mask_api, err, "Event_Await(in)");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = re.event;
        g_mask_api->PJRT_Event_Destroy(&eda);
    }

    PJRT_Buffer*  in_list[1]  = { in_buf };
    PJRT_Buffer** out_list[1] = { nullptr };
    PJRT_Buffer*  out_buf_storage = nullptr;
    out_list[0] = &out_buf_storage;

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size     = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable      = exec;
    ea.argument_lists  = (PJRT_Buffer***)&in_list;
    ea.num_devices     = 1;
    ea.num_args        = 1;
    ea.output_lists    = out_list;
    PJRT_ExecuteOptions options = single_device_execute_options();
    ea.options = &options;
    err = g_mask_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!gm_check(g_mask_api, err, "Execute")) return -1;

    PJRT_Buffer* out_buf = out_buf_storage;

    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_buf;
    tha.dst         = dst;
    tha.dst_size    = (size_t)dst_n * sizeof(double);
    err = g_mask_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (!gm_check(g_mask_api, err, "ToHostBuffer")) return -1;

    {
        PJRT_Event_Await_Args ev{};
        ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ev.event = tha.event;
        err = g_mask_api->PJRT_Event_Await(&ev);
        gm_check(g_mask_api, err, "Event_Await(out)");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = tha.event;
        g_mask_api->PJRT_Event_Destroy(&eda);
    }

    auto destroy_buf = [&](PJRT_Buffer* b) {
        PJRT_Buffer_Destroy_Args da{};
        da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        da.buffer = b;
        g_mask_api->PJRT_Buffer_Destroy(&da);
    };
    destroy_buf(in_buf);
    destroy_buf(out_buf);
    return 0;
}

// ---------------------------------------------------------------------------
// StableHLO module builders
// ---------------------------------------------------------------------------

// Causal mask: builds an [N, N] output (flattened to [N*N]) where
// element [i,j] = 0.0 if j<=i else -Inf.
//
// Strategy:
//   1. iota over rows dimension -> row_idx [N,N]
//   2. iota over cols dimension -> col_idx [N,N]
//   3. compare: col_idx <= row_idx  -> bool mask [N,N]
//   4. select: mask ? 0.0 : -Inf   -> f64 [N,N]
//   5. reshape to [N*N]
static std::string build_causal_mask(int n) {
    std::string nn = std::to_string(n);
    std::string n2 = std::to_string(n * n);
    // -Inf for f64: use a very large negative constant (XLA approximation)
    // stablehlo.constant does not accept special float literals directly;
    // we use the bit pattern via an integer constant converted via bitcast.
    // For simplicity use -1.7976931348623157e+308 (close to -DBL_MAX) or
    // compute via multiply: -1.0 * Inf.  XLA supports infinity literals in
    // dense attributes as "0xFF..." hex.  Use a large finite negative instead
    // of true -Inf to remain portable: -3.4028234663852886e+38 (approx -FLT_MAX
    // as double) — callers treat this as -Inf for attention purposes.
    // Alternatively express as -(1e38 * 1e38) but that overflows.
    // We use the standard approach: constant dense<0xFF...> with type f64.
    // StableHLO supports hex float literals.
    std::string t2d = "tensor<" + nn + "x" + nn + "xf64>";
    std::string t2i = "tensor<" + nn + "x" + nn + "xi64>";
    std::string tb  = "tensor<" + nn + "x" + nn + "xi1>";
    std::string tf  = "tensor<" + n2 + "xf64>";
    return
        "module @causal_mask {\n"
        "  func.func @main() -> " + tf + " {\n"
        "    %row = stablehlo.iota dim = 0 : " + t2d + "\n"
        "    %col = stablehlo.iota dim = 1 : " + t2d + "\n"
        "    %mask = stablehlo.compare LE, %col, %row, TOTALORDER : (" + t2d + ", " + t2d + ") -> " + tb + "\n"
        "    %zero = stablehlo.constant dense<0.0> : " + t2d + "\n"
        "    %ninf = stablehlo.constant dense<-1.7976931348623157e+308> : " + t2d + "\n"
        "    %out2d = stablehlo.select %mask, %zero, %ninf : " + tb + ", " + t2d + ", " + t2d + "\n"
        "    %out = stablehlo.reshape %out2d : (" + t2d + ") -> " + tf + "\n"
        "    return %out : " + tf + "\n"
        "  }\n"
        "}\n";
}

static std::string build_apply_mask(int n) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    // Input is a flattened [2*n] tensor: first n = scores, next n = mask.
    std::string t2 = "tensor<" + std::to_string(2 * n) + "xf64>";
    return
        "module @apply_mask {\n"
        "  func.func @main(%arg0: " + t2 + ") -> " + t + " {\n"
        "    %scores = stablehlo.slice %arg0 [0:" + std::to_string(n) + "] : (" + t2 + ") -> " + t + "\n"
        "    %mask   = stablehlo.slice %arg0 [" + std::to_string(n) + ":" + std::to_string(2 * n) + "] : (" + t2 + ") -> " + t + "\n"
        "    %out    = stablehlo.add %scores, %mask : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_masking_init(const char* platform) {
    g_mask_api = gm_load_plugin(platform);
    if (!g_mask_api) return -1;

    PJRT_Client_Create_Args ca{};
    ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* err = g_mask_api->PJRT_Client_Create(&ca);
    if (!gm_check(g_mask_api, err, "PJRT_Client_Create")) return -1;
    g_mask_client = ca.client;
    return 0;
}

int xla_causal_mask(double* out, int seq_len) {
    if (!g_mask_client || !out) return -1;
    int n2 = seq_len * seq_len;

    // CausalMask takes no inputs — we pass a dummy 1-element input.
    // The module @main has no arguments but PJRT execute needs >=0 inputs.
    // We compile/cache keyed on seq_len.
    std::string key = "causal_mask_" + std::to_string(seq_len);
    if (!g_mask_execs.count(key)) {
        auto* exec = gm_compile(build_causal_mask(seq_len));
        if (!exec) return -1;
        g_mask_execs[key] = exec;
    }

    // Execute with no input buffer: use a dummy empty double.
    // We need to call PJRT with num_args=0, but the run_executable helper
    // requires src. Use a workaround: call directly.
    PJRT_LoadedExecutable* exec = g_mask_execs[key];

    // No-input execute
    PJRT_Buffer** no_inputs[1]  = { nullptr };
    PJRT_Buffer** out_list[1]   = { nullptr };
    PJRT_Buffer*  out_buf_store = nullptr;
    out_list[0] = &out_buf_store;

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size     = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable      = exec;
    ea.argument_lists  = (PJRT_Buffer***)no_inputs;
    ea.num_devices     = 1;
    ea.num_args        = 0;
    ea.output_lists    = out_list;
    PJRT_ExecuteOptions options = single_device_execute_options();
    ea.options = &options;
    PJRT_Error* err    = g_mask_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!gm_check(g_mask_api, err, "xla_causal_mask Execute")) return -1;

    PJRT_Buffer* out_buf = out_buf_store;
    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_buf;
    tha.dst         = out;
    tha.dst_size    = (size_t)n2 * sizeof(double);
    err = g_mask_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (!gm_check(g_mask_api, err, "xla_causal_mask ToHostBuffer")) return -1;

    {
        PJRT_Event_Await_Args ev{};
        ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ev.event = tha.event;
        err = g_mask_api->PJRT_Event_Await(&ev);
        gm_check(g_mask_api, err, "Event_Await(causal_mask)");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = tha.event;
        g_mask_api->PJRT_Event_Destroy(&eda);
    }
    {
        PJRT_Buffer_Destroy_Args da{};
        da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        da.buffer = out_buf;
        g_mask_api->PJRT_Buffer_Destroy(&da);
    }
    return 0;
}

int xla_apply_mask(const double* scores, const double* mask, double* out, int n) {
    if (!g_mask_client) return -1;
    if (!scores || !mask || !out) return -1;

    std::string key = "apply_mask_" + std::to_string(n);
    if (!g_mask_execs.count(key)) {
        auto* exec = gm_compile(build_apply_mask(n));
        if (!exec) return -1;
        g_mask_execs[key] = exec;
    }

    // Pack scores and mask into a single [2*n] buffer for the module.
    std::vector<double> combined(2 * n);
    memcpy(combined.data(),     scores, (size_t)n * sizeof(double));
    memcpy(combined.data() + n, mask,   (size_t)n * sizeof(double));

    return gm_run(g_mask_execs[key], combined.data(), 2 * n, out, n);
}

void xla_masking_shutdown(void) {
    for (auto& kv : g_mask_execs) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = kv.second;
        g_mask_api->PJRT_LoadedExecutable_Destroy(&da);
    }
    g_mask_execs.clear();
    if (g_mask_client) {
        PJRT_Client_Destroy_Args da{};
        da.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        da.client      = g_mask_client;
        g_mask_api->PJRT_Client_Destroy(&da);
        g_mask_client = nullptr;
    }
}

} // extern "C"
