// XLA activation backend — PJRT C API implementation.
//
// Compile requirements:
//   - openxla/xla headers on the include path
//   - Link against the PJRT plugin shared library for your platform:
//       CPU: pjrt_c_api_cpu_plugin.so
//       GPU: pjrt_c_api_gpu_plugin.so
//
// Each activation is expressed as a StableHLO text module compiled once
// via PJRT_Client_Compile and cached for reuse.

#include "activation.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <string>

// PJRT C API header from the XLA distribution.
// Adjust the include path to match your XLA installation.
#include "xla/pjrt/c/pjrt_c_api.h"

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

const PJRT_Api*        g_api     = nullptr;
PJRT_Client*           g_client  = nullptr;

// one cached dlopen handle for the PJRT plugin (closed in xla_shutdown).
static void* g_plugin_handle = nullptr;

// One cached executable per activation name.
static std::unordered_map<std::string, PJRT_LoadedExecutable*> g_execs;

// Cached element count for which executables were compiled.
static int g_compiled_n = 0;

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

static void free_error(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
}

static bool check(const PJRT_Api* api, PJRT_Error* err, const char* ctx) {
    if (!err) return true;
    PJRT_Error_Message_Args ma{};
    ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    ma.error = err;
    api->PJRT_Error_Message(&ma);
    fprintf(stderr, "XLA PJRT error in %s: %.*s\n",
            ctx, (int)ma.message_size, ma.message);
    free_error(api, err);
    return false;
}

// ---------------------------------------------------------------------------
// Platform plugin loader
// ---------------------------------------------------------------------------

// dlopen-based loader for the PJRT plugin.
#include <dlfcn.h>

typedef const PJRT_Api* (*GetPjrtApiFn)();

static const PJRT_Api* load_pjrt_plugin(const char* platform) {
    char path[256];
    if (strcmp(platform, "gpu") == 0) {
        snprintf(path, sizeof(path), "pjrt_c_api_gpu_plugin.so");
    } else {
        snprintf(path, sizeof(path), "pjrt_c_api_cpu_plugin.so");
    }

    void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "XLA: failed to dlopen %s: %s\n", path, dlerror());
        return nullptr;
    }

    auto get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) {
        fprintf(stderr, "XLA: GetPjrtApi not found in %s\n", path);
        dlclose(handle);
        return nullptr;
    }

    const PJRT_Api* api = get_api();
    if (!api) {
        fprintf(stderr, "XLA: GetPjrtApi returned null\n");
        dlclose(handle);
        return nullptr;
    }

    if (g_plugin_handle) {
        dlclose(g_plugin_handle);
    }
    g_plugin_handle = handle;
    return api;
}

// ---------------------------------------------------------------------------
// Compile a StableHLO module string into a PJRT_LoadedExecutable.
// Returns nullptr on failure.
// ---------------------------------------------------------------------------

static PJRT_LoadedExecutable* compile_stablehlo(const std::string& mlir_text) {
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = mlir_text.c_str();
    prog.code_size   = mlir_text.size();
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client      = g_client;
    ca.program     = &prog;
    // Compile options: leave as default (no serialized options).
    ca.compile_options      = nullptr;
    ca.compile_options_size = 0;

    PJRT_Error* err = g_api->PJRT_Client_Compile(&ca);
    if (!check(g_api, err, "PJRT_Client_Compile")) return nullptr;
    return ca.executable;
}

// ---------------------------------------------------------------------------
// StableHLO module builders
// ---------------------------------------------------------------------------

// Helper: f64 tensor type string for shape [n]
static std::string f64_type(int n) {
    return "tensor<" + std::to_string(n) + "xf64>";
}
static std::string f64_2type(int n) {
    return "tensor<" + std::to_string(2*n) + "xf64>";
}

static std::string build_relu(int n) {
    std::string t = f64_type(n);
    return
        "module @relu {\n"
        "  func.func @main(%arg0: " + t + ") -> " + t + " {\n"
        "    %zero = stablehlo.constant dense<0.0> : " + t + "\n"
        "    %out = stablehlo.maximum %arg0, %zero : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string build_leaky_relu(int n, double alpha) {
    std::string t = f64_type(n);
    char abuf[64];
    snprintf(abuf, sizeof(abuf), "%.17g", alpha);
    return
        "module @leaky_relu {\n"
        "  func.func @main(%arg0: " + t + ") -> " + t + " {\n"
        "    %zero  = stablehlo.constant dense<0.0>   : " + t + "\n"
        "    %alpha = stablehlo.constant dense<" + abuf + "> : " + t + "\n"
        "    %scaled = stablehlo.multiply %arg0, %alpha : " + t + "\n"
        "    %mask   = stablehlo.compare GT, %arg0, %zero, TOTALORDER : ("+t+","+t+") -> tensor<"+std::to_string(n)+"xi1>\n"
        "    %out    = stablehlo.select %mask, %arg0, %scaled : tensor<"+std::to_string(n)+"xi1>, "+t+", "+t+"\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string build_gelu(int n) {
    std::string t = f64_type(n);
    // GELU(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    // Using tanh approximation via stablehlo.tanh
    const double sqrt2pi = 0.7978845608028654;
    char cbuf[64];
    snprintf(cbuf, sizeof(cbuf), "%.17g", sqrt2pi);
    return
        "module @gelu {\n"
        "  func.func @main(%arg0: " + t + ") -> " + t + " {\n"
        "    %half    = stablehlo.constant dense<5.000000e-01>   : " + t + "\n"
        "    %one     = stablehlo.constant dense<1.0>             : " + t + "\n"
        "    %c044    = stablehlo.constant dense<4.471500e-02>   : " + t + "\n"
        "    %sqrt2pi = stablehlo.constant dense<" + cbuf + "> : " + t + "\n"
        "    %x3  = stablehlo.multiply %arg0, %arg0 : " + t + "\n"
        "    %x3b = stablehlo.multiply %x3, %arg0  : " + t + "\n"
        "    %t1  = stablehlo.multiply %c044, %x3b : " + t + "\n"
        "    %t2  = stablehlo.add %arg0, %t1        : " + t + "\n"
        "    %t3  = stablehlo.multiply %sqrt2pi, %t2: " + t + "\n"
        "    %th  = stablehlo.tanh %t3              : " + t + "\n"
        "    %t4  = stablehlo.add %one, %th         : " + t + "\n"
        "    %t5  = stablehlo.multiply %arg0, %t4   : " + t + "\n"
        "    %out = stablehlo.multiply %half, %t5   : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string build_tanh(int n) {
    std::string t = f64_type(n);
    return
        "module @tanh_act {\n"
        "  func.func @main(%arg0: " + t + ") -> " + t + " {\n"
        "    %out = stablehlo.tanh %arg0 : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string build_sigmoid(int n) {
    std::string t = f64_type(n);
    return
        "module @sigmoid {\n"
        "  func.func @main(%arg0: " + t + ") -> " + t + " {\n"
        "    %out = stablehlo.logistic %arg0 : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string build_swiglu(int n) {
    // src layout: [gates(n), values(n)] as flat tensor of 2n.
    // Split, apply logistic to gates, multiply.
    std::string t  = f64_type(n);
    std::string t2 = f64_2type(n);
    return
        "module @swiglu {\n"
        "  func.func @main(%arg0: " + t2 + ") -> " + t + " {\n"
        "    %gates  = stablehlo.slice %arg0 [0:" + std::to_string(n) + "] : (" + t2 + ") -> " + t + "\n"
        "    %values = stablehlo.slice %arg0 [" + std::to_string(n) + ":" + std::to_string(2*n) + "] : (" + t2 + ") -> " + t + "\n"
        "    %sig    = stablehlo.logistic %gates   : " + t + "\n"
        "    %out    = stablehlo.multiply %sig, %values : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

// ---------------------------------------------------------------------------
// Execute a compiled executable: copy host→device, run, copy device→host.
// ---------------------------------------------------------------------------

static int run_executable(
    PJRT_LoadedExecutable* exec,
    const double* src, int src_n,
    double* dst,       int dst_n)
{
    // --- Create input buffer ---
    PJRT_Client_BufferFromHostBuffer_Args ba{};
    ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client      = g_client;
    ba.data        = src;
    ba.type        = PJRT_Buffer_Type_F64;
    int64_t dims[1] = { (int64_t)src_n };
    ba.dims        = dims;
    ba.num_dims    = 1;
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;

    PJRT_Error* err = g_api->PJRT_Client_BufferFromHostBuffer(&ba);
    if (!check(g_api, err, "BufferFromHostBuffer")) return -1;

    PJRT_Buffer* in_buf = ba.buffer;

    auto destroy_buf = [&](PJRT_Buffer* b) {
        if (!b) return;
        PJRT_Buffer_Destroy_Args da{};
        da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        da.buffer = b;
        g_api->PJRT_Buffer_Destroy(&da);
    };

    // Wait for transfer to complete.
    {
        PJRT_Buffer_ReadyEvent_Args re{};
        re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        re.buffer = in_buf;
        err = g_api->PJRT_Buffer_ReadyEvent(&re);
        if (!check(g_api, err, "ReadyEvent(in)")) {
            destroy_buf(in_buf);
            return -1;
        }
        // Wait on the event.
        PJRT_Event_Await_Args ea{};
        ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ea.event = re.event;
        err = g_api->PJRT_Event_Await(&ea);
        check(g_api, err, "Event_Await(in)");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = re.event;
        g_api->PJRT_Event_Destroy(&eda);
    }

    // --- Execute ---
    PJRT_Buffer*  in_arr[1]        = { in_buf };
    PJRT_Buffer** in_lists[1]      = { in_arr };
    PJRT_Buffer*  out_buf_storage  = nullptr;
    PJRT_Buffer** out_list[1]      = { &out_buf_storage };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size        = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable         = exec;
    ea.argument_lists     = in_lists;
    ea.num_devices        = 1;
    ea.num_args           = 1;
    ea.output_lists       = out_list;
    ea.execute_options    = nullptr;

    err = g_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!check(g_api, err, "Execute")) {
        destroy_buf(in_buf);
        return -1;
    }

    PJRT_Buffer* out_buf = out_buf_storage;

    // --- Copy output device→host ---
    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_buf;
    tha.dst         = dst;
    tha.dst_size    = (size_t)dst_n * sizeof(double);

    err = g_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (!check(g_api, err, "ToHostBuffer")) {
        destroy_buf(in_buf);
        destroy_buf(out_buf);
        return -1;
    }

    // Await transfer event.
    {
        PJRT_Event_Await_Args ev{};
        ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ev.event = tha.event;
        err = g_api->PJRT_Event_Await(&ev);
        check(g_api, err, "Event_Await(out)");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = tha.event;
        g_api->PJRT_Event_Destroy(&eda);
    }

    // --- Cleanup buffers ---
    destroy_buf(in_buf);
    destroy_buf(out_buf);

    return 0;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_init(const char* platform) {
    g_api = load_pjrt_plugin(platform);
    if (!g_api) return -1;

    PJRT_Client_Create_Args ca{};
    ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    // No extra create options needed for default CPU/GPU.
    PJRT_Error* err = g_api->PJRT_Client_Create(&ca);
    if (!check(g_api, err, "PJRT_Client_Create")) return -1;

    g_client = ca.client;
    return 0;
}

int xla_compile_activations(int n) {
    if (!g_client) return -1;

    // Destroy old executables if n changed.
    if (g_compiled_n != n) {
        for (auto& kv : g_execs) {
            PJRT_LoadedExecutable_Destroy_Args da{};
            da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
            da.executable  = kv.second;
            g_api->PJRT_LoadedExecutable_Destroy(&da);
        }
        g_execs.clear();
        g_compiled_n = 0;
    }

    struct { const char* name; std::string mlir; } ops[] = {
        { "relu",       build_relu(n)         },
        { "leaky_relu", build_leaky_relu(n, 0.01) }, // default alpha; recompiled per-call if needed
        { "gelu",       build_gelu(n)         },
        { "tanh_act",   build_tanh(n)         },
        { "sigmoid",    build_sigmoid(n)      },
        { "swiglu",     build_swiglu(n)       },
    };

    std::unordered_map<std::string, PJRT_LoadedExecutable*> tmp;
    for (auto& op : ops) {
        PJRT_LoadedExecutable* exec = compile_stablehlo(op.mlir);
        if (!exec) {
            // Destroy anything compiled so far.
            for (auto& kv : tmp) {
                PJRT_LoadedExecutable_Destroy_Args da{};
                da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
                da.executable  = kv.second;
                g_api->PJRT_LoadedExecutable_Destroy(&da);
            }
            return -1;
        }
        tmp[op.name] = exec;
    }

    g_execs = std::move(tmp);
    g_compiled_n = n;
    return 0;
}

int xla_relu(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["relu"], src, n, dst, n);
}

int xla_leaky_relu(const double* src, double* dst, double alpha, int n) {
    // Recompile with correct alpha.
    std::string mlir = build_leaky_relu(n, alpha);
    PJRT_LoadedExecutable* exec = compile_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_executable(exec, src, n, dst, n);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

int xla_gelu(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["gelu"], src, n, dst, n);
}

int xla_tanh_act(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["tanh_act"], src, n, dst, n);
}

int xla_sigmoid(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["sigmoid"], src, n, dst, n);
}

int xla_swiglu(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["swiglu"], src, 2*n, dst, n);
}

void xla_shutdown(void) {
    for (auto& kv : g_execs) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = kv.second;
        g_api->PJRT_LoadedExecutable_Destroy(&da);
    }
    g_execs.clear();

    if (g_client) {
        PJRT_Client_Destroy_Args da{};
        da.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        da.client      = g_client;
        g_api->PJRT_Client_Destroy(&da);
        g_client = nullptr;
    }

    g_compiled_n = 0;
    g_api        = nullptr;

    if (g_plugin_handle) {
        dlclose(g_plugin_handle);
        g_plugin_handle = nullptr;
    }
}

} // extern "C"
