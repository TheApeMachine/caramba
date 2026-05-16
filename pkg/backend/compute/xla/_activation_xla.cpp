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
#include <cstring>
#include <unordered_map>
#include <string>
#include <vector>

// PJRT C API header from the XLA distribution.
// Adjust the include path to match your XLA installation.
#include "xla/pjrt/c/pjrt_c_api.h"

// ---------------------------------------------------------------------------
// Globals (internal linkage — single amalgamation TU in xla_sources.cpp)
// ---------------------------------------------------------------------------

static const PJRT_Api* g_api     = nullptr;
static PJRT_Client*    g_client  = nullptr;
static PJRT_Device*    g_device  = nullptr;
static PJRT_Memory*    g_memory  = nullptr;

// Normalized platform string from the successful xla_init ("cpu", "gpu", …).
static std::string g_xla_platform;

// one cached dlopen handle for the PJRT plugin (closed in xla_shutdown).
static void* g_plugin_handle = nullptr;

// One cached executable per activation name.
static std::unordered_map<std::string, PJRT_LoadedExecutable*> g_execs;

// PJRT plugin paths resolved by Go from cmd/asset/config.yml.
static std::unordered_map<std::string, std::string> g_configured_plugins;

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
// Compile options
// ---------------------------------------------------------------------------

// Serialized xla::CompileOptionsProto (proto3 wire format):
//   Field 3 (ExecutableBuildOptions executable_build_options) = submessage:
//     Field 4 (int32 num_replicas)   = 1  (varint 0x20 0x01)
//     Field 5 (int32 num_partitions) = 1  (varint 0x28 0x01)
// Hex: 1a 04 20 01 28 01 — outer tag 1a = field 3, length 4.
//
// Regenerate if xla CompileOptionsProto schema changes (field numbers / nesting):
// link against libprotobuf + xla headers and SerializeAsString(
//   CompileOptionsProto with executable_build_options set), or run a small
// proto encode helper in your XLA checkout and paste the bytes here.
static const char k_single_device_compile_options[] = {
    0x1a, 0x04, 0x20, 0x01, 0x28, 0x01,
};

void set_single_device_compile_options(PJRT_Client_Compile_Args* args) {
    args->compile_options = k_single_device_compile_options;
    args->compile_options_size = sizeof(k_single_device_compile_options);
}

PJRT_ExecuteOptions single_device_execute_options() {
    PJRT_ExecuteOptions options{};
    options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    options.use_major_to_minor_data_layout_for_callbacks = true;

    return options;
}

static bool await_and_destroy_event(PJRT_Event* event, const char* context) {
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

// ---------------------------------------------------------------------------
// Platform plugin loader
// ---------------------------------------------------------------------------

// dlopen-based loader for the PJRT plugin.
#include <dlfcn.h>

typedef const PJRT_Api* (*GetPjrtApiFn)();

static bool pjrt_file_exists(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");

    if (!file) return false;

    std::fclose(file);
    return true;
}

static std::vector<std::string> pjrt_plugin_names(const char* platform) {
    if (strcmp(platform, "gpu") == 0 || strcmp(platform, "cuda") == 0) {
        return {
            "pjrt_c_api_gpu_plugin.so",
            "pjrt_c_api_gpu_plugin.dylib",
        };
    }

    return {
        "pjrt_c_api_cpu_plugin.so",
        "pjrt_c_api_cpu_plugin.dylib",
    };
}

std::string pjrt_plugin_path(const char* platform) {
    std::string key(platform);
    auto configured = g_configured_plugins.find(key);

    if (configured != g_configured_plugins.end()) {
        return configured->second;
    }

    std::vector<std::string> plugin_names = pjrt_plugin_names(platform);

    return plugin_names[0];
}

static const PJRT_Api* load_pjrt_plugin(const char* platform) {
    std::string path = pjrt_plugin_path(platform);

    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "XLA: failed to dlopen %s: %s\n", path.c_str(), dlerror());
        return nullptr;
    }

    auto get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) {
        fprintf(stderr, "XLA: GetPjrtApi not found in %s\n", path.c_str());
        dlclose(handle);
        return nullptr;
    }

    const PJRT_Api* api = get_api();
    if (!api) {
        fprintf(stderr, "XLA: GetPjrtApi returned null\n");
        dlclose(handle);
        return nullptr;
    }

    if (api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
        fprintf(stderr,
                "XLA: PJRT API major mismatch, plugin=%d header=%d\n",
                api->pjrt_api_version.major_version, PJRT_API_MAJOR);
        dlclose(handle);
        return nullptr;
    }

    if (!api->PJRT_Plugin_Initialize) {
        fprintf(stderr, "XLA: PJRT plugin does not expose PJRT_Plugin_Initialize\n");
        dlclose(handle);
        return nullptr;
    }

    PJRT_Plugin_Initialize_Args init_args{};
    init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    PJRT_Error* err = api->PJRT_Plugin_Initialize(&init_args);
    if (!check(api, err, "PJRT_Plugin_Initialize")) {
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
    std::string mlir_owned(mlir_text);
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = mlir_owned.empty() ? nullptr : mlir_owned.data();
    prog.code_size   = mlir_owned.size();
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client      = g_client;
    ca.program     = &prog;
    set_single_device_compile_options(&ca);

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
        "    %out    = stablehlo.select %mask, %arg0, %scaled : (tensor<"+std::to_string(n)+"xi1>, "+t+", "+t+") -> "+t+"\n"
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

static std::string build_swish(int n) {
    std::string t = f64_type(n);
    return
        "module @swish {\n"
        "  func.func @main(%arg0: " + t + ") -> " + t + " {\n"
        "    %sig = stablehlo.logistic %arg0 : " + t + "\n"
        "    %out = stablehlo.multiply %arg0, %sig : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string build_selu(int n) {
    std::string t = f64_type(n);
    return
        "module @selu {\n"
        "  func.func @main(%arg0: " + t + ") -> " + t + " {\n"
        "    %zero  = stablehlo.constant dense<0.0> : " + t + "\n"
        "    %one   = stablehlo.constant dense<1.0> : " + t + "\n"
        "    %scale = stablehlo.constant dense<1.0507009873554805> : " + t + "\n"
        "    %alpha_scale = stablehlo.constant dense<1.7580993408473766> : " + t + "\n"
        "    %pos = stablehlo.multiply %arg0, %scale : " + t + "\n"
        "    %exp = stablehlo.exponential %arg0 : " + t + "\n"
        "    %shifted = stablehlo.subtract %exp, %one : " + t + "\n"
        "    %neg = stablehlo.multiply %shifted, %alpha_scale : " + t + "\n"
        "    %mask = stablehlo.compare GT, %arg0, %zero, TOTALORDER : (" + t + "," + t + ") -> tensor<" + std::to_string(n) + "xi1>\n"
        "    %out = stablehlo.select %mask, %pos, %neg : (tensor<" + std::to_string(n) + "xi1>, " + t + ", " + t + ") -> " + t + "\n"
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
// Execute a compiled executable: host→device, run, device→host.
//
// Input staging:
// - borrow_host_input_until_transfer_complete == false: copy into a local vector and use
//   PJRT_HostBufferSemantics_kImmutableOnlyDuringCall (safe if src is only immutable for
//   the PJRT_Client_BufferFromHostBuffer call).
// - borrow_host_input_until_transfer_complete == true: pass src through with
//   PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes (caller guarantees src stays
//   valid and unmodified until transfer completes). Synchronous xla_* wrappers satisfy this
//   and avoid doubling activation buffer memory.
// ---------------------------------------------------------------------------

static int run_executable(
    PJRT_LoadedExecutable* exec,
    const double* src, int src_n,
    double* dst,       int dst_n,
    bool borrow_host_input_until_transfer_complete)
{
    std::vector<double> host_copy;
    const double* host_ptr = nullptr;
    PJRT_HostBufferSemantics host_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;

    if (src_n > 0) {
        if (!src) return -1;

        if (borrow_host_input_until_transfer_complete) {
            host_ptr = src;
            host_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
        } else {
            host_copy.assign(src, src + src_n);
            host_ptr = host_copy.data();
            host_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        }
    }

    // --- Create input buffer ---
    PJRT_Client_BufferFromHostBuffer_Args ba{};
    ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client      = g_client;
    ba.data        = host_ptr ? const_cast<double*>(host_ptr) : nullptr;
    ba.type        = PJRT_Buffer_Type_F64;
    int64_t dims[1] = { (int64_t)src_n };
    ba.dims        = dims;
    ba.num_dims    = 1;
    ba.host_buffer_semantics = host_semantics;
    ba.device      = g_device;
    ba.memory      = g_memory;

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

    if (!await_and_destroy_event(ba.done_with_host_buffer, "Event_Await(host buffer)")) {
        destroy_buf(in_buf);
        return -1;
    }

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
    PJRT_ExecuteOptions options = single_device_execute_options();
    ea.options = &options;

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

static std::string xla_normalize_platform(const char* platform) {
    if (!platform || platform[0] == '\0') {
        return "cpu";
    }

    std::string normalized(platform);

    for (char& character : normalized) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }

    if (normalized == "cuda") {
        return "gpu";
    }

    return normalized;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_configure_plugin(const char* platform, const char* plugin_path) {
    if (!platform || platform[0] == '\0' || !plugin_path || plugin_path[0] == '\0') {
        return -1;
    }

    std::string path(plugin_path);

    if (!pjrt_file_exists(path)) {
        return -1;
    }

    g_configured_plugins[std::string(platform)] = path;
    g_configured_plugins[xla_normalize_platform(platform)] = path;

    return 0;
}

int xla_init(const char* platform) {
    std::string requested = xla_normalize_platform(platform);

    if (g_client) {
        if (g_xla_platform != requested) {
            fprintf(stderr,
                    "XLA: xla_init refused platform %s (already initialized as %s)\n",
                    requested.c_str(), g_xla_platform.c_str());
            return -1;
        }

        return 0;
    }

    g_api = load_pjrt_plugin(requested.c_str());
    if (!g_api) return -1;

    PJRT_Client_Create_Args ca{};
    ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    // No extra create options needed for default CPU/GPU.
    PJRT_Error* err = g_api->PJRT_Client_Create(&ca);
    if (!check(g_api, err, "PJRT_Client_Create")) return -1;

    g_client = ca.client;

    PJRT_Client_AddressableDevices_Args da{};
    da.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    da.client = g_client;
    err = g_api->PJRT_Client_AddressableDevices(&da);
    if (!check(g_api, err, "PJRT_Client_AddressableDevices")) return -1;
    if (da.num_addressable_devices == 0 || !da.addressable_devices) {
        fprintf(stderr, "XLA: PJRT client returned no addressable devices\n");
        return -1;
    }

    g_device = da.addressable_devices[0];

    PJRT_Client_AddressableMemories_Args ma{};
    ma.struct_size = PJRT_Client_AddressableMemories_Args_STRUCT_SIZE;
    ma.client = g_client;
    err = g_api->PJRT_Client_AddressableMemories(&ma);
    if (!check(g_api, err, "PJRT_Client_AddressableMemories")) return -1;
    if (ma.num_addressable_memories > 0 && ma.addressable_memories) {
        g_memory = ma.addressable_memories[0];
    }

    g_xla_platform = requested;

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
        { "swish",      build_swish(n)        },
        { "selu",       build_selu(n)         },
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
    return run_executable(g_execs["relu"], src, n, dst, n, true);
}

int xla_leaky_relu(const double* src, double* dst, double alpha, int n) {
    // Recompile with correct alpha.
    std::string mlir = build_leaky_relu(n, alpha);
    PJRT_LoadedExecutable* exec = compile_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_executable(exec, src, n, dst, n, true);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

int xla_gelu(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["gelu"], src, n, dst, n, true);
}

int xla_tanh_act(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["tanh_act"], src, n, dst, n, true);
}

int xla_sigmoid(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["sigmoid"], src, n, dst, n, true);
}

int xla_swish(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["swish"], src, n, dst, n, true);
}

int xla_selu(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["selu"], src, n, dst, n, true);
}

int xla_swiglu(const double* src, double* dst, int n) {
    if (g_compiled_n != n && xla_compile_activations(n) != 0) return -1;
    return run_executable(g_execs["swiglu"], src, 2*n, dst, n, true);
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
    g_device     = nullptr;
    g_memory     = nullptr;
    g_xla_platform.clear();

    if (g_plugin_handle) {
        dlclose(g_plugin_handle);
        g_plugin_handle = nullptr;
    }
}

} // extern "C"
