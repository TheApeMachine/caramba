// XLA VSA backend — PJRT C API implementation.
//
// Implements bind (elementwise multiply), bundle (sum + L2-normalise),
// and similarity (dot product) for FHRR-style VSA hypervectors.
//
// Each operation is expressed as a StableHLO text module, compiled once via
// PJRT_Client_Compile and cached for reuse.

#include "xla_vsa.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <dlfcn.h>
#endif

#include "xla/pjrt/c/pjrt_c_api.h"

namespace {

constexpr double kVsaNormEps = 1e-12;

} // namespace

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static std::mutex gv_mutex;

static const PJRT_Api*    gv_api    = nullptr;
static PJRT_Client*       gv_client = nullptr;

#ifdef __linux__
static void* gv_dl_handle = nullptr;
#endif

static std::unordered_map<std::string, PJRT_LoadedExecutable*> gv_execs;
static std::string gv_plugin_path;

static std::mutex gv_err_mu;
static std::string gv_last_err;

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

static bool vcheck(const PJRT_Api* api, PJRT_Error* err, const char* ctx) {
    if (!err) return true;
    PJRT_Error_Message_Args ma{};
    ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    ma.error = err;
    api->PJRT_Error_Message(&ma);
    std::string detail;
    if (ma.message && ma.message_size > 0) {
        detail.assign(ma.message, ma.message + ma.message_size);
    }
    {
        std::lock_guard<std::mutex> elock(gv_err_mu);
        gv_last_err = std::string(ctx) + ": " + detail;
    }
    fprintf(stderr, "XLA VSA PJRT error in %s: %s\n", ctx, detail.c_str());
    vfree_error(api, err);
    return false;
}

// ---------------------------------------------------------------------------
// PJRT plugin path helper (matches _math_xla.cpp convention)
// ---------------------------------------------------------------------------

static std::string pjrt_plugin_path(const char* platform) {
    if (!gv_plugin_path.empty()) return gv_plugin_path;

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

static void destroy_buf(PJRT_Buffer* buf);

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
    if (!vcheck(gv_api, err, "PJRT_Client_BufferFromHostBuffer")) return nullptr;

    PJRT_Buffer_ReadyEvent_Args re{};
    re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    re.buffer = ba.buffer;
    auto* err_re = gv_api->PJRT_Buffer_ReadyEvent(&re);
    if (!vcheck(gv_api, err_re, "PJRT_Buffer_ReadyEvent")) {
        destroy_buf(ba.buffer);
        return nullptr;
    }

    PJRT_Event_Await_Args ea{};
    ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea.event = re.event;
    auto* err_ea = gv_api->PJRT_Event_Await(&ea);
    if (!vcheck(gv_api, err_ea, "PJRT_Event_Await(host_to_device)")) {
        PJRT_Event_Destroy_Args ed_fail{};
        ed_fail.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        ed_fail.event = re.event;
        gv_api->PJRT_Event_Destroy(&ed_fail);
        destroy_buf(ba.buffer);
        return nullptr;
    }

    PJRT_Event_Destroy_Args ed{};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = re.event;
    auto* err_ed = gv_api->PJRT_Event_Destroy(&ed);
    if (!vcheck(gv_api, err_ed, "PJRT_Event_Destroy(host_to_device)")) {
        destroy_buf(ba.buffer);
        return nullptr;
    }

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
    if (!vcheck(gv_api, err, "PJRT_Buffer_ToHostBuffer")) return -1;

    PJRT_Event_Await_Args ea{};
    ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea.event = ca.event;
    auto* err_ea = gv_api->PJRT_Event_Await(&ea);
    if (!vcheck(gv_api, err_ea, "PJRT_Event_Await(device_to_host)")) {
        PJRT_Event_Destroy_Args ed_fail{};
        ed_fail.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        ed_fail.event = ca.event;
        gv_api->PJRT_Event_Destroy(&ed_fail);
        return -1;
    }

    PJRT_Event_Destroy_Args ed{};
    ed.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    ed.event = ca.event;
    auto* err_ed = gv_api->PJRT_Event_Destroy(&ed);
    if (!vcheck(gv_api, err_ed, "PJRT_Event_Destroy(device_to_host)")) return -1;

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
    if (mlir.empty() || !gv_api || !gv_client) return nullptr;

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
    if (!vcheck(gv_api, err, "PJRT_Client_Compile")) return nullptr;

    gv_execs[key] = ca.executable;
    return ca.executable;
}

// ---------------------------------------------------------------------------
// StableHLO module builders
// ---------------------------------------------------------------------------

static bool elementwise_op_allowed(const std::string& op) {
    return op == "stablehlo.multiply" || op == "stablehlo.add";
}

static std::string elementwise_module(const std::string& op, int n) {
    if (!elementwise_op_allowed(op)) {
        fprintf(stderr, "XLA VSA: disallowed elementwise op (expected stablehlo.multiply|stablehlo.add): %s\n",
                op.c_str());
        return {};
    }
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    return
        "module @m {"
        "  func.func @main(%a: " + t + ", %b: " + t + ") -> " + t + " {"
        "    %r = " + op + " %a, %b : " + t +
        "    return %r : " + t +
        "  }"
        "}";
}

static std::string scalar_literal(double value) {
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);

    return std::string(buffer);
}

static std::string similarity_module(int n) {
    std::string vector_type = "tensor<" + std::to_string(n) + "xf64>";
    std::string scalar_type = "tensor<f64>";

    return
        "module @m {"
        "  func.func @main(%a: " + vector_type + ", %b: " + vector_type + ") -> " + scalar_type + " {"
        "    %product = stablehlo.multiply %a, %b : " + vector_type +
        "    %zero = stablehlo.constant dense<0.0> : " + scalar_type +
        "    %sum = stablehlo.reduce(%product init: %zero)"
        "      applies stablehlo.add across dimensions = [0] : (" + vector_type + ", " + scalar_type + ") -> " + scalar_type +
        "    return %sum : " + scalar_type +
        "  }"
        "}";
}

static std::string bundle_module(int n, int num_vecs) {
    std::string vector_type = "tensor<" + std::to_string(n) + "xf64>";
    std::string scalar_type = "tensor<f64>";
    std::ostringstream args;
    std::ostringstream body;

    for (int input_index = 0; input_index < num_vecs; input_index++) {
        if (input_index > 0) args << ", ";
        args << "%v" << input_index << ": " << vector_type;
    }

    std::string sum_name = "%v0";

    for (int input_index = 1; input_index < num_vecs; input_index++) {
        std::string next_sum = "%sum" + std::to_string(input_index);
        body
            << "    " << next_sum << " = stablehlo.add " << sum_name
            << ", %v" << input_index << " : " << vector_type << "\n";
        sum_name = next_sum;
    }

    body
        << "    %squares = stablehlo.multiply " << sum_name << ", " << sum_name
        << " : " << vector_type << "\n"
        << "    %zero = stablehlo.constant dense<0.0> : " << scalar_type << "\n"
        << "    %sumsq = stablehlo.reduce(%squares init: %zero)"
        << " applies stablehlo.add across dimensions = [0] : ("
        << vector_type << ", " << scalar_type << ") -> " << scalar_type << "\n"
        << "    %norm = stablehlo.sqrt %sumsq : " << scalar_type << "\n"
        << "    %eps = stablehlo.constant dense<" << scalar_literal(kVsaNormEps)
        << "> : " << scalar_type << "\n"
        << "    %one = stablehlo.constant dense<1.0> : " << scalar_type << "\n"
        << "    %norm_ok = stablehlo.compare GT, %norm, %eps, TOTALORDER : ("
        << scalar_type << ", " << scalar_type << ") -> tensor<i1>\n"
        << "    %inv_norm = stablehlo.divide %one, %norm : " << scalar_type << "\n"
        << "    %scale = stablehlo.select %norm_ok, %inv_norm, %one : (tensor<i1>, "
        << scalar_type << ", " << scalar_type << ") -> " << scalar_type << "\n"
        << "    %scale_v = stablehlo.broadcast_in_dim %scale, dims = [] : ("
        << scalar_type << ") -> " << vector_type << "\n"
        << "    %out_scaled = stablehlo.multiply " << sum_name << ", %scale_v : "
        << vector_type << "\n"
        << "    return %out_scaled : " << vector_type << "\n";

    return
        "module @m {"
        "  func.func @main(" + args.str() + ") -> " + vector_type + " {\n" +
        body.str() +
        "  }"
        "}";
}

// cyclic_shift_module expects k pre-normalized into [0,n-1].
static std::string cyclic_shift_module(int n, int k) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";

    if (k == 0) {
        return
            "module @m {"
            "  func.func @main(%x: " + t + ") -> " + t + " {"
            "    return %x : " + t +
            "  }"
            "}";
    }

    std::string first = "tensor<" + std::to_string(k) + "xf64>";
    std::string second = "tensor<" + std::to_string(n - k) + "xf64>";

    return
        "module @m {"
        "  func.func @main(%x: " + t + ") -> " + t + " {"
        "    %tail = stablehlo.slice %x [" + std::to_string(n - k) + ":" + std::to_string(n) + "] : (" + t + ") -> " + first +
        "    %head = stablehlo.slice %x [0:" + std::to_string(n - k) + "] : (" + t + ") -> " + second +
        "    %out = stablehlo.concatenate %tail, %head, dim = 0 : (" + first + ", " + second + ") -> " + t +
        "    return %out : " + t +
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
    if (!vcheck(gv_api, xerr, "PJRT_LoadedExecutable_Execute(binary)")) return -1;

    int rc = device_to_host(out_buf, out_ptr, out_bytes);
    destroy_buf(out_buf);
    return rc;
}

// ---------------------------------------------------------------------------
// run_variadic: execute an N-input → 1-output compiled executable
// ---------------------------------------------------------------------------

static int run_variadic(
    PJRT_LoadedExecutable* exec,
    const double** input_ptrs, int num_inputs, size_t input_n,
    double* out_ptr, size_t out_bytes)
{
    if (!exec || !input_ptrs || num_inputs <= 0) return -1;

    std::vector<PJRT_Buffer*> args((size_t)num_inputs, nullptr);

    for (int input_index = 0; input_index < num_inputs; input_index++) {
        args[(size_t)input_index] = host_to_device(input_ptrs[input_index], input_n);

        if (args[(size_t)input_index]) continue;

        for (PJRT_Buffer* buffer : args) destroy_buf(buffer);
        return -1;
    }

    PJRT_Buffer** arg_lists[1] = { args.data() };
    PJRT_Buffer* out_buf = nullptr;
    PJRT_Buffer** out_lists[1] = { &out_buf };

    PJRT_LoadedExecutable_Execute_Args xa{};
    xa.struct_size    = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    xa.executable     = exec;
    PJRT_ExecuteOptions opts = single_device_execute_options();
    xa.options        = &opts;
    xa.argument_lists = arg_lists;
    xa.num_devices    = 1;
    xa.num_args       = (size_t)num_inputs;
    xa.output_lists   = out_lists;
    xa.device_complete_events = nullptr;

    auto* xerr = gv_api->PJRT_LoadedExecutable_Execute(&xa);

    for (PJRT_Buffer* buffer : args) destroy_buf(buffer);

    if (!vcheck(gv_api, xerr, "PJRT_LoadedExecutable_Execute(variadic)")) return -1;

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
    if (!vcheck(gv_api, xerr, "PJRT_LoadedExecutable_Execute(unary)")) return -1;

    int rc = device_to_host(out_buf, out_ptr, out_bytes);
    destroy_buf(out_buf);
    return rc;
}

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_vsa_configure_plugin(const char* platform, const char* plugin_path) {
    (void)platform;

    if (!plugin_path || plugin_path[0] == '\0') return -1;

    std::lock_guard<std::mutex> lock(gv_mutex);
    gv_plugin_path = plugin_path;

    return 0;
}

int xla_vsa_init(const char* platform) {
    std::lock_guard<std::mutex> lock(gv_mutex);

    if (!platform) return -1;
    std::string plugin = pjrt_plugin_path(platform);

#ifdef __linux__
    void* handle = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) return -1;
    typedef const PJRT_Api* (*GetPJRTApiFn)();
    auto* get_api = (GetPJRTApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) {
        dlclose(handle);
        return -1;
    }
    const PJRT_Api* api = get_api();
    if (!api) {
        dlclose(handle);
        return -1;
    }

    PJRT_Client_Create_Args cca{};
    cca.struct_size    = PJRT_Client_Create_Args_STRUCT_SIZE;
    cca.create_options = nullptr;
    cca.num_options    = 0;

    auto* err = api->PJRT_Client_Create(&cca);
    if (!vcheck(api, err, "PJRT_Client_Create")) {
        dlclose(handle);
        return -1;
    }

    gv_api     = api;
    gv_client  = cca.client;
    gv_dl_handle = handle;
    return 0;
#else
    (void)plugin;
    return -1;
#endif
}

void xla_vsa_shutdown(void) {
    std::lock_guard<std::mutex> lock(gv_mutex);

    if (!gv_api) {
#ifdef __linux__
        if (gv_dl_handle) {
            dlclose(gv_dl_handle);
            gv_dl_handle = nullptr;
        }
#endif
        return;
    }

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

    gv_api = nullptr;

#ifdef __linux__
    if (gv_dl_handle) {
        dlclose(gv_dl_handle);
        gv_dl_handle = nullptr;
    }
#endif
}

int xla_vsa_bind(const double* a, const double* b, double* out, int n) {
    std::lock_guard<std::mutex> lock(gv_mutex);

    if (!a || !b || !out || n <= 0 || !gv_client) return -1;

    std::string key = "vsa_bind_" + std::to_string(n);
    std::string mlir = elementwise_module("stablehlo.multiply", n);
    auto* exec = compile_module(key, mlir);
    if (!exec) return -1;

    return run_binary(exec, a, (size_t)n, b, (size_t)n, out, (size_t)n * sizeof(double));
}

int xla_vsa_bundle(const double** vecs, int num_vecs, double* out, int n) {
    std::lock_guard<std::mutex> lock(gv_mutex);

    if (!vecs || !out || n <= 0 || num_vecs <= 0 || !gv_client) return -1;

    for (int i = 0; i < num_vecs; i++) {
        if (!vecs[i]) return -1;
    }

    std::string key = "vsa_bundle_" + std::to_string(n) + "_" + std::to_string(num_vecs);
    auto* red_exec = compile_module(key, bundle_module(n, num_vecs));
    if (!red_exec) return -1;

    return run_variadic(red_exec, vecs, num_vecs, (size_t)n, out, (size_t)n * sizeof(double));
}

int xla_vsa_similarity(const double* a, const double* b, double* out, int n) {
    std::lock_guard<std::mutex> lock(gv_mutex);

    if (!a || !b || !out || n <= 0 || !gv_client) return -1;

    std::string red_key = "vsa_similarity_" + std::to_string(n);
    auto* red_exec = compile_module(red_key, similarity_module(n));
    if (!red_exec) return -1;

    return run_binary(red_exec, a, (size_t)n, b, (size_t)n, out, sizeof(double));
}

int xla_vsa_permute(const double* src, double* out, int n, int shift) {
    std::lock_guard<std::mutex> lock(gv_mutex);

    if (!src || !out || n <= 0 || !gv_client) return -1;

    int k = ((shift % n) + n) % n;
    std::string key = "vsa_permute_" + std::to_string(n) + "_" + std::to_string(k);
    auto* exec = compile_module(key, cyclic_shift_module(n, k));
    if (!exec) return -1;

    return run_unary(exec, src, (size_t)n, out, (size_t)n * sizeof(double));
}

int xla_vsa_inverse_permute(const double* src, double* out, int n, int shift) {
    std::lock_guard<std::mutex> lock(gv_mutex);

    if (!src || !out || n <= 0 || !gv_client) return -1;

    int k = ((shift % n) + n) % n;
    if (k != 0) k = n - k;
    std::string key = "vsa_inverse_permute_" + std::to_string(n) + "_" + std::to_string(k);
    auto* exec = compile_module(key, cyclic_shift_module(n, k));
    if (!exec) return -1;

    return run_unary(exec, src, (size_t)n, out, (size_t)n * sizeof(double));
}

const char* xla_vsa_get_last_error(void) {
    static char buf[4096];
    std::lock_guard<std::mutex> lock(gv_err_mu);
    std::snprintf(buf, sizeof buf, "%s", gv_last_err.c_str());
    return buf;
}

} // extern "C"
