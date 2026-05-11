// XLA projection backend — PJRT C API implementation.
//
// Implements Linear, FusedQKV, and TiedEmbedding via stablehlo.dot_general.
// Shares the g_api / g_client globals and infrastructure from activation_xla.cc.
//
// Build requirements (same as activation_xla.cc):
//   CGO_CPPFLAGS="-I/path/to/xla" CGO_LDFLAGS="-ldl -lstdc++"
//   go build -tags "cgo xla" ./backend/compute/xla/

#include "projection.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include "xla/pjrt/c/pjrt_c_api.h"

// Shared PJRT state from amalgamated _activation_xla.cpp.

static std::unordered_map<std::string, PJRT_LoadedExecutable*> g_proj_execs;

// ---------------------------------------------------------------------------
// Error helpers (duplicated from activation_xla.cc for self-containment)
// ---------------------------------------------------------------------------

static void proj_free_error(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
}

static bool proj_check(const PJRT_Api* api, PJRT_Error* err, const char* ctx) {
    if (!err) return true;
    PJRT_Error_Message_Args ma{};
    ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    ma.error = err;
    api->PJRT_Error_Message(&ma);
    fprintf(stderr, "XLA projection error in %s: %.*s\n", ctx, (int)ma.message_size, ma.message);
    proj_free_error(api, err);
    return false;
}

// ---------------------------------------------------------------------------
// Compile helper
// ---------------------------------------------------------------------------

static PJRT_LoadedExecutable* proj_compile(const std::string& mlir) {
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = const_cast<char*>(mlir.c_str());
    prog.code_size   = mlir.size();
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client      = g_client;
    ca.program      = &prog;
    set_single_device_compile_options(&ca);

    PJRT_Error* err = g_api->PJRT_Client_Compile(&ca);
    if (!proj_check(g_api, err, "PJRT_Client_Compile(proj)")) return nullptr;
    return ca.executable;
}

// ---------------------------------------------------------------------------
// StableHLO builders
// ---------------------------------------------------------------------------

static std::string s(int v) { return std::to_string(v); }

// f64 tensor type helpers
static std::string proj_t1(int n)      { return "tensor<" + s(n) + "xf64>"; }
static std::string t2(int r, int c)    { return "tensor<" + s(r) + "x" + s(c) + "xf64>"; }

// Linear / FusedQKV: result = A[M,K] dot W[N,K]^T  => [M,N]
// stablehlo.dot_general with lhs contracting dim 1, rhs contracting dim 1.
static std::string build_linear(int M, int K, int N, bool has_bias) {
    // A: [M,K], W: [N,K], bias: [N] (optional)
    std::string tA    = t2(M, K);
    std::string tW    = t2(N, K);
    std::string tBias = proj_t1(N);
    std::string tC    = t2(M, N);

    std::string args = "%a: " + tA + ", %w: " + tW;
    if (has_bias) args += ", %bias: " + tBias;

    std::string body =
        "    %c = stablehlo.dot_general %a, %w, "
        "contracting_dims = [1] x [1] : (" + tA + ", " + tW + ") -> " + tC + "\n";
    if (has_bias) {
        // Broadcast bias [N] -> [M, N] and add.
        body +=
            "    %bias2d = stablehlo.broadcast_in_dim %bias, dims = [1] : (" +
            tBias + ") -> " + tC + "\n"
            "    %out = stablehlo.add %c, %bias2d : " + tC + "\n"
            "    return %out : " + tC + "\n";
    } else {
        body += "    return %c : " + tC + "\n";
    }

    return
        "module @linear {\n"
        "  func.func @main(" + args + ") -> " + tC + " {\n" +
        body +
        "  }\n"
        "}\n";
}

static std::string build_tied_embedding(int M, int D, int V) {
    std::string tH = t2(M, D);
    std::string tW = t2(V, D);
    std::string tL = t2(M, V);
    return
        "module @tied_emb {\n"
        "  func.func @main(%h: " + tH + ", %w: " + tW + ") -> " + tL + " {\n"
        "    %out = stablehlo.dot_general %h, %w, "
        "contracting_dims = [1] x [1] : (" + tH + ", " + tW + ") -> " + tL + "\n"
        "    return %out : " + tL + "\n"
        "  }\n"
        "}\n";
}

// ---------------------------------------------------------------------------
// Execute helper: 2-input (src, weight) -> dst
// ---------------------------------------------------------------------------

static int proj_run2(
    PJRT_LoadedExecutable* exec,
    const double* h_a, int a_n,
    const double* h_b, int b_n,
    double* h_dst,     int dst_n)
{
    auto make_buf = [&](const double* data, int n) -> PJRT_Buffer* {
        PJRT_Client_BufferFromHostBuffer_Args ba{};
        ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        ba.client = g_client;
        ba.data   = data;
        ba.type   = PJRT_Buffer_Type_F64;
        int64_t dims[1] = { (int64_t)n };
        ba.dims     = dims;
        ba.num_dims = 1;
        ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        PJRT_Error* err = g_api->PJRT_Client_BufferFromHostBuffer(&ba);
        if (!proj_check(g_api, err, "BufferFromHostBuffer")) return nullptr;
        // Wait for ready event
        PJRT_Buffer_ReadyEvent_Args re{};
        re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        re.buffer = ba.buffer;
        err = g_api->PJRT_Buffer_ReadyEvent(&re);
        if (!proj_check(g_api, err, "ReadyEvent")) return nullptr;
        PJRT_Event_Await_Args ea{};
        ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ea.event = re.event;
        err = g_api->PJRT_Event_Await(&ea);
        proj_check(g_api, err, "Event_Await");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = re.event;
        g_api->PJRT_Event_Destroy(&eda);
        return ba.buffer;
    };

    PJRT_Buffer* in_a = make_buf(h_a, a_n);
    PJRT_Buffer* in_b = make_buf(h_b, b_n);
    if (!in_a || !in_b) return -1;

    PJRT_Buffer* in_list[2]  = { in_a, in_b };
    PJRT_Buffer* out_storage = nullptr;
    PJRT_Buffer** out_list[1] = { &out_storage };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size     = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable      = exec;
    ea.argument_lists  = (PJRT_Buffer***)&in_list;
    ea.num_devices     = 1;
    ea.num_args        = 2;
    ea.output_lists    = out_list;
    PJRT_ExecuteOptions options = single_device_execute_options();
    ea.options = &options;

    PJRT_Error* err = g_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!proj_check(g_api, err, "Execute")) return -1;

    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_storage;
    tha.dst         = h_dst;
    tha.dst_size    = (size_t)dst_n * sizeof(double);
    err = g_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (!proj_check(g_api, err, "ToHostBuffer")) return -1;

    PJRT_Event_Await_Args ev{};
    ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ev.event = tha.event;
    err = g_api->PJRT_Event_Await(&ev);
    proj_check(g_api, err, "Event_Await(out)");
    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event = tha.event;
    g_api->PJRT_Event_Destroy(&eda);

    auto destroy_buf = [&](PJRT_Buffer* b) {
        PJRT_Buffer_Destroy_Args da{};
        da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        da.buffer = b;
        g_api->PJRT_Buffer_Destroy(&da);
    };
    destroy_buf(in_a);
    destroy_buf(in_b);
    destroy_buf(out_storage);
    return 0;
}

// ---------------------------------------------------------------------------
// 3-input execute (A, W, bias)
// ---------------------------------------------------------------------------

static int proj_run3(
    PJRT_LoadedExecutable* exec,
    const double* h_a, int a_n,
    const double* h_b, int b_n,
    const double* h_c, int c_n,
    double* h_dst,     int dst_n)
{
    auto make_buf = [&](const double* data, int n) -> PJRT_Buffer* {
        PJRT_Client_BufferFromHostBuffer_Args ba{};
        ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        ba.client = g_client;
        ba.data   = data;
        ba.type   = PJRT_Buffer_Type_F64;
        int64_t dims[1] = { (int64_t)n };
        ba.dims = dims; ba.num_dims = 1;
        ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        PJRT_Error* err = g_api->PJRT_Client_BufferFromHostBuffer(&ba);
        if (!proj_check(g_api, err, "BufferFromHostBuffer3")) return nullptr;
        PJRT_Buffer_ReadyEvent_Args re{};
        re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        re.buffer = ba.buffer;
        err = g_api->PJRT_Buffer_ReadyEvent(&re);
        if (!proj_check(g_api, err, "ReadyEvent3")) return nullptr;
        PJRT_Event_Await_Args ea{};
        ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ea.event = re.event;
        err = g_api->PJRT_Event_Await(&ea);
        proj_check(g_api, err, "Event_Await3");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = re.event;
        g_api->PJRT_Event_Destroy(&eda);
        return ba.buffer;
    };

    PJRT_Buffer* in_a = make_buf(h_a, a_n);
    PJRT_Buffer* in_b = make_buf(h_b, b_n);
    PJRT_Buffer* in_c = make_buf(h_c, c_n);
    if (!in_a || !in_b || !in_c) return -1;

    PJRT_Buffer* in_list[3]  = { in_a, in_b, in_c };
    PJRT_Buffer* out_storage = nullptr;
    PJRT_Buffer** out_list[1] = { &out_storage };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size     = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable      = exec;
    ea.argument_lists  = (PJRT_Buffer***)&in_list;
    ea.num_devices     = 1;
    ea.num_args        = 3;
    ea.output_lists    = out_list;
    PJRT_ExecuteOptions options = single_device_execute_options();
    ea.options = &options;

    PJRT_Error* err = g_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!proj_check(g_api, err, "Execute3")) return -1;

    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src = out_storage; tha.dst = h_dst;
    tha.dst_size = (size_t)dst_n * sizeof(double);
    err = g_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (!proj_check(g_api, err, "ToHostBuffer3")) return -1;

    PJRT_Event_Await_Args ev{};
    ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ev.event = tha.event;
    err = g_api->PJRT_Event_Await(&ev);
    proj_check(g_api, err, "Event_Await3(out)");
    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event = tha.event;
    g_api->PJRT_Event_Destroy(&eda);

    auto destroy_buf = [&](PJRT_Buffer* b) {
        PJRT_Buffer_Destroy_Args da{};
        da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        da.buffer = b;
        g_api->PJRT_Buffer_Destroy(&da);
    };
    destroy_buf(in_a); destroy_buf(in_b); destroy_buf(in_c); destroy_buf(out_storage);
    return 0;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_projection_init(const char* /*platform*/) {
    // g_api and g_client are already initialised by xla_init() in
    // activation_xla.cc.  If called standalone, delegate to xla_init.
    if (!g_client) {
        fprintf(stderr, "xla_projection_init: call xla_init first\n");
        return -1;
    }
    return 0;
}

int xla_compile_projections(int /*M*/, int /*K*/, int /*N*/) {
    // Executables are compiled on-demand per shape in the dispatch functions.
    return 0;
}

int xla_linear(const double* src, const double* weight, const double* bias,
               double* dst, int M, int K, int N, int has_bias)
{
    std::string key = "linear_" + s(M) + "_" + s(K) + "_" + s(N) + "_" + s(has_bias);
    if (g_proj_execs.find(key) == g_proj_execs.end()) {
        PJRT_LoadedExecutable* ex = proj_compile(build_linear(M, K, N, (bool)has_bias));
        if (!ex) return -1;
        g_proj_execs[key] = ex;
    }
    PJRT_LoadedExecutable* exec = g_proj_execs[key];
    if (has_bias) {
        return proj_run3(exec, src, M*K, weight, N*K, bias, N, dst, M*N);
    }
    return proj_run2(exec, src, M*K, weight, N*K, dst, M*N);
}

int xla_fused_qkv(const double* src, const double* weight, const double* bias,
                  double* dst, int M, int K, int N, int has_bias)
{
    return xla_linear(src, weight, bias, dst, M, K, N, has_bias);
}

int xla_tied_embedding(const double* src, const double* weight,
                       double* dst, int M, int D, int V)
{
    std::string key = "tied_emb_" + s(M) + "_" + s(D) + "_" + s(V);
    if (g_proj_execs.find(key) == g_proj_execs.end()) {
        PJRT_LoadedExecutable* ex = proj_compile(build_tied_embedding(M, D, V));
        if (!ex) return -1;
        g_proj_execs[key] = ex;
    }
    return proj_run2(g_proj_execs[key], src, M*D, weight, V*D, dst, M*V);
}

void xla_projection_shutdown(void) {
    for (auto& kv : g_proj_execs) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = kv.second;
        g_api->PJRT_LoadedExecutable_Destroy(&da);
    }
    g_proj_execs.clear();
}

} // extern "C"
