// XLA convolution backend — PJRT C API implementation.
//
// Expresses Conv1d/2d/3d and ConvTranspose2d as StableHLO
// stablehlo.convolution with appropriate dimension_numbers.
//
// Build requirements: same as activation_xla.cc.

#include "convolution.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

// ---------------------------------------------------------------------------
// Reuse the same globals as activation_xla.cc if linked into the same
// translation unit.  If compiled separately, we redeclare them as extern.
// ---------------------------------------------------------------------------

extern const PJRT_Api*  g_api;
extern PJRT_Client*     g_client;

// Local executable cache for convolution ops.
static std::unordered_map<std::string, PJRT_LoadedExecutable*> g_conv_execs;

// ---------------------------------------------------------------------------
// Forward declarations of helpers defined in activation_xla.cc
// ---------------------------------------------------------------------------

extern bool check(const PJRT_Api* api, PJRT_Error* err, const char* ctx);
extern int  run_executable(PJRT_LoadedExecutable* exec,
                           const double* src, int src_n,
                           double* dst,       int dst_n);
// Two-input variant for conv (x + packed weights+bias concatenated).
// We define our own two-input runner below.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static PJRT_LoadedExecutable* compile_stablehlo_conv(const std::string& mlir) {
    PJRT_Client_Compile_Args ca{};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client      = g_client;
    ca.program     = &(PJRT_Program{
        .struct_size = PJRT_Program_STRUCT_SIZE,
        .code        = mlir.c_str(),
        .code_size   = mlir.size(),
        .format      = "mlir",
        .format_size = 4,
    });
    ca.compile_options      = nullptr;
    ca.compile_options_size = 0;
    PJRT_Error* err = g_api->PJRT_Client_Compile(&ca);
    if (!check(g_api, err, "compile_stablehlo_conv")) return nullptr;
    return ca.executable;
}

static PJRT_Buffer* make_buffer(const double* data, const int64_t* dims, int ndims) {
    PJRT_Client_BufferFromHostBuffer_Args ba{};
    ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client      = g_client;
    ba.data        = data;
    ba.type        = PJRT_Buffer_Type_F64;
    ba.dims        = dims;
    ba.num_dims    = (size_t)ndims;
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    PJRT_Error* err = g_api->PJRT_Client_BufferFromHostBuffer(&ba);
    if (!check(g_api, err, "BufferFromHostBuffer(conv)")) return nullptr;

    // Wait for transfer.
    {
        PJRT_Buffer_ReadyEvent_Args re{};
        re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        re.buffer = ba.buffer;
        err = g_api->PJRT_Buffer_ReadyEvent(&re);
        if (!check(g_api, err, "ReadyEvent(conv)")) return nullptr;
        PJRT_Event_Await_Args ea{};
        ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ea.event = re.event;
        g_api->PJRT_Event_Await(&ea);
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = re.event;
        g_api->PJRT_Event_Destroy(&eda);
    }
    return ba.buffer;
}

static void destroy_buffer(PJRT_Buffer* b) {
    if (!b) return;
    PJRT_Buffer_Destroy_Args da{};
    da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    da.buffer = b;
    g_api->PJRT_Buffer_Destroy(&da);
}

// Run a 3-input executable (x, weight, bias) and write result to dst.
static int run_conv_exec(
    PJRT_LoadedExecutable* exec,
    PJRT_Buffer* bx, PJRT_Buffer* bw, PJRT_Buffer* bb,
    double* dst, int dst_n)
{
    PJRT_Buffer* in_list[3]   = { bx, bw, bb };
    PJRT_Buffer* out_storage  = nullptr;
    PJRT_Buffer** out_list[1] = { &out_storage };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size    = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable     = exec;
    ea.argument_lists = (PJRT_Buffer***)&in_list;
    ea.num_devices    = 1;
    ea.num_args       = 3;
    ea.output_lists   = out_list;
    ea.execute_options = nullptr;

    PJRT_Error* err = g_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!check(g_api, err, "Execute(conv)")) return -1;

    PJRT_Buffer* out_buf = out_storage;

    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_buf;
    tha.dst         = dst;
    tha.dst_size    = (size_t)dst_n * sizeof(double);
    err = g_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (!check(g_api, err, "ToHostBuffer(conv)")) { destroy_buffer(out_buf); return -1; }

    {
        PJRT_Event_Await_Args ev{};
        ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ev.event = tha.event;
        g_api->PJRT_Event_Await(&ev);
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = tha.event;
        g_api->PJRT_Event_Destroy(&eda);
    }

    destroy_buffer(out_buf);
    return 0;
}

// ---------------------------------------------------------------------------
// StableHLO module builders
// ---------------------------------------------------------------------------

static std::string t(const std::vector<int>& dims) {
    std::string s = "tensor<";
    for (int i = 0; i < (int)dims.size(); i++) {
        if (i) s += "x";
        s += std::to_string(dims[i]);
    }
    s += "xf64>";
    return s;
}

// Build Conv1d StableHLO module.
// StableHLO conv1d: input [N,InC,L], weight [OutC, InC/g, K]
// dimension_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0]
static std::string build_conv1d(
    int N, int InC, int L,
    int OutC, int K, int stride, int pad, int dilation, int groups, int L_out)
{
    int icPg = InC / groups;
    auto tx = t({N, InC, L});
    auto tw = t({OutC, icPg, K});
    auto tb = t({OutC});
    auto tout = t({N, OutC, L_out});

    char buf[4096];
    snprintf(buf, sizeof(buf),
        "module @conv1d {\n"
        "  func.func @main(%%x: %s, %%w: %s, %%b: %s) -> %s {\n"
        "    %%c = stablehlo.convolution(%%x, %%w)\n"
        "         dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],\n"
        "         window = {stride = [%d], pad = [[%d, %d]], rhs_dilate = [%d]}\n"
        "         {batch_group_count = 1 : i64, feature_group_count = %d : i64}\n"
        "         : (%s, %s) -> %s\n"
        "    %%bcast = stablehlo.broadcast_in_dim %%b, dims = [1] : (%s) -> %s\n"
        "    %%out = stablehlo.add %%c, %%bcast : %s\n"
        "    return %%out : %s\n"
        "  }\n"
        "}\n",
        tx.c_str(), tw.c_str(), tb.c_str(), tout.c_str(),
        stride, pad, pad, dilation, groups,
        tx.c_str(), tw.c_str(), tout.c_str(),
        tb.c_str(), tout.c_str(),
        tout.c_str(),
        tout.c_str());
    return std::string(buf);
}

// Build Conv2d StableHLO module.
static std::string build_conv2d(
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout)
{
    int icPg = InC / groups;
    auto tx   = t({N, InC, H, W});
    auto tw   = t({OutC, icPg, KH, KW});
    auto tb   = t({OutC});
    auto tout = t({N, OutC, Hout, Wout});

    char buf[4096];
    snprintf(buf, sizeof(buf),
        "module @conv2d {\n"
        "  func.func @main(%%x: %s, %%w: %s, %%b: %s) -> %s {\n"
        "    %%c = stablehlo.convolution(%%x, %%w)\n"
        "         dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n"
        "         window = {stride = [%d, %d], pad = [[%d, %d], [%d, %d]],"
                           " rhs_dilate = [%d, %d]}\n"
        "         {batch_group_count = 1 : i64, feature_group_count = %d : i64}\n"
        "         : (%s, %s) -> %s\n"
        "    %%bcast = stablehlo.broadcast_in_dim %%b, dims = [1] : (%s) -> %s\n"
        "    %%out = stablehlo.add %%c, %%bcast : %s\n"
        "    return %%out : %s\n"
        "  }\n"
        "}\n",
        tx.c_str(), tw.c_str(), tb.c_str(), tout.c_str(),
        sH, sW, pH, pH, pW, pW, dH, dW, groups,
        tx.c_str(), tw.c_str(), tout.c_str(),
        tb.c_str(), tout.c_str(),
        tout.c_str(),
        tout.c_str());
    return std::string(buf);
}

// Build Conv3d StableHLO module.
static std::string build_conv3d(
    int N, int InC, int D, int H, int W,
    int OutC, int KD, int KH, int KW,
    int sD, int sH, int sW, int pD, int pH, int pW,
    int dD, int dH, int dW, int groups,
    int Dout, int Hout, int Wout)
{
    int icPg = InC / groups;
    auto tx   = t({N, InC, D, H, W});
    auto tw   = t({OutC, icPg, KD, KH, KW});
    auto tb   = t({OutC});
    auto tout = t({N, OutC, Dout, Hout, Wout});

    char buf[4096];
    snprintf(buf, sizeof(buf),
        "module @conv3d {\n"
        "  func.func @main(%%x: %s, %%w: %s, %%b: %s) -> %s {\n"
        "    %%c = stablehlo.convolution(%%x, %%w)\n"
        "         dim_numbers = [b, f, 0, 1, 2]x[o, i, 0, 1, 2]->[b, f, 0, 1, 2],\n"
        "         window = {stride = [%d, %d, %d],"
                           " pad = [[%d, %d], [%d, %d], [%d, %d]],"
                           " rhs_dilate = [%d, %d, %d]}\n"
        "         {batch_group_count = 1 : i64, feature_group_count = %d : i64}\n"
        "         : (%s, %s) -> %s\n"
        "    %%bcast = stablehlo.broadcast_in_dim %%b, dims = [1] : (%s) -> %s\n"
        "    %%out = stablehlo.add %%c, %%bcast : %s\n"
        "    return %%out : %s\n"
        "  }\n"
        "}\n",
        tx.c_str(), tw.c_str(), tb.c_str(), tout.c_str(),
        sD, sH, sW,
        pD, pD, pH, pH, pW, pW,
        dD, dH, dW, groups,
        tx.c_str(), tw.c_str(), tout.c_str(),
        tb.c_str(), tout.c_str(),
        tout.c_str(),
        tout.c_str());
    return std::string(buf);
}

// Build ConvTranspose2d — expressed as conv with lhs_dilate (transposed conv).
// StableHLO represents transposed convolution via lhs_dilate window attribute.
static std::string build_conv_transpose2d(
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout)
{
    // Weight layout for transposed conv in StableHLO: [InC, OutC/g, KH, KW]
    // dimension_numbers for transposed conv: [b,f,0,1]x[i,o,0,1]->[b,f,0,1]
    // effective padding = KH - 1 - pH (for 'same' after flip)
    int icPg = OutC / groups;
    auto tx   = t({N, InC, H, W});
    auto tw   = t({InC, icPg, KH, KW});
    auto tb   = t({OutC});
    auto tout = t({N, OutC, Hout, Wout});

    int effPadH = dH * (KH - 1) - pH;
    int effPadW = dW * (KW - 1) - pW;

    char buf[4096];
    snprintf(buf, sizeof(buf),
        "module @conv_transpose2d {\n"
        "  func.func @main(%%x: %s, %%w: %s, %%b: %s) -> %s {\n"
        "    %%c = stablehlo.convolution(%%x, %%w)\n"
        "         dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1],\n"
        "         window = {stride = [1, 1],"
                           " pad = [[%d, %d], [%d, %d]],"
                           " lhs_dilate = [%d, %d],"
                           " rhs_dilate = [%d, %d]}\n"
        "         {batch_group_count = 1 : i64, feature_group_count = %d : i64}\n"
        "         : (%s, %s) -> %s\n"
        "    %%bcast = stablehlo.broadcast_in_dim %%b, dims = [1] : (%s) -> %s\n"
        "    %%out = stablehlo.add %%c, %%bcast : %s\n"
        "    return %%out : %s\n"
        "  }\n"
        "}\n",
        tx.c_str(), tw.c_str(), tb.c_str(), tout.c_str(),
        effPadH, effPadH, effPadW, effPadW,
        sH, sW,
        dH, dW,
        groups,
        tx.c_str(), tw.c_str(), tout.c_str(),
        tb.c_str(), tout.c_str(),
        tout.c_str(),
        tout.c_str());
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Dispatch helper
// ---------------------------------------------------------------------------

static int dispatch_conv(
    const std::string& key,
    const std::string& mlir,
    const double* x, int xn, const int64_t* x_dims, int x_ndims,
    const double* w, int wn, const int64_t* w_dims, int w_ndims,
    const double* b, int bn,
    double* dst, int dn)
{
    if (!g_client) return -1;

    // Compile or retrieve cached executable.
    PJRT_LoadedExecutable* exec = nullptr;
    auto it = g_conv_execs.find(key);
    if (it == g_conv_execs.end()) {
        exec = compile_stablehlo_conv(mlir);
        if (!exec) return -1;
        g_conv_execs[key] = exec;
    } else {
        exec = it->second;
    }

    // Create device buffers.
    PJRT_Buffer* bx = make_buffer(x, x_dims, x_ndims);
    PJRT_Buffer* bw = make_buffer(w, w_dims, w_ndims);
    int64_t b_dim[1] = { (int64_t)bn };
    PJRT_Buffer* bb = make_buffer(b, b_dim, 1);
    if (!bx || !bw || !bb) {
        destroy_buffer(bx); destroy_buffer(bw); destroy_buffer(bb);
        return -1;
    }

    int rc = run_conv_exec(exec, bx, bw, bb, dst, dn);
    destroy_buffer(bx); destroy_buffer(bw); destroy_buffer(bb);
    return rc;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

// We share g_api / g_client with activation_xla.cc.
// xla_conv_init just calls through to the same underlying loader.
#include <dlfcn.h>
typedef const PJRT_Api* (*GetPjrtApiFn)();

extern "C" {

int xla_conv_init(const char* platform) {
    // Reuse existing client if already initialized by activation backend.
    if (g_client) return 0;

    char path[256];
    if (strcmp(platform, "gpu") == 0) {
        snprintf(path, sizeof(path), "pjrt_c_api_gpu_plugin.so");
    } else {
        snprintf(path, sizeof(path), "pjrt_c_api_cpu_plugin.so");
    }
    void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) return -1;
    auto get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) return -1;
    const_cast<const PJRT_Api*&>(g_api) = get_api();
    if (!g_api) return -1;

    PJRT_Client_Create_Args ca{};
    ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* err = g_api->PJRT_Client_Create(&ca);
    if (!check(g_api, err, "xla_conv_init")) return -1;
    g_client = ca.client;
    return 0;
}

void xla_conv_shutdown(void) {
    for (auto& kv : g_conv_execs) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = kv.second;
        g_api->PJRT_LoadedExecutable_Destroy(&da);
    }
    g_conv_execs.clear();
}

int xla_conv1d(
    const double* x, double* dst,
    int N, int InC, int L,
    int OutC, int K, int stride, int pad, int dilation, int groups, int L_out,
    const double* weight, const double* bias)
{
    char key[256];
    snprintf(key, sizeof(key), "conv1d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d",
             N, InC, L, OutC, K, stride, pad, dilation, groups, L_out);
    std::string mlir = build_conv1d(N, InC, L, OutC, K, stride, pad, dilation, groups, L_out);
    int64_t xd[3] = {N, InC, L};
    int64_t wd[3] = {OutC, InC/groups, K};
    return dispatch_conv(key, mlir,
        x, N*InC*L, xd, 3,
        weight, OutC*(InC/groups)*K, wd, 3,
        bias, OutC,
        dst, N*OutC*L_out);
}

int xla_conv2d(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight, const double* bias)
{
    char key[256];
    snprintf(key, sizeof(key), "conv2d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d",
             N, InC, H, W, OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout, 0);
    std::string mlir = build_conv2d(N, InC, H, W, OutC, KH, KW,
                                    sH, sW, pH, pW, dH, dW, groups, Hout, Wout);
    int64_t xd[4] = {N, InC, H, W};
    int64_t wd[4] = {OutC, InC/groups, KH, KW};
    return dispatch_conv(key, mlir,
        x, N*InC*H*W, xd, 4,
        weight, OutC*(InC/groups)*KH*KW, wd, 4,
        bias, OutC,
        dst, N*OutC*Hout*Wout);
}

int xla_conv3d(
    const double* x, double* dst,
    int N, int InC, int D, int H, int W,
    int OutC, int KD, int KH, int KW,
    int sD, int sH, int sW, int pD, int pH, int pW,
    int dD, int dH, int dW, int groups,
    int Dout, int Hout, int Wout,
    const double* weight, const double* bias)
{
    char key[256];
    snprintf(key, sizeof(key), "conv3d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d",
             N, InC, D, H, W, OutC, KD, KH, KW,
             sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, Dout, Hout, Wout, 0);
    std::string mlir = build_conv3d(N, InC, D, H, W, OutC, KD, KH, KW,
                                    sD, sH, sW, pD, pH, pW, dD, dH, dW,
                                    groups, Dout, Hout, Wout);
    int64_t xd[5] = {N, InC, D, H, W};
    int64_t wd[5] = {OutC, InC/groups, KD, KH, KW};
    return dispatch_conv(key, mlir,
        x, N*InC*D*H*W, xd, 5,
        weight, OutC*(InC/groups)*KD*KH*KW, wd, 5,
        bias, OutC,
        dst, N*OutC*Dout*Hout*Wout);
}

int xla_conv_transpose2d(
    const double* x, double* dst,
    int N, int InC, int H, int W,
    int OutC, int KH, int KW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int Hout, int Wout,
    const double* weight, const double* bias)
{
    char key[256];
    snprintf(key, sizeof(key), "convt2d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d",
             N, InC, H, W, OutC, KH, KW, sH, sW, pH, pW, dH, dW, groups, Hout, Wout, 0);
    std::string mlir = build_conv_transpose2d(N, InC, H, W, OutC, KH, KW,
                                              sH, sW, pH, pW, dH, dW, groups,
                                              Hout, Wout);
    int64_t xd[4] = {N, InC, H, W};
    int64_t wd[4] = {InC, OutC/groups, KH, KW};
    return dispatch_conv(key, mlir,
        x, N*InC*H*W, xd, 4,
        weight, InC*(OutC/groups)*KH*KW, wd, 4,
        bias, OutC,
        dst, N*OutC*Hout*Wout);
}

} // extern "C"
