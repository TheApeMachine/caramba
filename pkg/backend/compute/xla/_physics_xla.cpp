// XLA physics-stencil backend — PJRT C API implementation.
//
// Each operation builds a StableHLO module that expresses the periodic
// Laplacian as: shifted-left + shifted-right - 2*src (1D), or the equivalent
// sum-of-axis-neighbours - 2*rank*src (2D/3D), times an inv_h2 scalar input.
//
// Periodic shifts use stablehlo.slice + stablehlo.concatenate to perform a
// roll without dynamic indexing. The cache key includes only shape, not the
// inv_h2 value, because inv_h2 is plumbed as a runtime scalar input.
//
// Compile/run helpers mirror the conventions used by _positional_xla.cpp.

#include "xla_physics.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"
#include <dlfcn.h>

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static const PJRT_Api*    gph_api    = nullptr;
static PJRT_Client*       gph_client = nullptr;
static std::unordered_map<std::string, PJRT_LoadedExecutable*> gph_execs;

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

static void physics_free_error(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
}

static bool physics_check(const PJRT_Api* api, PJRT_Error* err, const char* ctx) {
    if (!err) return true;
    PJRT_Error_Message_Args ma{};
    ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    ma.error = err;
    api->PJRT_Error_Message(&ma);
    fprintf(stderr, "XLA physics error in %s: %.*s\n",
            ctx, (int)ma.message_size, ma.message);
    physics_free_error(api, err);
    return false;
}

// ---------------------------------------------------------------------------
// Plugin loader
// ---------------------------------------------------------------------------

typedef const PJRT_Api* (*PhysicsGetPjrtApiFn)();

static const PJRT_Api* physics_load_pjrt_plugin(const char* platform) {
    std::string path = pjrt_plugin_path(platform);
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "XLA physics: dlopen %s failed: %s\n", path.c_str(), dlerror());
        return nullptr;
    }
    auto fn = (PhysicsGetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!fn) return nullptr;
    return fn();
}

// ---------------------------------------------------------------------------
// StableHLO module builders
// ---------------------------------------------------------------------------

// Periodic 1D Laplacian:
//   left  = concat(src[n-1:n], src[0:n-1])
//   right = concat(src[1:n],   src[0:1])
//   out   = (left + right - 2*src) * inv_h2
static std::string build_laplacian_1d_hlo(int n) {
    std::string tFull = "tensor<" + std::to_string(n) + "xf64>";
    std::string tOne  = "tensor<1xf64>";
    std::string tHead = "tensor<" + std::to_string(n - 1) + "xf64>";
    std::string tScal = "tensor<f64>";

    std::string n_str  = std::to_string(n);
    std::string nm1    = std::to_string(n - 1);

    return
        "module @laplacian_1d {\n"
        "  func.func @main(%src: " + tFull + ", %inv_h2: " + tScal + ") -> " + tFull + " {\n"
        "    %tail = stablehlo.slice %src [" + nm1 + ":" + n_str + "] : (" + tFull + ") -> " + tOne + "\n"
        "    %head = stablehlo.slice %src [0:" + nm1 + "] : (" + tFull + ") -> " + tHead + "\n"
        "    %left = stablehlo.concatenate %tail, %head, dim = 0 : (" + tOne + ", " + tHead + ") -> " + tFull + "\n"
        "    %rmain = stablehlo.slice %src [1:" + n_str + "] : (" + tFull + ") -> " + tHead + "\n"
        "    %rwrap = stablehlo.slice %src [0:1] : (" + tFull + ") -> " + tOne + "\n"
        "    %right = stablehlo.concatenate %rmain, %rwrap, dim = 0 : (" + tHead + ", " + tOne + ") -> " + tFull + "\n"
        "    %sum = stablehlo.add %left, %right : " + tFull + "\n"
        "    %two = stablehlo.constant dense<2.0> : " + tFull + "\n"
        "    %twoc = stablehlo.multiply %two, %src : " + tFull + "\n"
        "    %diff = stablehlo.subtract %sum, %twoc : " + tFull + "\n"
        "    %inv_h2_bc = stablehlo.broadcast_in_dim %inv_h2, dims = [] : (" + tScal + ") -> " + tFull + "\n"
        "    %out = stablehlo.multiply %inv_h2_bc, %diff : " + tFull + "\n"
        "    return %out : " + tFull + "\n"
        "  }\n"
        "}\n";
}

// Periodic 2D 5-point Laplacian on [H, W].
//   shift_x: concat(src[:, W-1:W], src[:, 0:W-1]) along dim 1  (left neighbour)
//   shift_y: concat(src[H-1:H, :], src[0:H-1, :]) along dim 0  (up neighbour)
//   ...and their +1 counterparts (right, down)
//   out = (left + right + up + down - 4*src) * inv_h2
static std::string build_laplacian_2d_hlo(int H, int W) {
    std::string tFull = "tensor<" + std::to_string(H) + "x" + std::to_string(W) + "xf64>";
    std::string tColOne  = "tensor<" + std::to_string(H) + "x1xf64>";
    std::string tColHead = "tensor<" + std::to_string(H) + "x" + std::to_string(W - 1) + "xf64>";
    std::string tRowOne  = "tensor<1x" + std::to_string(W) + "xf64>";
    std::string tRowHead = "tensor<" + std::to_string(H - 1) + "x" + std::to_string(W) + "xf64>";
    std::string tScal    = "tensor<f64>";

    std::string H_str = std::to_string(H);
    std::string W_str = std::to_string(W);
    std::string Hm1   = std::to_string(H - 1);
    std::string Wm1   = std::to_string(W - 1);

    return
        "module @laplacian_2d {\n"
        "  func.func @main(%src: " + tFull + ", %inv_h2: " + tScal + ") -> " + tFull + " {\n"
        "    %left_tail = stablehlo.slice %src [0, " + Wm1 + "] [" + H_str + ", " + W_str + "] [1, 1] : (" + tFull + ") -> " + tColOne + "\n"
        "    %left_head = stablehlo.slice %src [0, 0] [" + H_str + ", " + Wm1 + "] [1, 1] : (" + tFull + ") -> " + tColHead + "\n"
        "    %left = stablehlo.concatenate %left_tail, %left_head, dim = 1 : (" + tColOne + ", " + tColHead + ") -> " + tFull + "\n"
        "    %right_main = stablehlo.slice %src [0, 1] [" + H_str + ", " + W_str + "] [1, 1] : (" + tFull + ") -> " + tColHead + "\n"
        "    %right_wrap = stablehlo.slice %src [0, 0] [" + H_str + ", 1] [1, 1] : (" + tFull + ") -> " + tColOne + "\n"
        "    %right = stablehlo.concatenate %right_main, %right_wrap, dim = 1 : (" + tColHead + ", " + tColOne + ") -> " + tFull + "\n"
        "    %up_tail = stablehlo.slice %src [" + Hm1 + ", 0] [" + H_str + ", " + W_str + "] [1, 1] : (" + tFull + ") -> " + tRowOne + "\n"
        "    %up_head = stablehlo.slice %src [0, 0] [" + Hm1 + ", " + W_str + "] [1, 1] : (" + tFull + ") -> " + tRowHead + "\n"
        "    %up = stablehlo.concatenate %up_tail, %up_head, dim = 0 : (" + tRowOne + ", " + tRowHead + ") -> " + tFull + "\n"
        "    %down_main = stablehlo.slice %src [1, 0] [" + H_str + ", " + W_str + "] [1, 1] : (" + tFull + ") -> " + tRowHead + "\n"
        "    %down_wrap = stablehlo.slice %src [0, 0] [1, " + W_str + "] [1, 1] : (" + tFull + ") -> " + tRowOne + "\n"
        "    %down = stablehlo.concatenate %down_main, %down_wrap, dim = 0 : (" + tRowHead + ", " + tRowOne + ") -> " + tFull + "\n"
        "    %h_sum = stablehlo.add %left, %right : " + tFull + "\n"
        "    %v_sum = stablehlo.add %up, %down : " + tFull + "\n"
        "    %sum = stablehlo.add %h_sum, %v_sum : " + tFull + "\n"
        "    %four = stablehlo.constant dense<4.0> : " + tFull + "\n"
        "    %four_c = stablehlo.multiply %four, %src : " + tFull + "\n"
        "    %diff = stablehlo.subtract %sum, %four_c : " + tFull + "\n"
        "    %inv_h2_bc = stablehlo.broadcast_in_dim %inv_h2, dims = [] : (" + tScal + ") -> " + tFull + "\n"
        "    %out = stablehlo.multiply %inv_h2_bc, %diff : " + tFull + "\n"
        "    return %out : " + tFull + "\n"
        "  }\n"
        "}\n";
}

// Periodic 3D 7-point Laplacian on [D, H, W].
// Same idea: six axis-aligned periodic shifts, subtract 6*src, scale by inv_h2.
static std::string build_laplacian_3d_hlo(int D, int H, int W) {
    std::string D_s = std::to_string(D);
    std::string H_s = std::to_string(H);
    std::string W_s = std::to_string(W);
    std::string Dm1 = std::to_string(D - 1);
    std::string Hm1 = std::to_string(H - 1);
    std::string Wm1 = std::to_string(W - 1);

    std::string tFull   = "tensor<" + D_s + "x" + H_s + "x" + W_s + "xf64>";
    std::string tColOne = "tensor<" + D_s + "x" + H_s + "x1xf64>";
    std::string tColHd  = "tensor<" + D_s + "x" + H_s + "x" + Wm1 + "xf64>";
    std::string tRowOne = "tensor<" + D_s + "x1x" + W_s + "xf64>";
    std::string tRowHd  = "tensor<" + D_s + "x" + Hm1 + "x" + W_s + "xf64>";
    std::string tDepOne = "tensor<1x" + H_s + "x" + W_s + "xf64>";
    std::string tDepHd  = "tensor<" + Dm1 + "x" + H_s + "x" + W_s + "xf64>";
    std::string tScal   = "tensor<f64>";

    return
        "module @laplacian_3d {\n"
        "  func.func @main(%src: " + tFull + ", %inv_h2: " + tScal + ") -> " + tFull + " {\n"
        // x-axis shifts (along W)
        "    %left_tail = stablehlo.slice %src [0, 0, " + Wm1 + "] [" + D_s + ", " + H_s + ", " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tColOne + "\n"
        "    %left_head = stablehlo.slice %src [0, 0, 0] [" + D_s + ", " + H_s + ", " + Wm1 + "] [1, 1, 1] : (" + tFull + ") -> " + tColHd + "\n"
        "    %left = stablehlo.concatenate %left_tail, %left_head, dim = 2 : (" + tColOne + ", " + tColHd + ") -> " + tFull + "\n"
        "    %right_main = stablehlo.slice %src [0, 0, 1] [" + D_s + ", " + H_s + ", " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tColHd + "\n"
        "    %right_wrap = stablehlo.slice %src [0, 0, 0] [" + D_s + ", " + H_s + ", 1] [1, 1, 1] : (" + tFull + ") -> " + tColOne + "\n"
        "    %right = stablehlo.concatenate %right_main, %right_wrap, dim = 2 : (" + tColHd + ", " + tColOne + ") -> " + tFull + "\n"
        // y-axis shifts (along H)
        "    %up_tail = stablehlo.slice %src [0, " + Hm1 + ", 0] [" + D_s + ", " + H_s + ", " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tRowOne + "\n"
        "    %up_head = stablehlo.slice %src [0, 0, 0] [" + D_s + ", " + Hm1 + ", " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tRowHd + "\n"
        "    %up = stablehlo.concatenate %up_tail, %up_head, dim = 1 : (" + tRowOne + ", " + tRowHd + ") -> " + tFull + "\n"
        "    %down_main = stablehlo.slice %src [0, 1, 0] [" + D_s + ", " + H_s + ", " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tRowHd + "\n"
        "    %down_wrap = stablehlo.slice %src [0, 0, 0] [" + D_s + ", 1, " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tRowOne + "\n"
        "    %down = stablehlo.concatenate %down_main, %down_wrap, dim = 1 : (" + tRowHd + ", " + tRowOne + ") -> " + tFull + "\n"
        // z-axis shifts (along D)
        "    %front_tail = stablehlo.slice %src [" + Dm1 + ", 0, 0] [" + D_s + ", " + H_s + ", " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tDepOne + "\n"
        "    %front_head = stablehlo.slice %src [0, 0, 0] [" + Dm1 + ", " + H_s + ", " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tDepHd + "\n"
        "    %front = stablehlo.concatenate %front_tail, %front_head, dim = 0 : (" + tDepOne + ", " + tDepHd + ") -> " + tFull + "\n"
        "    %back_main = stablehlo.slice %src [1, 0, 0] [" + D_s + ", " + H_s + ", " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tDepHd + "\n"
        "    %back_wrap = stablehlo.slice %src [0, 0, 0] [1, " + H_s + ", " + W_s + "] [1, 1, 1] : (" + tFull + ") -> " + tDepOne + "\n"
        "    %back = stablehlo.concatenate %back_main, %back_wrap, dim = 0 : (" + tDepHd + ", " + tDepOne + ") -> " + tFull + "\n"
        // accumulate
        "    %h_sum = stablehlo.add %left, %right : " + tFull + "\n"
        "    %v_sum = stablehlo.add %up, %down : " + tFull + "\n"
        "    %t_sum = stablehlo.add %front, %back : " + tFull + "\n"
        "    %hv = stablehlo.add %h_sum, %v_sum : " + tFull + "\n"
        "    %sum = stablehlo.add %hv, %t_sum : " + tFull + "\n"
        "    %six = stablehlo.constant dense<6.0> : " + tFull + "\n"
        "    %six_c = stablehlo.multiply %six, %src : " + tFull + "\n"
        "    %diff = stablehlo.subtract %sum, %six_c : " + tFull + "\n"
        "    %inv_h2_bc = stablehlo.broadcast_in_dim %inv_h2, dims = [] : (" + tScal + ") -> " + tFull + "\n"
        "    %out = stablehlo.multiply %inv_h2_bc, %diff : " + tFull + "\n"
        "    return %out : " + tFull + "\n"
        "  }\n"
        "}\n";
}

// ---------------------------------------------------------------------------
// Compile + execute helpers
// ---------------------------------------------------------------------------

static PJRT_LoadedExecutable* physics_compile(const std::string& key, const std::string& hlo) {
    auto it = gph_execs.find(key);
    if (it != gph_execs.end()) return it->second;

    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = const_cast<char*>(hlo.c_str());
    prog.code_size   = hlo.size();
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client      = gph_client;
    ca.program     = &prog;
    set_single_device_compile_options(&ca);

    PJRT_Error* err = gph_api->PJRT_Client_Compile(&ca);
    if (!physics_check(gph_api, err, "PJRT_Client_Compile")) return nullptr;

    gph_execs[key] = ca.executable;
    return ca.executable;
}

static PJRT_Buffer* physics_make_buf(const double* host, const std::vector<int64_t>& dims) {
    PJRT_Client_BufferFromHostBuffer_Args ba{};
    ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client      = gph_client;
    ba.data        = host;
    ba.type        = PJRT_Buffer_Type_F64;
    ba.dims        = const_cast<int64_t*>(dims.data());
    ba.num_dims    = (int)dims.size();
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;

    PJRT_Error* err = gph_api->PJRT_Client_BufferFromHostBuffer(&ba);
    if (!physics_check(gph_api, err, "BufferFromHostBuffer")) return nullptr;

    PJRT_Buffer_ReadyEvent_Args re{};
    re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    re.buffer      = ba.buffer;
    err = gph_api->PJRT_Buffer_ReadyEvent(&re);
    physics_free_error(gph_api, err);

    PJRT_Event_Await_Args ea{};
    ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea.event       = re.event;
    err = gph_api->PJRT_Event_Await(&ea);
    physics_free_error(gph_api, err);

    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event       = re.event;
    gph_api->PJRT_Event_Destroy(&eda);

    return ba.buffer;
}

static void physics_destroy_buf(PJRT_Buffer* buf) {
    if (!buf) return;
    PJRT_Buffer_Destroy_Args da{};
    da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    da.buffer      = buf;
    gph_api->PJRT_Buffer_Destroy(&da);
}

static int physics_copy_to_host(PJRT_Buffer* buf, double* host, size_t bytes) {
    PJRT_Buffer_ToHostBuffer_Args ta{};
    ta.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    ta.src         = buf;
    ta.dst         = host;
    ta.dst_size    = bytes;

    PJRT_Error* err = gph_api->PJRT_Buffer_ToHostBuffer(&ta);
    if (!physics_check(gph_api, err, "ToHostBuffer")) return -1;

    PJRT_Event_Await_Args ea{};
    ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea.event       = ta.event;
    err = gph_api->PJRT_Event_Await(&ea);
    physics_free_error(gph_api, err);

    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event       = ta.event;
    gph_api->PJRT_Event_Destroy(&eda);
    return 0;
}

static int physics_run(
    PJRT_LoadedExecutable* exec,
    const double* src, const std::vector<int64_t>& src_dims,
    double inv_h2,
    double* dst, size_t dst_bytes)
{
    PJRT_Buffer* src_buf = physics_make_buf(src, src_dims);
    if (!src_buf) return -1;

    std::vector<int64_t> scalar_dims; // rank-0 scalar
    PJRT_Buffer* iv_buf = physics_make_buf(&inv_h2, scalar_dims);
    if (!iv_buf) {
        physics_destroy_buf(src_buf);
        return -1;
    }

    PJRT_Buffer* in_bufs[2] = { src_buf, iv_buf };

    PJRT_LoadedExecutable_Execute_Args xa{};
    xa.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    xa.executable  = exec;
    PJRT_ExecuteOptions options = single_device_execute_options();
    xa.options     = &options;
    PJRT_Buffer* const* arg_lists[1] = { in_bufs };
    xa.argument_lists = arg_lists;
    xa.num_devices    = 1;
    xa.num_args       = 2;
    PJRT_Buffer* out_bufs[1] = {};
    PJRT_Buffer** out_lists[1] = { out_bufs };
    xa.output_lists = out_lists;

    PJRT_Error* xerr = gph_api->PJRT_LoadedExecutable_Execute(&xa);
    if (!physics_check(gph_api, xerr, "PJRT_LoadedExecutable_Execute")) {
        physics_destroy_buf(src_buf);
        physics_destroy_buf(iv_buf);
        return -1;
    }

    int rc = physics_copy_to_host(out_bufs[0], dst, dst_bytes);

    physics_destroy_buf(src_buf);
    physics_destroy_buf(iv_buf);
    physics_destroy_buf(out_bufs[0]);
    return rc;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

int xla_physics_init(const char* platform) {
    if (gph_api && gph_client) return 0;

    gph_api = physics_load_pjrt_plugin(platform);
    if (!gph_api) return -1;

    PJRT_Client_Create_Args args{};
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* err = gph_api->PJRT_Client_Create(&args);
    if (!physics_check(gph_api, err, "PJRT_Client_Create")) return -1;
    gph_client = args.client;
    return 0;
}

void xla_physics_shutdown(void) {
    if (!gph_api) return;

    for (auto& entry : gph_execs) {
        if (!entry.second) continue;
        PJRT_LoadedExecutable_Destroy_Args ea{};
        ea.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        ea.executable  = entry.second;
        gph_api->PJRT_LoadedExecutable_Destroy(&ea);
    }
    gph_execs.clear();

    if (gph_client) {
        PJRT_Client_Destroy_Args da{};
        da.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        da.client      = gph_client;
        gph_api->PJRT_Client_Destroy(&da);
        gph_client = nullptr;
    }
}

int xla_laplacian_1d(const double* src, double* dst, int n, double inv_h2) {
    if (n < 2 || !src || !dst) return -1;
    std::string key = "lap_1d_" + std::to_string(n);
    auto* exec = physics_compile(key, build_laplacian_1d_hlo(n));
    if (!exec) return -1;
    std::vector<int64_t> dims = { (int64_t)n };
    return physics_run(exec, src, dims, inv_h2, dst, (size_t)n * sizeof(double));
}

int xla_laplacian_2d(const double* src, double* dst, int H, int W, double inv_h2) {
    if (H < 2 || W < 2 || !src || !dst) return -1;
    std::string key = "lap_2d_" + std::to_string(H) + "_" + std::to_string(W);
    auto* exec = physics_compile(key, build_laplacian_2d_hlo(H, W));
    if (!exec) return -1;
    std::vector<int64_t> dims = { (int64_t)H, (int64_t)W };
    return physics_run(exec, src, dims, inv_h2, dst, (size_t)H * (size_t)W * sizeof(double));
}

int xla_laplacian_3d(const double* src, double* dst, int D, int H, int W, double inv_h2) {
    if (D < 2 || H < 2 || W < 2 || !src || !dst) return -1;
    std::string key = "lap_3d_" + std::to_string(D) + "_" + std::to_string(H) + "_" + std::to_string(W);
    auto* exec = physics_compile(key, build_laplacian_3d_hlo(D, H, W));
    if (!exec) return -1;
    std::vector<int64_t> dims = { (int64_t)D, (int64_t)H, (int64_t)W };
    return physics_run(exec, src, dims, inv_h2, dst, (size_t)D * (size_t)H * (size_t)W * sizeof(double));
}
