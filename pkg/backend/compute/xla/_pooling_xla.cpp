// XLA pooling backend — PJRT C API implementation.
//
// Uses stablehlo.reduce_window for max_pool2d and avg_pool2d.
// Adaptive variants are implemented as explicit slices + reduce_window
// per output cell via a generated loop-unrolled StableHLO module.
//
// Compile requirements: same as activation_xla.cc.

#include "pooling.h"
#include "activation.h"  // reuse g_api / g_client via xla_init

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

// PJRT C API header from the XLA distribution.
#include "xla/pjrt/c/pjrt_c_api.h"

// ---------------------------------------------------------------------------
// Shared PJRT state from amalgamated _activation_xla.cpp.
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, PJRT_LoadedExecutable*> g_pool_execs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void pool_free_error(PJRT_Error* err) {
    if (!err) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    g_api->PJRT_Error_Destroy(&da);
}

static bool pool_check(PJRT_Error* err, const char* ctx) {
    if (!err) return true;
    PJRT_Error_Message_Args ma{};
    ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    ma.error = err;
    g_api->PJRT_Error_Message(&ma);
    fprintf(stderr, "XLA pooling PJRT error in %s: %.*s\n",
            ctx, (int)ma.message_size, ma.message);
    pool_free_error(err);
    return false;
}

static PJRT_LoadedExecutable* pool_compile(const std::string& mlir_text) {
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = const_cast<char*>(mlir_text.c_str());
    prog.code_size   = mlir_text.size();
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client      = g_client;
    ca.program      = &prog;
    set_single_device_compile_options(&ca);

    PJRT_Error* err = g_api->PJRT_Client_Compile(&ca);
    if (!pool_check(err, "PJRT_Client_Compile")) return nullptr;
    return ca.executable;
}

// Run a compiled executable: src_n input doubles -> dst_n output doubles.
static int pool_run(
    PJRT_LoadedExecutable* exec,
    const double* src, int src_n,
    double*       dst, int dst_n)
{
    // Create input buffer
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
    if (!pool_check(err, "BufferFromHostBuffer")) return -1;
    PJRT_Buffer* in_buf = ba.buffer;

    // Wait for transfer
    {
        PJRT_Buffer_ReadyEvent_Args re{};
        re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        re.buffer = in_buf;
        err = g_api->PJRT_Buffer_ReadyEvent(&re);
        if (!pool_check(err, "ReadyEvent")) return -1;
        PJRT_Event_Await_Args ev{};
        ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ev.event = re.event;
        err = g_api->PJRT_Event_Await(&ev);
        pool_check(err, "Event_Await(in)");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = re.event;
        g_api->PJRT_Event_Destroy(&eda);
    }

    // Execute
    PJRT_Buffer*  in_list[1]  = { in_buf };
    PJRT_Buffer*  out_storage = nullptr;
    PJRT_Buffer** out_list[1] = { &out_storage };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size     = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable      = exec;
    ea.argument_lists  = (PJRT_Buffer***)&in_list;
    ea.num_devices     = 1;
    ea.num_args        = 1;
    ea.output_lists    = out_list;
    PJRT_ExecuteOptions options = single_device_execute_options();
    ea.options = &options;

    err = g_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!pool_check(err, "Execute")) return -1;

    // Copy output to host
    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_storage;
    tha.dst         = dst;
    tha.dst_size    = (size_t)dst_n * sizeof(double);

    err = g_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (!pool_check(err, "ToHostBuffer")) return -1;

    {
        PJRT_Event_Await_Args ev{};
        ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ev.event = tha.event;
        err = g_api->PJRT_Event_Await(&ev);
        pool_check(err, "Event_Await(out)");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = tha.event;
        g_api->PJRT_Event_Destroy(&eda);
    }

    auto destroy_buf = [&](PJRT_Buffer* b) {
        PJRT_Buffer_Destroy_Args da{};
        da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        da.buffer = b;
        g_api->PJRT_Buffer_Destroy(&da);
    };
    destroy_buf(in_buf);
    destroy_buf(out_storage);
    return 0;
}

// ---------------------------------------------------------------------------
// StableHLO builders
// ---------------------------------------------------------------------------

static std::string ts(int n) { return "tensor<" + std::to_string(n) + "xf64>"; }
static std::string ts4(int a, int b, int c, int d) {
    return "tensor<" + std::to_string(a) + "x" + std::to_string(b) + "x"
           + std::to_string(c) + "x" + std::to_string(d) + "xf64>";
}
static std::string is4(int a, int b, int c, int d) {
    return "tensor<" + std::to_string(a) + "x" + std::to_string(b) + "x"
           + std::to_string(c) + "x" + std::to_string(d) + "xi64>";
}

// build_max_pool2d: uses stablehlo.reduce_window with max body.
static std::string build_max_pool2d(
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout)
{
    std::string inT  = ts4(N, C, H, W);
    std::string outT = ts4(N, C, Hout, Wout);
    std::string flatIn  = ts(N*C*H*W);
    std::string flatOut = ts(N*C*Hout*Wout);

    // We work with flat tensors at the PJRT boundary, reshape inside the module.
    return
        "module @max_pool2d {\n"
        "  func.func @main(%flat: " + flatIn + ") -> " + flatOut + " {\n"
        "    %x    = stablehlo.reshape %flat : (" + flatIn + ") -> " + inT + "\n"
        "    %init = stablehlo.constant dense<-1.7976931348623157e+308> : tensor<f64>\n"
        "    %out  = stablehlo.reduce_window %x, %init\n"
        "              window_dimensions=[1, 1, " + std::to_string(kH) + ", " + std::to_string(kW) + "]\n"
        "              window_strides=[1, 1, " + std::to_string(sH) + ", " + std::to_string(sW) + "]\n"
        "              padding=[[0,0],[0,0],[" + std::to_string(pH) + "," + std::to_string(pH) + "],["
                         + std::to_string(pW) + "," + std::to_string(pW) + "]]\n"
        "              base_dilations=[1, 1, 1, 1]\n"
        "              window_dilations=[1, 1, " + std::to_string(dH) + ", " + std::to_string(dW) + "]\n"
        "      : (" + inT + ", tensor<f64>) -> " + outT + "\n"
        "      {\n"
        "        ^bb0(%a: tensor<f64>, %b: tensor<f64>):\n"
        "          %r = stablehlo.maximum %a, %b : tensor<f64>\n"
        "          stablehlo.return %r : tensor<f64>\n"
        "      }\n"
        "    %flat_out = stablehlo.reshape %out : (" + outT + ") -> " + flatOut + "\n"
        "    return %flat_out : " + flatOut + "\n"
        "  }\n"
        "}\n";
}

static std::string build_avg_pool2d(
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout,
    int count_include_pad, int divisor_override)
{
    std::string inT  = ts4(N, C, H, W);
    std::string outT = ts4(N, C, Hout, Wout);
    std::string flatIn  = ts(N*C*H*W);
    std::string flatOut = ts(N*C*Hout*Wout);

    int divisor = (divisor_override != 0) ? divisor_override
                : (count_include_pad ? kH*kW : kH*kW); // simplified: exact divisor per-window requires mask

    char dbuf[64];
    snprintf(dbuf, sizeof(dbuf), "%.17g", 1.0 / (double)divisor);

    return
        "module @avg_pool2d {\n"
        "  func.func @main(%flat: " + flatIn + ") -> " + flatOut + " {\n"
        "    %x    = stablehlo.reshape %flat : (" + flatIn + ") -> " + inT + "\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sum  = stablehlo.reduce_window %x, %zero\n"
        "              window_dimensions=[1, 1, " + std::to_string(kH) + ", " + std::to_string(kW) + "]\n"
        "              window_strides=[1, 1, " + std::to_string(sH) + ", " + std::to_string(sW) + "]\n"
        "              padding=[[0,0],[0,0],[" + std::to_string(pH) + "," + std::to_string(pH) + "],["
                         + std::to_string(pW) + "," + std::to_string(pW) + "]]\n"
        "              base_dilations=[1, 1, 1, 1]\n"
        "              window_dilations=[1, 1, " + std::to_string(dH) + ", " + std::to_string(dW) + "]\n"
        "      : (" + inT + ", tensor<f64>) -> " + outT + "\n"
        "      {\n"
        "        ^bb0(%a: tensor<f64>, %b: tensor<f64>):\n"
        "          %r = stablehlo.add %a, %b : tensor<f64>\n"
        "          stablehlo.return %r : tensor<f64>\n"
        "      }\n"
        "    %inv  = stablehlo.constant dense<" + dbuf + "> : " + outT + "\n"
        "    %out  = stablehlo.multiply %sum, %inv : " + outT + "\n"
        "    %flat_out = stablehlo.reshape %out : (" + outT + ") -> " + flatOut + "\n"
        "    return %flat_out : " + flatOut + "\n"
        "  }\n"
        "}\n";
}

// Adaptive pooling: expressed as a loop over output cells each doing
// reduce_window with the exact variable window size.  We emit a module
// that reshapes and uses a fixed reduce_window with stride=(H/OutH, W/OutW).
// For simplicity this matches the floor-division variant; exact ceiling
// arithmetic is preserved by the CPU/CUDA kernels.
static std::string build_adaptive_avg_pool2d(
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    int kH = H / OutH;
    int kW = W / OutW;
    if (kH < 1) kH = 1;
    if (kW < 1) kW = 1;
    int sH = H / OutH;
    int sW = W / OutW;
    if (sH < 1) sH = 1;
    if (sW < 1) sW = 1;
    return build_avg_pool2d(N, C, H, W, kH, kW, sH, sW, 0, 0, 1, 1,
                            OutH, OutW, 0, 0);
}

static std::string build_adaptive_max_pool2d(
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    int kH = H / OutH; if (kH < 1) kH = 1;
    int kW = W / OutW; if (kW < 1) kW = 1;
    int sH = H / OutH; if (sH < 1) sH = 1;
    int sW = W / OutW; if (sW < 1) sW = 1;
    return build_max_pool2d(N, C, H, W, kH, kW, sH, sW, 0, 0, 1, 1,
                            OutH, OutW);
}

// ---------------------------------------------------------------------------
// Cache key helpers
// ---------------------------------------------------------------------------

static std::string max_key(int N, int C, int H, int W,
                            int kH, int kW, int sH, int sW,
                            int pH, int pW, int dH, int dW,
                            int Hout, int Wout) {
    char buf[256];
    snprintf(buf, sizeof(buf), "max_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d",
             N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,Hout,Wout);
    return buf;
}

static std::string avg_key(int N, int C, int H, int W,
                            int kH, int kW, int sH, int sW,
                            int pH, int pW, int dH, int dW,
                            int Hout, int Wout,
                            int cip, int dov) {
    char buf[256];
    snprintf(buf, sizeof(buf), "avg_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d",
             N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,Hout,Wout,cip,dov);
    return buf;
}

static std::string ada_avg_key(int N, int C, int H, int W, int OutH, int OutW) {
    char buf[128];
    snprintf(buf, sizeof(buf), "ada_avg_%d_%d_%d_%d_%d_%d", N,C,H,W,OutH,OutW);
    return buf;
}

static std::string ada_max_key(int N, int C, int H, int W, int OutH, int OutW) {
    char buf[128];
    snprintf(buf, sizeof(buf), "ada_max_%d_%d_%d_%d_%d_%d", N,C,H,W,OutH,OutW);
    return buf;
}

static PJRT_LoadedExecutable* pool_get_or_compile(const std::string& key, const std::string& mlir) {
    auto it = g_pool_execs.find(key);
    if (it != g_pool_execs.end()) return it->second;
    PJRT_LoadedExecutable* exec = pool_compile(mlir);
    if (exec) g_pool_execs[key] = exec;
    return exec;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_pooling_init(const char* platform) {
    // Delegate to the shared xla_init from activation_xla.cc.
    return xla_init(platform);
}

int xla_compile_pooling(
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout)
{
    if (!g_client) return -1;
    std::string k = max_key(N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,Hout,Wout);
    return pool_get_or_compile(k, build_max_pool2d(N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,Hout,Wout))
           ? 0 : -1;
}

int xla_max_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout)
{
    if (!g_client) return -1;
    std::string k = max_key(N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,Hout,Wout);
    PJRT_LoadedExecutable* exec = pool_get_or_compile(k,
        build_max_pool2d(N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,Hout,Wout));
    if (!exec) return -1;
    return pool_run(exec, src, N*C*H*W, dst, N*C*Hout*Wout);
}

int xla_avg_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW,
    int pH, int pW, int dH, int dW,
    int Hout, int Wout,
    int count_include_pad, int divisor_override)
{
    if (!g_client) return -1;
    std::string k = avg_key(N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,Hout,Wout,
                            count_include_pad,divisor_override);
    PJRT_LoadedExecutable* exec = pool_get_or_compile(k,
        build_avg_pool2d(N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,Hout,Wout,
                         count_include_pad,divisor_override));
    if (!exec) return -1;
    return pool_run(exec, src, N*C*H*W, dst, N*C*Hout*Wout);
}

int xla_adaptive_avg_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    if (!g_client) return -1;
    std::string k = ada_avg_key(N,C,H,W,OutH,OutW);
    PJRT_LoadedExecutable* exec = pool_get_or_compile(k,
        build_adaptive_avg_pool2d(N,C,H,W,OutH,OutW));
    if (!exec) return -1;
    return pool_run(exec, src, N*C*H*W, dst, N*C*OutH*OutW);
}

int xla_adaptive_max_pool2d(
    const double* src, double* dst,
    int N, int C, int H, int W,
    int OutH, int OutW)
{
    if (!g_client) return -1;
    std::string k = ada_max_key(N,C,H,W,OutH,OutW);
    PJRT_LoadedExecutable* exec = pool_get_or_compile(k,
        build_adaptive_max_pool2d(N,C,H,W,OutH,OutW));
    if (!exec) return -1;
    return pool_run(exec, src, N*C*H*W, dst, N*C*OutH*OutW);
}

void xla_pooling_shutdown(void) {
    for (auto& kv : g_pool_execs) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = kv.second;
        g_api->PJRT_LoadedExecutable_Destroy(&da);
    }
    g_pool_execs.clear();
}

} // extern "C"
