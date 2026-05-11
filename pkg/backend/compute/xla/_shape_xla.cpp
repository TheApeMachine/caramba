// XLA shape backend — PJRT C API implementation.
//
// Build requirements: same as activation_xla.cc.
// These functions reuse the global g_api / g_client established by xla_init().

#include "shape.h"
#include "activation.h"  // for PJRT types and xla_init / run_executable helper

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>

// g_api / g_client / set_single_device_compile_options come from amalgamated
// _activation_xla.cpp (included earlier in xla_sources.cpp).

#include "xla/pjrt/c/pjrt_c_api.h"

static PJRT_LoadedExecutable* compile_shape_stablehlo(const std::string& mlir_text) {
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
    if (err) {
        PJRT_Error_Message_Args ma{};
        ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        ma.error = err;
        g_api->PJRT_Error_Message(&ma);
        fprintf(stderr, "XLA shape compile error: %.*s\n",
                (int)ma.message_size, ma.message);
        PJRT_Error_Destroy_Args da{};
        da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        da.error = err;
        g_api->PJRT_Error_Destroy(&da);
        return nullptr;
    }
    return ca.executable;
}

// ---------------------------------------------------------------------------
// run_shape_executable: transfer src -> device, run, device -> dst.
// Identical logic to activation_xla.cc run_executable but declared locally.
// ---------------------------------------------------------------------------

static int run_shape_exec(
    PJRT_LoadedExecutable* exec,
    const double* src, int src_n,
    double* dst,       int dst_n)
{
    // --- Upload input ---
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
    if (err) return -1;
    PJRT_Buffer* in_buf = ba.buffer;

    // Wait for upload.
    {
        PJRT_Buffer_ReadyEvent_Args re{};
        re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        re.buffer = in_buf;
        err = g_api->PJRT_Buffer_ReadyEvent(&re);
        if (err) return -1;
        PJRT_Event_Await_Args ev{};
        ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ev.event = re.event;
        g_api->PJRT_Event_Await(&ev);
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = re.event;
        g_api->PJRT_Event_Destroy(&eda);
    }

    // --- Execute ---
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
    if (err) return -1;

    // --- Download output ---
    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_storage;
    tha.dst         = dst;
    tha.dst_size    = (size_t)dst_n * sizeof(double);

    err = g_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (err) return -1;

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

    // Cleanup.
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
// StableHLO module builders
// ---------------------------------------------------------------------------

static std::string f64t(int n) {
    return "tensor<" + std::to_string(n) + "xf64>";
}

// Transpose: use stablehlo.transpose with a permutation that swaps dim0/dim1.
static std::string build_transpose(const int* shape, int rank, int dim0, int dim1, int n) {
    // Build tensor type string e.g. tensor<2x3x4xf64>
    std::ostringstream type_ss;
    type_ss << "tensor<";
    for (int i = 0; i < rank; i++) {
        if (i) type_ss << "x";
        type_ss << shape[i];
    }
    type_ss << "xf64>";
    std::string inType = type_ss.str();

    // Build output tensor type string (shape with dim0/dim1 swapped).
    std::ostringstream otype_ss;
    otype_ss << "tensor<";
    for (int i = 0; i < rank; i++) {
        if (i) otype_ss << "x";
        otype_ss << (i == dim0 ? shape[dim1] : (i == dim1 ? shape[dim0] : shape[i]));
    }
    otype_ss << "xf64>";
    std::string outType = otype_ss.str();

    // Build permutation: identity with dim0 and dim1 swapped.
    std::ostringstream perm_ss;
    perm_ss << "[";
    for (int i = 0; i < rank; i++) {
        if (i) perm_ss << ", ";
        perm_ss << (i == dim0 ? dim1 : (i == dim1 ? dim0 : i));
    }
    perm_ss << "]";

    // The input is a flat 1-D tensor; we first reshape to N-D, then transpose.
    return
        "module @transpose {\n"
        "  func.func @main(%arg0: " + f64t(n) + ") -> " + f64t(n) + " {\n"
        "    %nd  = stablehlo.reshape %arg0 : (" + f64t(n) + ") -> " + inType + "\n"
        "    %tr  = stablehlo.transpose %nd, dims = " + perm_ss.str() + " : (" + inType + ") -> " + outType + "\n"
        "    %out = stablehlo.reshape %tr  : (" + outType + ") -> " + f64t(n) + "\n"
        "    return %out : " + f64t(n) + "\n"
        "  }\n"
        "}\n";
}

static std::string build_copy(int n) {
    std::string t = f64t(n);
    return
        "module @copy {\n"
        "  func.func @main(%arg0: " + t + ") -> " + t + " {\n"
        "    return %arg0 : " + t + "\n"
        "  }\n"
        "}\n";
}

// Concat along a flat virtual axis: simply concatenate two 1-D tensors.
static std::string build_concat(int n_a, int n_b) {
    std::string tA   = f64t(n_a);
    std::string tB   = f64t(n_b);
    std::string tOut = f64t(n_a + n_b);
    return
        "module @concat {\n"
        "  func.func @main(%a: " + tA + ", %b: " + tB + ") -> " + tOut + " {\n"
        "    %out = stablehlo.concatenate %a, %b, dim = 0 : (" + tA + ", " + tB + ") -> " + tOut + "\n"
        "    return %out : " + tOut + "\n"
        "  }\n"
        "}\n";
}

// ViewAsHeads: [B,T,H,head_dim] reshape+transpose -> [B,H,T,head_dim].
static std::string build_view_as_heads(int B, int T, int H, int hd) {
    int n = B * T * H * hd;
    // Input: flat n, reshape to [B,T,H,hd], transpose dims 1,2 -> [B,H,T,hd], reshape flat.
    std::ostringstream in_ss, out_ss;
    in_ss  << "tensor<" << B << "x" << T << "x" << H << "x" << hd << "xf64>";
    out_ss << "tensor<" << B << "x" << H << "x" << T << "x" << hd << "xf64>";
    std::string inT  = in_ss.str();
    std::string outT = out_ss.str();
    return
        "module @view_as_heads {\n"
        "  func.func @main(%arg0: " + f64t(n) + ") -> " + f64t(n) + " {\n"
        "    %nd  = stablehlo.reshape %arg0 : (" + f64t(n) + ") -> " + inT + "\n"
        "    %tr  = stablehlo.transpose %nd, dims = [0, 2, 1, 3] : (" + inT + ") -> " + outT + "\n"
        "    %out = stablehlo.reshape %tr  : (" + outT + ") -> " + f64t(n) + "\n"
        "    return %out : " + f64t(n) + "\n"
        "  }\n"
        "}\n";
}

// MergeHeads: [B,H,T,head_dim] transpose dims 1,2 -> [B,T,H,head_dim].
static std::string build_merge_heads(int B, int H, int T, int hd) {
    int n = B * H * T * hd;
    std::ostringstream in_ss, out_ss;
    in_ss  << "tensor<" << B << "x" << H << "x" << T << "x" << hd << "xf64>";
    out_ss << "tensor<" << B << "x" << T << "x" << H << "x" << hd << "xf64>";
    std::string inT  = in_ss.str();
    std::string outT = out_ss.str();
    return
        "module @merge_heads {\n"
        "  func.func @main(%arg0: " + f64t(n) + ") -> " + f64t(n) + " {\n"
        "    %nd  = stablehlo.reshape %arg0 : (" + f64t(n) + ") -> " + inT + "\n"
        "    %tr  = stablehlo.transpose %nd, dims = [0, 2, 1, 3] : (" + inT + ") -> " + outT + "\n"
        "    %out = stablehlo.reshape %tr  : (" + outT + ") -> " + f64t(n) + "\n"
        "    return %out : " + f64t(n) + "\n"
        "  }\n"
        "}\n";
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_transpose(const double* src, double* dst,
                  const int* shape, int rank,
                  int dim0, int dim1, int n)
{
    if (!g_client) return -1;
    std::string mlir = build_transpose(shape, rank, dim0, dim1, n);
    PJRT_LoadedExecutable* exec = compile_shape_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_shape_exec(exec, src, n, dst, n);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

int xla_copy(const double* src, double* dst, int n) {
    if (!g_client) return -1;
    std::string mlir = build_copy(n);
    PJRT_LoadedExecutable* exec = compile_shape_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_shape_exec(exec, src, n, dst, n);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

int xla_concat(const double* srcA, int n_a,
               const double* srcB, int n_b,
               double* dst)
{
    if (!g_client) return -1;
    // Pack both inputs into a single flat buffer and use a custom two-input module.
    // For simplicity: concatenate on host and use copy kernel.
    // A proper two-input PJRT dispatch requires a more elaborate execute path.
    // Here we concatenate on host, then run copy to at least exercise the pipeline.
    int total = n_a + n_b;
    double* combined = (double*)malloc((size_t)total * sizeof(double));
    if (!combined) return -1;
    memcpy(combined,        srcA, (size_t)n_a * sizeof(double));
    memcpy(combined + n_a, srcB, (size_t)n_b * sizeof(double));

    std::string mlir = build_copy(total);
    PJRT_LoadedExecutable* exec = compile_shape_stablehlo(mlir);
    if (!exec) { free(combined); return -1; }
    int rc = run_shape_exec(exec, combined, total, dst, total);
    free(combined);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

int xla_view_as_heads(const double* src, double* dst,
                      int B, int T, int H, int head_dim)
{
    if (!g_client) return -1;
    int n = B * T * H * head_dim;
    std::string mlir = build_view_as_heads(B, T, H, head_dim);
    PJRT_LoadedExecutable* exec = compile_shape_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_shape_exec(exec, src, n, dst, n);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

int xla_merge_heads(const double* src, double* dst,
                    int B, int H, int T, int head_dim)
{
    if (!g_client) return -1;
    int n = B * H * T * head_dim;
    std::string mlir = build_merge_heads(B, H, T, head_dim);
    PJRT_LoadedExecutable* exec = compile_shape_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_shape_exec(exec, src, n, dst, n);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

} // extern "C"
