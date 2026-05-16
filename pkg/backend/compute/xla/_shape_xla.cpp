// XLA shape backend — PJRT C API implementation.
//
// Build requirements: same as activation_xla.cc.
// These functions reuse the global g_api / g_client established by xla_init().

#include "shape.h"
#include "activation.h"  // for PJRT types and xla_init / run_executable helper

#include <cstdio>
#include <cstdint>
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

static PJRT_Buffer* shape_host_to_device(const double* src, int src_n) {
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
    if (err) return nullptr;

    PJRT_Buffer_ReadyEvent_Args re{};
    re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    re.buffer = ba.buffer;
    err = g_api->PJRT_Buffer_ReadyEvent(&re);
    if (err) {
        PJRT_Buffer_Destroy_Args da{};
        da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        da.buffer = ba.buffer;
        g_api->PJRT_Buffer_Destroy(&da);
        return nullptr;
    }

    PJRT_Event_Await_Args ev{};
    ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ev.event = re.event;
    g_api->PJRT_Event_Await(&ev);

    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event = re.event;
    g_api->PJRT_Event_Destroy(&eda);

    return ba.buffer;
}

static void shape_destroy_buf(PJRT_Buffer* buffer) {
    if (!buffer) return;

    PJRT_Buffer_Destroy_Args da{};
    da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    da.buffer = buffer;
    g_api->PJRT_Buffer_Destroy(&da);
}

static int run_shape_exec2(
    PJRT_LoadedExecutable* exec,
    const double* src_a, int src_a_n,
    const double* src_b, int src_b_n,
    double* dst, int dst_n)
{
    PJRT_Buffer* buffer_a = shape_host_to_device(src_a, src_a_n);
    if (!buffer_a) return -1;

    PJRT_Buffer* buffer_b = shape_host_to_device(src_b, src_b_n);
    if (!buffer_b) {
        shape_destroy_buf(buffer_a);
        return -1;
    }

    PJRT_Buffer* args[2] = { buffer_a, buffer_b };
    PJRT_Buffer* out_storage = nullptr;
    PJRT_Buffer** out_list[1] = { &out_storage };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size     = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable      = exec;
    PJRT_ExecuteOptions options = single_device_execute_options();
    ea.options         = &options;
    ea.argument_lists  = (PJRT_Buffer***)&args;
    ea.num_devices     = 1;
    ea.num_args        = 2;
    ea.output_lists    = out_list;

    PJRT_Error* err = g_api->PJRT_LoadedExecutable_Execute(&ea);
    shape_destroy_buf(buffer_a);
    shape_destroy_buf(buffer_b);
    if (err) return -1;

    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_storage;
    tha.dst         = dst;
    tha.dst_size    = (size_t)dst_n * sizeof(double);

    err = g_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (err) {
        shape_destroy_buf(out_storage);
        return -1;
    }

    PJRT_Event_Await_Args ev{};
    ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ev.event = tha.event;
    g_api->PJRT_Event_Await(&ev);

    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event = tha.event;
    g_api->PJRT_Event_Destroy(&eda);

    shape_destroy_buf(out_storage);
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

static std::string build_split(int outer, int dim_size, int split_size, int inner) {
    int64_t total64 = (int64_t)outer * (int64_t)dim_size * (int64_t)inner;
    int64_t num_chunks64 = (int64_t)dim_size / (int64_t)split_size;
    int64_t element_count64 = (int64_t)split_size * (int64_t)inner;

    if (total64 <= 0 || total64 > INT32_MAX || element_count64 <= 0 || element_count64 > INT32_MAX) {
        return "";
    }

    if (num_chunks64 <= 0 || num_chunks64 > INT32_MAX) {
        return "";
    }

    int total = (int)total64;
    int num_chunks = (int)num_chunks64;
    int element_count = (int)element_count64;
    std::ostringstream body;
    int value_index = 0;

    body
        << "module @split {\n"
        << "  func.func @main(%arg0: " << f64t(total) << ") -> " << f64t(total) << " {\n";

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        for (int outer_index = 0; outer_index < outer; outer_index++) {
            int src_offset = (outer_index * dim_size + chunk * split_size) * inner;
            body
                << "    %v" << value_index
                << " = stablehlo.slice %arg0 [" << src_offset << ":"
                << (src_offset + element_count) << "] : (" << f64t(total)
                << ") -> " << f64t(element_count) << "\n";
            value_index++;
        }
    }

    if (value_index == 1) {
        body << "    return %v0 : " << f64t(total) << "\n";
    } else {
        int current_count = element_count;
        body
            << "    %cat0 = stablehlo.concatenate %v0, %v1, dim = 0 : ("
            << f64t(element_count) << ", " << f64t(element_count)
            << ") -> " << f64t(current_count + element_count) << "\n";
        current_count += element_count;

        for (int index = 2; index < value_index; index++) {
            body
                << "    %cat" << (index - 1)
                << " = stablehlo.concatenate %cat" << (index - 2)
                << ", %v" << index << ", dim = 0 : ("
                << f64t(current_count) << ", " << f64t(element_count)
                << ") -> " << f64t(current_count + element_count) << "\n";
            current_count += element_count;
        }

        body << "    return %cat" << (value_index - 2) << " : " << f64t(total) << "\n";
    }

    body << "  }\n}\n";
    return body.str();
}

static std::string type4(int B, int C, int H, int W) {
    return "tensor<" + std::to_string(B) + "x" + std::to_string(C) + "x" +
           std::to_string(H) + "x" + std::to_string(W) + "xf64>";
}

static std::string type5(int B, int C, int H, int S, int W) {
    return "tensor<" + std::to_string(B) + "x" + std::to_string(C) + "x" +
           std::to_string(H) + "x" + std::to_string(S) + "x" +
           std::to_string(W) + "xf64>";
}

static std::string type6(int B, int C, int H, int SH, int W, int SW) {
    return "tensor<" + std::to_string(B) + "x" + std::to_string(C) + "x" +
           std::to_string(H) + "x" + std::to_string(SH) + "x" +
           std::to_string(W) + "x" + std::to_string(SW) + "xf64>";
}

static std::string build_upsample_nearest2d(
    int B, int C, int H, int W, int scale_h, int scale_w)
{
    int64_t input64 = (int64_t)B * (int64_t)C * (int64_t)H * (int64_t)W;
    int64_t output64 = input64 * (int64_t)scale_h * (int64_t)scale_w;

    if (input64 <= 0 || input64 > INT32_MAX || output64 <= 0 || output64 > INT32_MAX) {
        return "";
    }

    int input_count = (int)input64;
    int output_count = (int)output64;
    std::string tIn = type4(B, C, H, W);
    std::string tH = type5(B, C, H, scale_h, W);
    std::string tHW = type6(B, C, H, scale_h, W, scale_w);
    std::string tOut = type4(B, C, H * scale_h, W * scale_w);

    return
        "module @upsample_nearest2d {\n"
        "  func.func @main(%arg0: " + f64t(input_count) + ") -> " + f64t(output_count) + " {\n"
        "    %nd = stablehlo.reshape %arg0 : (" + f64t(input_count) + ") -> " + tIn + "\n"
        "    %h = stablehlo.broadcast_in_dim %nd, dims = [0, 1, 2, 4] : (" + tIn + ") -> " + tH + "\n"
        "    %hw = stablehlo.broadcast_in_dim %h, dims = [0, 1, 2, 3, 4] : (" + tH + ") -> " + tHW + "\n"
        "    %out4 = stablehlo.reshape %hw : (" + tHW + ") -> " + tOut + "\n"
        "    %out = stablehlo.reshape %out4 : (" + tOut + ") -> " + f64t(output_count) + "\n"
        "    return %out : " + f64t(output_count) + "\n"
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

static std::string build_last_token(int outer, int seq_len, int feature) {
    int64_t input64 = (int64_t)outer * (int64_t)seq_len * (int64_t)feature;
    int64_t output64 = (int64_t)outer * (int64_t)feature;

    if (input64 <= 0 || input64 > INT32_MAX || output64 <= 0 || output64 > INT32_MAX) {
        return "";
    }

    int input_count = (int)input64;
    int output_count = (int)output64;
    std::ostringstream body;

    body
        << "module @last_token {\n"
        << "  func.func @main(%arg0: " << f64t(input_count) << ") -> "
        << f64t(output_count) << " {\n";

    for (int outer_index = 0; outer_index < outer; outer_index++) {
        int source_offset = (outer_index * seq_len + (seq_len - 1)) * feature;
        body
            << "    %v" << outer_index
            << " = stablehlo.slice %arg0 [" << source_offset << ":"
            << (source_offset + feature) << "] : (" << f64t(input_count)
            << ") -> " << f64t(feature) << "\n";
    }

    if (outer == 1) {
        body << "    return %v0 : " << f64t(output_count) << "\n";
    } else {
        int current_count = feature;

        body
            << "    %cat0 = stablehlo.concatenate %v0, %v1, dim = 0 : ("
            << f64t(feature) << ", " << f64t(feature)
            << ") -> " << f64t(current_count + feature) << "\n";
        current_count += feature;

        for (int index = 2; index < outer; index++) {
            body
                << "    %cat" << (index - 1)
                << " = stablehlo.concatenate %cat" << (index - 2)
                << ", %v" << index << ", dim = 0 : ("
                << f64t(current_count) << ", " << f64t(feature)
                << ") -> " << f64t(current_count + feature) << "\n";
            current_count += feature;
        }

        body << "    return %cat" << (outer - 2) << " : " << f64t(output_count) << "\n";
    }

    body << "  }\n}\n";
    return body.str();
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
    int total = n_a + n_b;
    std::string mlir = build_concat(n_a, n_b);
    PJRT_LoadedExecutable* exec = compile_shape_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_shape_exec2(exec, srcA, n_a, srcB, n_b, dst, total);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

int xla_split(const double* src, double* dst,
              int outer, int dim_size, int split_size, int inner, int total_count)
{
    if (!g_client) return -1;
    if (!src || !dst) return -1;
    if (outer <= 0 || dim_size <= 0 || split_size <= 0 || inner <= 0) return -1;
    if (dim_size % split_size != 0) return -1;

    int64_t total64 = (int64_t)outer * (int64_t)dim_size * (int64_t)inner;

    if (total64 <= 0 || total64 > INT32_MAX || total64 != (int64_t)total_count) return -1;

    int total = (int)total64;
    std::string mlir = build_split(outer, dim_size, split_size, inner);
    if (mlir.empty()) return -1;
    PJRT_LoadedExecutable* exec = compile_shape_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_shape_exec(exec, src, total, dst, total);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

int xla_upsample_nearest2d(const double* src, double* dst,
                           int B, int C, int H, int W,
                           int scale_h, int scale_w)
{
    if (!g_client) return -1;
    if (!src || !dst || B <= 0 || C <= 0 || H <= 0 || W <= 0 ||
        scale_h <= 0 || scale_w <= 0) {
        return -1;
    }

    int64_t input64 = (int64_t)B * (int64_t)C * (int64_t)H * (int64_t)W;
    int64_t output64 = input64 * (int64_t)scale_h * (int64_t)scale_w;

    if (input64 <= 0 || input64 > INT32_MAX || output64 <= 0 || output64 > INT32_MAX) {
        return -1;
    }

    std::string mlir = build_upsample_nearest2d(B, C, H, W, scale_h, scale_w);
    if (mlir.empty()) return -1;
    PJRT_LoadedExecutable* exec = compile_shape_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_shape_exec(exec, src, (int)input64, dst, (int)output64);
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

int xla_last_token(const double* src, double* dst,
                   int outer, int seq_len, int feature)
{
    if (!g_client) return -1;
    if (!src || !dst || outer <= 0 || seq_len <= 0 || feature <= 0) return -1;

    int64_t input64 = (int64_t)outer * (int64_t)seq_len * (int64_t)feature;
    int64_t output64 = (int64_t)outer * (int64_t)feature;

    if (input64 <= 0 || input64 > INT32_MAX || output64 <= 0 || output64 > INT32_MAX) {
        return -1;
    }

    std::string mlir = build_last_token(outer, seq_len, feature);
    if (mlir.empty()) return -1;
    PJRT_LoadedExecutable* exec = compile_shape_stablehlo(mlir);
    if (!exec) return -1;
    int rc = run_shape_exec(exec, src, (int)input64, dst, (int)output64);
    PJRT_LoadedExecutable_Destroy_Args da{};
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable  = exec;
    g_api->PJRT_LoadedExecutable_Destroy(&da);
    return rc;
}

} // extern "C"
