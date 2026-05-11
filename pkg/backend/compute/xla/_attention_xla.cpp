// XLA attention backend — PJRT C API implementation.
//
// Implements SDPA, MQA, GQA, and SlidingWindow attention via StableHLO
// modules compiled through the PJRT C API.  Each variant is compiled once
// per (shape) configuration and cached for reuse.

#include "attention.h"
#include "activation.h"  // for g_api / g_client shared state

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <unordered_map>

#include "xla/pjrt/c/pjrt_c_api.h"

// ---------------------------------------------------------------------------
// Shared PJRT globals (defined in activation_xla.cc; declared extern here)
// ---------------------------------------------------------------------------

extern const PJRT_Api*  g_api;
extern PJRT_Client*     g_client;

// ---------------------------------------------------------------------------
// Attention executable cache keyed by a string encoding the call parameters.
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, PJRT_LoadedExecutable*> g_attn_execs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void attn_free_error(PJRT_Error* err) {
    if (!err || !g_api) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    g_api->PJRT_Error_Destroy(&da);
}

static bool attn_check(PJRT_Error* err, const char* ctx) {
    if (!err) return true;
    if (g_api) {
        PJRT_Error_Message_Args ma{};
        ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        ma.error = err;
        g_api->PJRT_Error_Message(&ma);
        fprintf(stderr, "XLA attention PJRT error in %s: %.*s\n",
                ctx, (int)ma.message_size, ma.message);
    }
    attn_free_error(err);
    return false;
}

// Compile a StableHLO module string; returns nullptr on failure.
static PJRT_LoadedExecutable* attn_compile(const std::string& mlir) {
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
    if (!attn_check(err, "PJRT_Client_Compile(attention)")) return nullptr;
    return ca.executable;
}

// Execute a compiled executable with two input buffers (src_n elements each)
// and one output buffer (dst_n elements).
static int attn_run2(
    PJRT_LoadedExecutable* exec,
    const double* src0, int src0_n,
    const double* src1, int src1_n,
    const double* src2, int src2_n,
    double* dst, int dst_n)
{
    auto make_buf = [&](const double* data, int n) -> PJRT_Buffer* {
        PJRT_Client_BufferFromHostBuffer_Args ba{};
        ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        ba.client      = g_client;
        ba.data        = data;
        ba.type        = PJRT_Buffer_Type_F64;
        int64_t dims[1] = { (int64_t)n };
        ba.dims        = dims;
        ba.num_dims    = 1;
        ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        PJRT_Error* err = g_api->PJRT_Client_BufferFromHostBuffer(&ba);
        if (!attn_check(err, "BufferFromHostBuffer(attn)")) return nullptr;
        // Wait for transfer.
        PJRT_Buffer_ReadyEvent_Args re{};
        re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
        re.buffer = ba.buffer;
        err = g_api->PJRT_Buffer_ReadyEvent(&re);
        if (!attn_check(err, "ReadyEvent(attn)")) return nullptr;
        PJRT_Event_Await_Args ea{};
        ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ea.event = re.event;
        err = g_api->PJRT_Event_Await(&ea);
        attn_check(err, "Event_Await(attn in)");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = re.event;
        g_api->PJRT_Event_Destroy(&eda);
        return ba.buffer;
    };

    auto destroy_buf = [&](PJRT_Buffer* b) {
        if (!b) return;
        PJRT_Buffer_Destroy_Args da{};
        da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        da.buffer = b;
        g_api->PJRT_Buffer_Destroy(&da);
    };

    PJRT_Buffer* in0 = make_buf(src0, src0_n);
    PJRT_Buffer* in1 = make_buf(src1, src1_n);
    PJRT_Buffer* in2 = make_buf(src2, src2_n);
    if (!in0 || !in1 || !in2) {
        destroy_buf(in0); destroy_buf(in1); destroy_buf(in2);
        return -1;
    }

    PJRT_Buffer* in_arr[3]  = { in0, in1, in2 };
    PJRT_Buffer* out_storage = nullptr;
    PJRT_Buffer** out_arr[1] = { &out_storage };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size     = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable      = exec;
    ea.argument_lists  = (PJRT_Buffer***)&in_arr;
    ea.num_devices     = 1;
    ea.num_args        = 3;
    ea.output_lists    = out_arr;
    PJRT_ExecuteOptions options = single_device_execute_options();
    ea.options = &options;

    PJRT_Error* err = g_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!attn_check(err, "Execute(attn)")) {
        destroy_buf(in0); destroy_buf(in1); destroy_buf(in2);
        return -1;
    }

    // Copy output device→host.
    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_storage;
    tha.dst         = dst;
    tha.dst_size    = (size_t)dst_n * sizeof(double);
    err = g_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (!attn_check(err, "ToHostBuffer(attn)")) {
        destroy_buf(in0); destroy_buf(in1); destroy_buf(in2);
        destroy_buf(out_storage);
        return -1;
    }
    {
        PJRT_Event_Await_Args ev{};
        ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        ev.event = tha.event;
        err = g_api->PJRT_Event_Await(&ev);
        attn_check(err, "Event_Await(attn out)");
        PJRT_Event_Destroy_Args eda{};
        eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eda.event = tha.event;
        g_api->PJRT_Event_Destroy(&eda);
    }

    destroy_buf(in0); destroy_buf(in1); destroy_buf(in2);
    destroy_buf(out_storage);
    return 0;
}

// ---------------------------------------------------------------------------
// StableHLO module builders
// Each module receives three 1-D f64 tensors (flat Q, K, V) and returns one
// 1-D f64 tensor (flat output).  The attention math is expressed via
// stablehlo.dot_general + stablehlo.exponential + stablehlo.reduce +
// stablehlo.divide.
//
// For correctness with arbitrary batch/head/seq/dim sizes we implement a
// loop-free version using reshape + dot_general.  The full softmax over
// seq_len is done element-wise after the score matmul.
// ---------------------------------------------------------------------------

static std::string si(int v) { return std::to_string(v); }

// Build 4-D tensor type string: tensor<bxhxsxd x f64>
static std::string t4(int b, int h, int s, int d) {
    return "tensor<" + si(b) + "x" + si(h) + "x" + si(s) + "x" + si(d) + "xf64>";
}
static std::string t4sq(int b, int h, int s) {
    return "tensor<" + si(b) + "x" + si(h) + "x" + si(s) + "x" + si(s) + "xf64>";
}
static std::string t1(int n) {
    return "tensor<" + si(n) + "xf64>";
}

// Emit a StableHLO module that:
//   1. Reshapes flat Q/K/V → 4-D
//   2. Computes scores = Q @ K^T / sqrt(head_dim)  via dot_general
//   3. Softmax over last dim
//   4. Output = softmax(scores) @ V
//   5. Reshapes back to flat
static std::string build_sdpa_module(
    const std::string& name,
    int batch, int num_heads, int seq_len, int head_dim,
    int kv_heads) // kv_heads == num_heads for SDPA/SW, 1 for MQA, num_kv_heads for GQA
{
    int q_n  = batch * num_heads * seq_len * head_dim;
    int kv_n = batch * kv_heads  * seq_len * head_dim;
    int sc_n = batch * num_heads * seq_len * seq_len;

    std::string tQ  = t4(batch, num_heads, seq_len, head_dim);
    std::string tKV = t4(batch, kv_heads,  seq_len, head_dim);
    std::string tSc = t4sq(batch, num_heads, seq_len);
    // For KV broadcast in MQA/GQA we broadcast kv_heads → num_heads via the
    // dot_general batching dimensions.

    char scale_buf[64];
    snprintf(scale_buf, sizeof(scale_buf), "%.17g", 1.0 / sqrt((double)head_dim));
    std::string scale_str = scale_buf;

    // We use a scalar constant for the scale then broadcast.
    // Softmax: reduce_max then exp(x - max) / sum.

    // Dimension annotations for dot_general:
    //   Q  axes: [0=batch, 1=head, 2=seq_q, 3=dim]
    //   K  axes: [0=batch, 1=head, 2=seq_k, 3=dim]
    //   contracting: dim (axis 3 in Q, axis 3 in K)
    //   batching:    batch (axis 0), head (axis 1)
    //   result: [batch, head, seq_q, seq_k]
    //
    // For MQA/GQA we broadcast K/V heads by using a reshape before the matmul.

    std::string tQflat  = t1(q_n);
    std::string tKVflat = t1(kv_n);
    std::string tOutflat= t1(q_n);
    std::string tScFlat = t1(sc_n);

    // We'll build a module that broadcasts KV heads to num_heads (no-op if
    // kv_heads==num_heads).
    std::string tKV_bcast = t4(batch, num_heads, seq_len, head_dim);
    int kv_bcast_n = batch * num_heads * seq_len * head_dim;

    std::string module =
        "module @" + name + " {\n"
        "  func.func @main(\n"
        "      %q_flat:  " + tQflat  + ",\n"
        "      %k_flat:  " + tKVflat + ",\n"
        "      %v_flat:  " + tKVflat + "\n"
        "  ) -> " + tOutflat + " {\n"
        // Reshape Q
        "    %q = stablehlo.reshape %q_flat : (" + tQflat + ") -> " + tQ + "\n";

    if (kv_heads == num_heads) {
        module +=
        "    %k = stablehlo.reshape %k_flat : (" + tKVflat + ") -> " + tKV + "\n"
        "    %v = stablehlo.reshape %v_flat : (" + tKVflat + ") -> " + tKV + "\n";
    } else {
        // Broadcast kv_heads → num_heads via broadcast_in_dim
        // First reshape to [batch, kv_heads, seq_len, head_dim], then broadcast.
        // broadcast_in_dim maps dims [0,1,2,3] of kv tensor to dims [0,1,2,3] of output
        // with the num_heads dim broadcasting.  We use a simpler approach:
        // reshape to [batch, kv_heads, 1, seq_len, head_dim] then broadcast — but
        // StableHLO broadcast_in_dim does not have a repeat axis in the same way.
        // Instead we use stablehlo.broadcast_in_dim with output shape
        // [batch, num_heads, seq_len, head_dim] mapping:
        //   kv dim 0 (batch)    → out dim 0
        //   kv dim 1 (kv_heads) → out dim 1  (requires divisibility; replicated by XLA)
        // This only works if kv_heads divides num_heads evenly.
        // We fall back to reshape+broadcast via iota+gather which is complex;
        // instead emit a simple element-wise expansion loop via slices.
        //
        // Practical approach: since StableHLO doesn't have a native "repeat"
        // primitive, we express the broadcast as a reshape to
        // [batch, kv_heads, 1, seq_len, head_dim] then
        // broadcast_in_dim to [batch, kv_heads, group_size, seq_len, head_dim]
        // then reshape to [batch, num_heads, seq_len, head_dim].
        int group = num_heads / kv_heads;
        std::string tKV_5d  = "tensor<" + si(batch) + "x" + si(kv_heads) + "x1x" + si(seq_len) + "x" + si(head_dim) + "xf64>";
        std::string tKV_5db = "tensor<" + si(batch) + "x" + si(kv_heads) + "x" + si(group) + "x" + si(seq_len) + "x" + si(head_dim) + "xf64>";

        module +=
        "    %k_r  = stablehlo.reshape %k_flat : (" + tKVflat + ") -> " + tKV_5d + "\n"
        "    %k_b  = stablehlo.broadcast_in_dim %k_r, dims = [0, 1, 2, 3, 4] : (" + tKV_5d + ") -> " + tKV_5db + "\n"
        "    %k = stablehlo.reshape %k_b : (" + tKV_5db + ") -> " + tKV_bcast + "\n"
        "    %v_r  = stablehlo.reshape %v_flat : (" + tKVflat + ") -> " + tKV_5d + "\n"
        "    %v_b  = stablehlo.broadcast_in_dim %v_r, dims = [0, 1, 2, 3, 4] : (" + tKV_5d + ") -> " + tKV_5db + "\n"
        "    %v = stablehlo.reshape %v_b : (" + tKV_5db + ") -> " + tKV_bcast + "\n";
    }

    std::string tK_used = (kv_heads == num_heads) ? tKV : tKV_bcast;

    // scores = dot_general(Q, K, contracting=[3],[3], batching=[0,1],[0,1])
    // result type: [batch, num_heads, seq_len, seq_len]
    module +=
        "    %scores_raw = stablehlo.dot_general %q, " + std::string(kv_heads==num_heads?"%k":"%k") + ",\n"
        "        batching_dims = [0, 1] x [0, 1],\n"
        "        contracting_dims = [3] x [3] : (" + tQ + ", " + tK_used + ") -> " + tSc + "\n"
        // Scale scores
        "    %scale_c = stablehlo.constant dense<" + scale_str + "> : tensor<f64>\n"
        "    %scale   = stablehlo.broadcast_in_dim %scale_c, dims = [] : (tensor<f64>) -> " + tSc + "\n"
        "    %scores  = stablehlo.multiply %scores_raw, %scale : " + tSc + "\n"
        // Softmax: reduce max over last dim
        "    %neg_inf  = stablehlo.constant dense<-1.79769e+308> : tensor<f64>\n"
        "    %sc_max_r = stablehlo.reduce(%scores init: %neg_inf) applies stablehlo.maximum across dimensions = [3] : (" + tSc + ", tensor<f64>) -> tensor<" + si(batch) + "x" + si(num_heads) + "x" + si(seq_len) + "xf64>\n"
        "    %sc_max   = stablehlo.broadcast_in_dim %sc_max_r, dims = [0, 1, 2] : (tensor<" + si(batch) + "x" + si(num_heads) + "x" + si(seq_len) + "xf64>) -> " + tSc + "\n"
        "    %sc_shift = stablehlo.subtract %scores, %sc_max : " + tSc + "\n"
        "    %sc_exp   = stablehlo.exponential %sc_shift : " + tSc + "\n"
        "    %zero_f   = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sc_sum_r = stablehlo.reduce(%sc_exp init: %zero_f) applies stablehlo.add across dimensions = [3] : (" + tSc + ", tensor<f64>) -> tensor<" + si(batch) + "x" + si(num_heads) + "x" + si(seq_len) + "xf64>\n"
        "    %sc_sum   = stablehlo.broadcast_in_dim %sc_sum_r, dims = [0, 1, 2] : (tensor<" + si(batch) + "x" + si(num_heads) + "x" + si(seq_len) + "xf64>) -> " + tSc + "\n"
        "    %weights  = stablehlo.divide %sc_exp, %sc_sum : " + tSc + "\n"
        // Output = weights @ V  — dot_general contracting over seq_k (dim 3 of weights, dim 2 of V)
        "    %out4d = stablehlo.dot_general %weights, " + std::string(kv_heads==num_heads?"%v":"%v") + ",\n"
        "        batching_dims = [0, 1] x [0, 1],\n"
        "        contracting_dims = [3] x [2] : (" + tSc + ", " + tK_used + ") -> " + tQ + "\n"
        "    %out_flat = stablehlo.reshape %out4d : (" + tQ + ") -> " + tOutflat + "\n"
        "    return %out_flat : " + tOutflat + "\n"
        "  }\n"
        "}\n";

    return module;
}

// SlidingWindow: we add a mask tensor of -inf before softmax.
// We compute the mask as a constant iota-based expression.
// Simplest approach: emit the full module using same logic as SDPA but add
// an additive mask.  We construct the mask via stablehlo.iota + compare.
static std::string build_sliding_window_module(
    int batch, int num_heads, int seq_len, int head_dim, int window)
{
    int q_n  = batch * num_heads * seq_len * head_dim;
    int sc_n = batch * num_heads * seq_len * seq_len;

    std::string tQ  = t4(batch, num_heads, seq_len, head_dim);
    std::string tSc = t4sq(batch, num_heads, seq_len);
    std::string tQflat   = t1(q_n);
    std::string tOutflat = t1(q_n);

    char scale_buf[64];
    snprintf(scale_buf, sizeof(scale_buf), "%.17g", 1.0 / sqrt((double)head_dim));

    // We build the mask as a constant dense tensor of shape [seq_len, seq_len]
    // then broadcast it.  For large seq_len this could be large, but it's
    // correct.  mask[i][j] = -inf if j < i-window || j > i, else 0.
    std::string mask_vals = "";
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            bool masked = (j < i - window || j > i);
            if (j > 0 || i > 0) mask_vals += ", ";
            mask_vals += masked ? "-1.79769e+308" : "0.0";
        }
    }
    std::string tMask2d = "tensor<" + si(seq_len) + "x" + si(seq_len) + "xf64>";

    return
        "module @sliding_window {\n"
        "  func.func @main(\n"
        "      %q_flat: " + tQflat + ",\n"
        "      %k_flat: " + tQflat + ",\n"
        "      %v_flat: " + tQflat + "\n"
        "  ) -> " + tOutflat + " {\n"
        "    %q = stablehlo.reshape %q_flat : (" + tQflat + ") -> " + tQ + "\n"
        "    %k = stablehlo.reshape %k_flat : (" + tQflat + ") -> " + tQ + "\n"
        "    %v = stablehlo.reshape %v_flat : (" + tQflat + ") -> " + tQ + "\n"
        "    %scores_raw = stablehlo.dot_general %q, %k,\n"
        "        batching_dims = [0, 1] x [0, 1],\n"
        "        contracting_dims = [3] x [3] : (" + tQ + ", " + tQ + ") -> " + tSc + "\n"
        "    %scale_c = stablehlo.constant dense<" + std::string(scale_buf) + "> : tensor<f64>\n"
        "    %scale   = stablehlo.broadcast_in_dim %scale_c, dims = [] : (tensor<f64>) -> " + tSc + "\n"
        "    %scores_s = stablehlo.multiply %scores_raw, %scale : " + tSc + "\n"
        "    %mask2d  = stablehlo.constant dense<[" + mask_vals + "]> : " + tMask2d + "\n"
        "    %mask4d  = stablehlo.broadcast_in_dim %mask2d, dims = [2, 3] : (" + tMask2d + ") -> " + tSc + "\n"
        "    %scores  = stablehlo.add %scores_s, %mask4d : " + tSc + "\n"
        "    %neg_inf  = stablehlo.constant dense<-1.79769e+308> : tensor<f64>\n"
        "    %sc_max_r = stablehlo.reduce(%scores init: %neg_inf) applies stablehlo.maximum across dimensions = [3] : (" + tSc + ", tensor<f64>) -> tensor<" + si(batch) + "x" + si(num_heads) + "x" + si(seq_len) + "xf64>\n"
        "    %sc_max   = stablehlo.broadcast_in_dim %sc_max_r, dims = [0, 1, 2] : (tensor<" + si(batch) + "x" + si(num_heads) + "x" + si(seq_len) + "xf64>) -> " + tSc + "\n"
        "    %sc_shift = stablehlo.subtract %scores, %sc_max : " + tSc + "\n"
        "    %sc_exp   = stablehlo.exponential %sc_shift : " + tSc + "\n"
        "    %zero_f   = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sc_sum_r = stablehlo.reduce(%sc_exp init: %zero_f) applies stablehlo.add across dimensions = [3] : (" + tSc + ", tensor<f64>) -> tensor<" + si(batch) + "x" + si(num_heads) + "x" + si(seq_len) + "xf64>\n"
        "    %sc_sum   = stablehlo.broadcast_in_dim %sc_sum_r, dims = [0, 1, 2] : (tensor<" + si(batch) + "x" + si(num_heads) + "x" + si(seq_len) + "xf64>) -> " + tSc + "\n"
        "    %weights  = stablehlo.divide %sc_exp, %sc_sum : " + tSc + "\n"
        "    %out4d    = stablehlo.dot_general %weights, %v,\n"
        "        batching_dims = [0, 1] x [0, 1],\n"
        "        contracting_dims = [3] x [2] : (" + tSc + ", " + tQ + ") -> " + tQ + "\n"
        "    %out_flat = stablehlo.reshape %out4d : (" + tQ + ") -> " + tOutflat + "\n"
        "    return %out_flat : " + tOutflat + "\n"
        "  }\n"
        "}\n";
}

// Cache key helpers
static std::string sdpa_key(int b, int h, int s, int d) {
    return "sdpa:" + si(b) + ":" + si(h) + ":" + si(s) + ":" + si(d);
}
static std::string mqa_key(int b, int h, int s, int d) {
    return "mqa:" + si(b) + ":" + si(h) + ":" + si(s) + ":" + si(d);
}
static std::string gqa_key(int b, int h, int kv, int s, int d) {
    return "gqa:" + si(b) + ":" + si(h) + ":" + si(kv) + ":" + si(s) + ":" + si(d);
}
static std::string sw_key(int b, int h, int s, int d, int w) {
    return "sw:" + si(b) + ":" + si(h) + ":" + si(s) + ":" + si(d) + ":" + si(w);
}

static PJRT_LoadedExecutable* get_or_compile(const std::string& key, const std::string& mlir) {
    auto it = g_attn_execs.find(key);
    if (it != g_attn_execs.end()) return it->second;
    PJRT_LoadedExecutable* exec = attn_compile(mlir);
    if (exec) g_attn_execs[key] = exec;
    return exec;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_sdpa(const double* q, const double* k, const double* v, double* out,
             int batch, int num_heads, int seq_len, int head_dim)
{
    if (!g_client) return -1;
    std::string key = sdpa_key(batch, num_heads, seq_len, head_dim);
    PJRT_LoadedExecutable* exec = get_or_compile(
        key, build_sdpa_module("sdpa", batch, num_heads, seq_len, head_dim, num_heads));
    if (!exec) return -1;
    int n = batch * num_heads * seq_len * head_dim;
    return attn_run2(exec, q, n, k, n, v, n, out, n);
}

int xla_mqa(const double* q, const double* k, const double* v, double* out,
            int batch, int num_heads, int seq_len, int head_dim)
{
    if (!g_client) return -1;
    std::string key = mqa_key(batch, num_heads, seq_len, head_dim);
    PJRT_LoadedExecutable* exec = get_or_compile(
        key, build_sdpa_module("mqa", batch, num_heads, seq_len, head_dim, 1));
    if (!exec) return -1;
    int qn  = batch * num_heads * seq_len * head_dim;
    int kvn = batch * 1          * seq_len * head_dim;
    return attn_run2(exec, q, qn, k, kvn, v, kvn, out, qn);
}

int xla_gqa(const double* q, const double* k, const double* v, double* out,
            int batch, int num_heads, int num_kv_heads, int seq_len, int head_dim)
{
    if (!g_client) return -1;
    std::string key = gqa_key(batch, num_heads, num_kv_heads, seq_len, head_dim);
    PJRT_LoadedExecutable* exec = get_or_compile(
        key, build_sdpa_module("gqa", batch, num_heads, seq_len, head_dim, num_kv_heads));
    if (!exec) return -1;
    int qn  = batch * num_heads    * seq_len * head_dim;
    int kvn = batch * num_kv_heads * seq_len * head_dim;
    return attn_run2(exec, q, qn, k, kvn, v, kvn, out, qn);
}

int xla_sliding_window(const double* q, const double* k, const double* v, double* out,
                       int batch, int num_heads, int seq_len, int head_dim, int window)
{
    if (!g_client) return -1;
    std::string key = sw_key(batch, num_heads, seq_len, head_dim, window);
    PJRT_LoadedExecutable* exec = get_or_compile(
        key, build_sliding_window_module(batch, num_heads, seq_len, head_dim, window));
    if (!exec) return -1;
    int n = batch * num_heads * seq_len * head_dim;
    return attn_run2(exec, q, n, k, n, v, n, out, n);
}

} // extern "C"
