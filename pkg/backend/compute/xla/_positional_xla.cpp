// XLA positional backend — PJRT C API implementation.
//
// RoPE: expressed as StableHLO multiply+subtract/add over the input
//       with externally-supplied cos/sin constant tensors.
//
// ALiBi: expressed as StableHLO outer product of slopes and iota positions.

#include "positional.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

#include "xla/pjrt/c/pjrt_c_api.h"
#include <dlfcn.h>

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static const PJRT_Api*          gp_api    = nullptr;
static PJRT_Client*             gp_client = nullptr;
static PJRT_LoadedExecutable*   gp_rope_exec  = nullptr;
static PJRT_LoadedExecutable*   gp_alibi_exec = nullptr;

// Cached dimensions for compiled executables
static int gp_rope_total_heads = 0;
static int gp_rope_seq_len     = 0;
static int gp_rope_head_dim    = 0;
static int gp_alibi_num_heads  = 0;
static int gp_alibi_seq_len_q  = 0;
static int gp_alibi_seq_len_k  = 0;

// ---------------------------------------------------------------------------
// Error helpers (shared pattern from activation_xla.cc)
// ---------------------------------------------------------------------------

static void pos_free_error(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
}

static bool pos_check(const PJRT_Api* api, PJRT_Error* err, const char* ctx) {
    if (!err) return true;
    PJRT_Error_Message_Args ma{};
    ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    ma.error = err;
    api->PJRT_Error_Message(&ma);
    fprintf(stderr, "XLA PJRT error in %s: %.*s\n",
            ctx, (int)ma.message_size, ma.message);
    pos_free_error(api, err);
    return false;
}

// ---------------------------------------------------------------------------
// Plugin loader
// ---------------------------------------------------------------------------

typedef const PJRT_Api* (*GetPjrtApiFn)();

static const PJRT_Api* pos_load_pjrt_plugin(const char* platform) {
    std::string path = pjrt_plugin_path(platform);
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "XLA positional: failed to dlopen %s: %s\n", path.c_str(), dlerror());
        return nullptr;
    }
    auto fn = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!fn) return nullptr;
    return fn();
}

// ---------------------------------------------------------------------------
// StableHLO module builders
// ---------------------------------------------------------------------------

// RoPE module:
//   inputs: x[N, head_dim], cos[P], sin[P]  where N = total_heads*seq_len, P = num_pairs
//   Reshape x -> [N, P, 2]
//   even = x[:,:,0], odd = x[:,:,1]
//   out_even = even*cos - odd*sin
//   out_odd  = even*sin + odd*cos
//   Reshape back to [N, head_dim]
//
// We express this as a flat element-wise computation matching the CUDA kernel:
//   The module takes 4 inputs: x[total], cos_table[P*seq_len], sin_table[P*seq_len], seq_len_i64, head_dim_i64
//   But StableHLO does not have dynamic indexing, so we use a simpler approach:
//   pass x, cos_broadcast, sin_broadcast (already expanded to full tensor shape)
//   and do a single element-wise multiply+add.
//
// To keep the HLO simple we take:
//   x_even [total_heads * seq_len * num_pairs]  (even elements)
//   x_odd  [total_heads * seq_len * num_pairs]  (odd elements)
//   cos_t  [seq_len * num_pairs] broadcast across total_heads
//   sin_t  [seq_len * num_pairs] broadcast across total_heads
//
// The Go wrapper does the scatter/gather of even/odd elements and the broadcast.
static std::string build_rope_hlo(int n, int num_pairs) {
    // n = total_heads * seq_len * num_pairs
    std::string t = "tensor<" + std::to_string(n) + "xf64>";

    return
        "module @rope {\n"
        "  func.func @main(\n"
        "    %x_even: " + t + ",\n"
        "    %x_odd:  " + t + ",\n"
        "    %cos_t:  " + t + ",\n"
        "    %sin_t:  " + t + "\n"
        "  ) -> (" + t + ", " + t + ") {\n"
        "    %even_cos = stablehlo.multiply %x_even, %cos_t : " + t + "\n"
        "    %odd_sin = stablehlo.multiply %x_odd, %sin_t : " + t + "\n"
        "    %out_even = stablehlo.subtract %even_cos, %odd_sin : " + t + "\n"
        "    %even_sin = stablehlo.multiply %x_even, %sin_t : " + t + "\n"
        "    %odd_cos = stablehlo.multiply %x_odd, %cos_t : " + t + "\n"
        "    %out_odd = stablehlo.add %even_sin, %odd_cos : " + t + "\n"
        "    return %out_even, %out_odd : " + t + ", " + t + "\n"
        "  }\n"
        "}\n";
}

// ALiBi module: out[h*Tq*Tk + q*Tk + k] = slopes[h] * (k - q)
// We express: slopes broadcast [num_heads, 1, 1], iota_k [1,1,seq_k], iota_q [1,seq_q,1]
// rel = iota_k - iota_q, bias = slopes * rel
// For simplicity, accept flattened slopes and rel_pos arrays:
static std::string build_alibi_hlo(int total) {
    std::string t = "tensor<" + std::to_string(total) + "xf64>";

    return
        "module @alibi {\n"
        "  func.func @main(\n"
        "    %slopes_bc: " + t + ",\n"
        "    %rel_pos:   " + t + "\n"
        "  ) -> " + t + " {\n"
        "    %out = stablehlo.multiply %slopes_bc, %rel_pos : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

// ---------------------------------------------------------------------------
// Compile helper
// ---------------------------------------------------------------------------

static PJRT_LoadedExecutable* pos_compile(const char* hlo_text) {
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = const_cast<char*>(hlo_text);
    prog.code_size   = strlen(hlo_text);
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size  = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client       = gp_client;
    ca.program      = &prog;
    set_single_device_compile_options(&ca);

    PJRT_Error* err = gp_api->PJRT_Client_Compile(&ca);
    if (!pos_check(gp_api, err, "PJRT_Client_Compile")) return nullptr;
    return ca.executable;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int xla_positional_init(const char* platform) {
    gp_api = pos_load_pjrt_plugin(platform);
    if (!gp_api) return -1;

    PJRT_Client_Create_Args args{};
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* err = gp_api->PJRT_Client_Create(&args);
    if (!pos_check(gp_api, err, "PJRT_Client_Create")) return -1;
    gp_client = args.client;
    return 0;
}

int xla_compile_positional(int total_heads, int seq_len, int head_dim,
                            int num_heads_alibi, int seq_len_q, int seq_len_k) {
    // RoPE
    int num_pairs = head_dim / 2;
    int rope_n    = total_heads * seq_len * num_pairs;
    std::string rope_hlo = build_rope_hlo(rope_n, num_pairs);
    gp_rope_exec = pos_compile(rope_hlo.c_str());
    if (!gp_rope_exec) return -1;
    gp_rope_total_heads = total_heads;
    gp_rope_seq_len     = seq_len;
    gp_rope_head_dim    = head_dim;

    // ALiBi
    int alibi_total = num_heads_alibi * seq_len_q * seq_len_k;
    std::string alibi_hlo = build_alibi_hlo(alibi_total);
    gp_alibi_exec = pos_compile(alibi_hlo.c_str());
    if (!gp_alibi_exec) return -1;
    gp_alibi_num_heads = num_heads_alibi;
    gp_alibi_seq_len_q = seq_len_q;
    gp_alibi_seq_len_k = seq_len_k;

    return 0;
}

// ---------------------------------------------------------------------------
// Buffer dispatch helpers (reuse pattern from activation_xla.cc)
// ---------------------------------------------------------------------------

static PJRT_Buffer* make_device_buf(const double* host, int n) {
    PJRT_Client_BufferFromHostBuffer_Args ba{};
    ba.struct_size    = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client         = gp_client;
    ba.data           = host;
    ba.type           = PJRT_Buffer_Type_F64;
    int64_t dims[1]   = { (int64_t)n };
    ba.dims           = dims;
    ba.num_dims       = 1;
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;

    PJRT_Error* err = gp_api->PJRT_Client_BufferFromHostBuffer(&ba);
    if (!pos_check(gp_api, err, "BufferFromHostBuffer")) return nullptr;

    // Wait for transfer
    PJRT_Buffer_ReadyEvent_Args re{};
    re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    re.buffer = ba.buffer;
    err = gp_api->PJRT_Buffer_ReadyEvent(&re);
    pos_free_error(gp_api, err);

    PJRT_Event_Await_Args ea{};
    ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea.event = re.event;
    err = gp_api->PJRT_Event_Await(&ea);
    pos_free_error(gp_api, err);

    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event = re.event;
    gp_api->PJRT_Event_Destroy(&eda);

    return ba.buffer;
}

static int copy_to_host(PJRT_Buffer* buf, double* host, size_t bytes) {
    PJRT_Buffer_ToHostBuffer_Args ta{};
    ta.struct_size  = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    ta.src          = buf;
    ta.dst          = host;
    ta.dst_size     = bytes;
    PJRT_Error* err = gp_api->PJRT_Buffer_ToHostBuffer(&ta);
    if (!pos_check(gp_api, err, "ToHostBuffer")) return -1;

    PJRT_Event_Await_Args ea{};
    ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ea.event = ta.event;
    err = gp_api->PJRT_Event_Await(&ea);
    pos_free_error(gp_api, err);

    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event = ta.event;
    gp_api->PJRT_Event_Destroy(&eda);
    return 0;
}

static void destroy_buf(PJRT_Buffer* buf) {
    if (!buf) return;
    PJRT_Buffer_Destroy_Args da{};
    da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    da.buffer = buf;
    gp_api->PJRT_Buffer_Destroy(&da);
}

// ---------------------------------------------------------------------------

int xla_rope(
    const double* x,
    double*       out,
    const double* cos_table,
    const double* sin_table,
    int           seq_len,
    int           head_dim,
    int           total_heads)
{
    int num_pairs = head_dim / 2;
    int n         = total_heads * seq_len * num_pairs;

    // Split x into even/odd arrays
    std::vector<double> x_even(n), x_odd(n);
    for (int slot = 0; slot < total_heads * seq_len; slot++) {
        for (int i = 0; i < num_pairs; i++) {
            x_even[slot * num_pairs + i] = x[slot * head_dim + i * 2];
            x_odd [slot * num_pairs + i] = x[slot * head_dim + i * 2 + 1];
        }
    }

    // Broadcast cos/sin tables across total_heads
    std::vector<double> cos_bc(n), sin_bc(n);
    for (int bh = 0; bh < total_heads; bh++) {
        memcpy(&cos_bc[bh * seq_len * num_pairs], cos_table,
               (size_t)(seq_len * num_pairs) * sizeof(double));
        memcpy(&sin_bc[bh * seq_len * num_pairs], sin_table,
               (size_t)(seq_len * num_pairs) * sizeof(double));
    }

    PJRT_Buffer* b_even = make_device_buf(x_even.data(), n);
    PJRT_Buffer* b_odd  = make_device_buf(x_odd.data(),  n);
    PJRT_Buffer* b_cos  = make_device_buf(cos_bc.data(), n);
    PJRT_Buffer* b_sin  = make_device_buf(sin_bc.data(), n);
    if (!b_even || !b_odd || !b_cos || !b_sin) return -1;

    PJRT_Buffer* arg_list[4] = { b_even, b_odd, b_cos, b_sin };
    PJRT_Buffer** input_lists[1] = { arg_list };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size         = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable          = gp_rope_exec;
    ea.num_devices         = 1;
    ea.num_args            = 4;
    ea.argument_lists      = input_lists;

    PJRT_Buffer* out_bufs[2] = {};
    PJRT_Buffer** out_lists[1] = { out_bufs };
    ea.output_lists = out_lists;

    PJRT_Error* err = gp_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!pos_check(gp_api, err, "Execute rope")) return -1;

    // Gather even/odd back into output
    std::vector<double> out_even(n), out_odd(n);
    if (copy_to_host(out_bufs[0], out_even.data(), n * sizeof(double)) != 0) return -1;
    if (copy_to_host(out_bufs[1], out_odd.data(),  n * sizeof(double)) != 0) return -1;

    for (int slot = 0; slot < total_heads * seq_len; slot++) {
        for (int i = 0; i < num_pairs; i++) {
            out[slot * head_dim + i * 2]     = out_even[slot * num_pairs + i];
            out[slot * head_dim + i * 2 + 1] = out_odd [slot * num_pairs + i];
        }
    }

    destroy_buf(b_even); destroy_buf(b_odd);
    destroy_buf(b_cos);  destroy_buf(b_sin);
    destroy_buf(out_bufs[0]); destroy_buf(out_bufs[1]);
    return 0;
}

int xla_alibi(
    double*       out,
    const double* slopes,
    int           num_heads,
    int           seq_len_q,
    int           seq_len_k,
    int           causal)
{
    int total = num_heads * seq_len_q * seq_len_k;

    // Build slopes_bc and rel_pos arrays
    std::vector<double> slopes_bc(total), rel_pos(total);
    for (int h = 0; h < num_heads; h++) {
        for (int q = 0; q < seq_len_q; q++) {
            for (int k = 0; k < seq_len_k; k++) {
                int idx = h * seq_len_q * seq_len_k + q * seq_len_k + k;
                slopes_bc[idx] = slopes[h];
                double rel = (double)(k - q);
                if (!causal && rel < 0.0) rel = -rel;
                rel_pos[idx] = rel;
            }
        }
    }

    PJRT_Buffer* b_slopes = make_device_buf(slopes_bc.data(), total);
    PJRT_Buffer* b_rel    = make_device_buf(rel_pos.data(),   total);
    if (!b_slopes || !b_rel) return -1;

    PJRT_Buffer* arg_list[2] = { b_slopes, b_rel };
    PJRT_Buffer** input_lists[1] = { arg_list };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size    = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable     = gp_alibi_exec;
    ea.num_devices    = 1;
    ea.num_args       = 2;
    ea.argument_lists = input_lists;

    PJRT_Buffer* out_buf = nullptr;
    PJRT_Buffer** out_lists[1] = { &out_buf };
    ea.output_lists = out_lists;

    PJRT_Error* err = gp_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!pos_check(gp_api, err, "Execute alibi")) return -1;

    if (copy_to_host(out_buf, out, (size_t)total * sizeof(double)) != 0) return -1;

    destroy_buf(b_slopes); destroy_buf(b_rel); destroy_buf(out_buf);
    return 0;
}

void xla_positional_shutdown() {
    if (gp_rope_exec) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = gp_rope_exec;
        gp_api->PJRT_LoadedExecutable_Destroy(&da);
        gp_rope_exec = nullptr;
    }
    if (gp_alibi_exec) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = gp_alibi_exec;
        gp_api->PJRT_LoadedExecutable_Destroy(&da);
        gp_alibi_exec = nullptr;
    }
    if (gp_client) {
        PJRT_Client_Destroy_Args da{};
        da.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        da.client      = gp_client;
        gp_api->PJRT_Client_Destroy(&da);
        gp_client = nullptr;
    }
}
