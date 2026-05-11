// XLA token-embedding backend — PJRT C API implementation.
//
// The embedding is expressed as stablehlo.gather with the weight matrix as
// the source operand and token IDs as the start indices.
//
// Compile requirements:
//   - openxla/xla headers on the include path (XLA_INCLUDE / CGO_CPPFLAGS)
//   - Link against the PJRT plugin shared library for the target platform

#include "embedding.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "xla/pjrt/c/pjrt_c_api.h"
#include <dlfcn.h>

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

static const PJRT_Api*        g_emb_api    = nullptr;
static PJRT_Client*           g_emb_client = nullptr;
static PJRT_LoadedExecutable* g_emb_exec   = nullptr;

// Cache keys for the compiled executable.
static int g_emb_n          = 0;
static int g_emb_d_model    = 0;
static int g_emb_vocab_size = 0;

// ---------------------------------------------------------------------------
// Error helpers (same pattern as activation_xla.cc)
// ---------------------------------------------------------------------------

static void emb_free_error(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) return;
    PJRT_Error_Destroy_Args da{};
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
}

static bool emb_check(const PJRT_Api* api, PJRT_Error* err, const char* ctx) {
    if (!err) return true;
    PJRT_Error_Message_Args ma{};
    ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    ma.error = err;
    api->PJRT_Error_Message(&ma);
    fprintf(stderr, "XLA PJRT error in %s: %.*s\n", ctx, (int)ma.message_size, ma.message);
    emb_free_error(api, err);
    return false;
}

// ---------------------------------------------------------------------------
// Plugin loader
// ---------------------------------------------------------------------------

typedef const PJRT_Api* (*GetPjrtApiFn)();

static const PJRT_Api* emb_load_plugin(const char* platform) {
    std::string path = pjrt_plugin_path(platform);
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        fprintf(stderr, "XLA embedding: failed to dlopen %s: %s\n", path.c_str(), dlerror());
        return nullptr;
    }
    auto get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) {
        fprintf(stderr, "XLA embedding: GetPjrtApi not found in %s\n", path.c_str());
        return nullptr;
    }
    return get_api();
}

// ---------------------------------------------------------------------------
// StableHLO module builder
//
// The module takes two inputs:
//   %weight : tensor<vocab_size x d_model x f64>  — flat weight table
//   %ids    : tensor<n x i64>                      — token IDs
// and produces:
//   %out    : tensor<n x d_model x f64>
//
// We use stablehlo.gather to extract the rows.
// ---------------------------------------------------------------------------

static std::string build_token_embedding(int n, int d_model, int vocab_size) {
    // Tensor type strings
    auto ts = [](int rows, int cols, const char* dtype) -> std::string {
        return "tensor<" + std::to_string(rows) + "x" + std::to_string(cols) + "x" + dtype + ">";
    };
    auto ts1 = [](int size, const char* dtype) -> std::string {
        return "tensor<" + std::to_string(size) + "x" + dtype + ">";
    };

    std::string w_type   = ts(vocab_size, d_model, "f64");
    std::string id_type  = ts1(n, "i64");
    std::string out_type = ts(n, d_model, "f64");

    return
        "module @token_embedding {\n"
        "  func.func @main(%weight: " + w_type + ", %ids: " + id_type + ") -> " + out_type + " {\n"
        "    %out = stablehlo.gather %weight, %ids,\n"
        "      offset_dims = [1],\n"
        "      collapsed_slice_dims = [0],\n"
        "      start_index_map = [0],\n"
        "      index_vector_dim = 1,\n"
        "      slice_sizes = [1, " + std::to_string(d_model) + "]\n"
        "      : (" + w_type + ", " + id_type + ") -> " + out_type + "\n"
        "    return %out : " + out_type + "\n"
        "  }\n"
        "}\n";
}

// ---------------------------------------------------------------------------
// Compile helper (mirrors activation_xla.cc)
// ---------------------------------------------------------------------------

static PJRT_LoadedExecutable* emb_compile(const std::string& mlir_text) {
    PJRT_Program prog{};
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code        = const_cast<char*>(mlir_text.c_str());
    prog.code_size   = mlir_text.size();
    prog.format      = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca{};
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client      = g_emb_client;
    ca.program      = &prog;
    set_single_device_compile_options(&ca);

    PJRT_Error* err = g_emb_api->PJRT_Client_Compile(&ca);
    if (!emb_check(g_emb_api, err, "PJRT_Client_Compile(embedding)")) return nullptr;
    return ca.executable;
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

static PJRT_Buffer* make_buffer_f64(const double* data, int64_t rows, int64_t cols) {
    PJRT_Client_BufferFromHostBuffer_Args ba{};
    ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client      = g_emb_client;
    ba.data        = data;
    ba.type        = PJRT_Buffer_Type_F64;
    int64_t dims[2] = { rows, cols };
    ba.dims     = dims;
    ba.num_dims = (cols > 1) ? 2 : 1; // 1-D for token IDs
    if (cols <= 1) {
        ba.dims    = &rows;
        ba.num_dims = 1;
    }
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;

    PJRT_Error* err = g_emb_api->PJRT_Client_BufferFromHostBuffer(&ba);
    if (!emb_check(g_emb_api, err, "BufferFromHostBuffer")) return nullptr;

    // Wait for transfer.
    PJRT_Buffer_ReadyEvent_Args re{};
    re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    re.buffer = ba.buffer;
    err = g_emb_api->PJRT_Buffer_ReadyEvent(&re);
    if (!emb_check(g_emb_api, err, "ReadyEvent")) return nullptr;

    PJRT_Event_Await_Args ev{};
    ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ev.event = re.event;
    err = g_emb_api->PJRT_Event_Await(&ev);
    emb_check(g_emb_api, err, "Event_Await");

    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event = re.event;
    g_emb_api->PJRT_Event_Destroy(&eda);

    return ba.buffer;
}

// Build a 1-D i64 buffer from double token IDs (convert double → int64).
static PJRT_Buffer* make_token_buffer(const double* tokens, int n) {
    // PJRT wants native i64; convert.
    int64_t* ids = new int64_t[n];
    for (int i = 0; i < n; i++) ids[i] = (int64_t)tokens[i];

    PJRT_Client_BufferFromHostBuffer_Args ba{};
    ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client      = g_emb_client;
    ba.data        = ids;
    ba.type        = PJRT_Buffer_Type_S64;
    int64_t dims[1] = { (int64_t)n };
    ba.dims     = dims;
    ba.num_dims = 1;
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;

    PJRT_Error* err = g_emb_api->PJRT_Client_BufferFromHostBuffer(&ba);
    delete[] ids;
    if (!emb_check(g_emb_api, err, "BufferFromHostBuffer(ids)")) return nullptr;

    PJRT_Buffer_ReadyEvent_Args re{};
    re.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    re.buffer = ba.buffer;
    err = g_emb_api->PJRT_Buffer_ReadyEvent(&re);
    if (!emb_check(g_emb_api, err, "ReadyEvent(ids)")) return nullptr;

    PJRT_Event_Await_Args ev{};
    ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ev.event = re.event;
    err = g_emb_api->PJRT_Event_Await(&ev);
    emb_check(g_emb_api, err, "Event_Await(ids)");

    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event = re.event;
    g_emb_api->PJRT_Event_Destroy(&eda);

    return ba.buffer;
}

static void emb_destroy_buffer(PJRT_Buffer* b) {
    if (!b) return;
    PJRT_Buffer_Destroy_Args da{};
    da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    da.buffer = b;
    g_emb_api->PJRT_Buffer_Destroy(&da);
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

int xla_embedding_init(const char* platform) {
    g_emb_api = emb_load_plugin(platform);
    if (!g_emb_api) return -1;

    PJRT_Client_Create_Args ca{};
    ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    PJRT_Error* err = g_emb_api->PJRT_Client_Create(&ca);
    if (!emb_check(g_emb_api, err, "PJRT_Client_Create")) return -1;

    g_emb_client = ca.client;
    return 0;
}

int xla_compile_embedding(int n, int d_model, int vocab_size) {
    if (!g_emb_client) return -1;

    // Destroy previous executable if dimensions changed.
    if (g_emb_exec && (g_emb_n != n || g_emb_d_model != d_model || g_emb_vocab_size != vocab_size)) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = g_emb_exec;
        g_emb_api->PJRT_LoadedExecutable_Destroy(&da);
        g_emb_exec = nullptr;
    }

    if (g_emb_exec) return 0; // already compiled for these dims

    std::string mlir = build_token_embedding(n, d_model, vocab_size);
    g_emb_exec = emb_compile(mlir);
    if (!g_emb_exec) return -1;

    g_emb_n          = n;
    g_emb_d_model    = d_model;
    g_emb_vocab_size = vocab_size;
    return 0;
}

int xla_token_embedding(
    const double* tokens,
    double*       out,
    const double* weight,
    int           n,
    int           d_model,
    int           vocab_size)
{
    if (xla_compile_embedding(n, d_model, vocab_size) != 0) return -1;

    // Build input buffers.
    PJRT_Buffer* buf_weight = make_buffer_f64(weight, (int64_t)vocab_size, (int64_t)d_model);
    if (!buf_weight) return -1;

    PJRT_Buffer* buf_tokens = make_token_buffer(tokens, n);
    if (!buf_tokens) { emb_destroy_buffer(buf_weight); return -1; }

    // Execute.
    PJRT_Buffer* in_list[2]   = { buf_weight, buf_tokens };
    PJRT_Buffer* out_buf_stor = nullptr;
    PJRT_Buffer** out_list[1] = { &out_buf_stor };

    PJRT_LoadedExecutable_Execute_Args ea{};
    ea.struct_size     = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable      = g_emb_exec;
    ea.argument_lists  = (PJRT_Buffer***)&in_list;
    ea.num_devices     = 1;
    ea.num_args        = 2;
    ea.output_lists    = out_list;
    PJRT_ExecuteOptions options = single_device_execute_options();
    ea.options = &options;

    PJRT_Error* err = g_emb_api->PJRT_LoadedExecutable_Execute(&ea);
    if (!emb_check(g_emb_api, err, "Execute(embedding)")) {
        emb_destroy_buffer(buf_weight);
        emb_destroy_buffer(buf_tokens);
        return -1;
    }

    PJRT_Buffer* out_buf = out_buf_stor;

    // Copy device → host.
    PJRT_Buffer_ToHostBuffer_Args tha{};
    tha.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    tha.src         = out_buf;
    tha.dst         = out;
    tha.dst_size    = (size_t)n * (size_t)d_model * sizeof(double);

    err = g_emb_api->PJRT_Buffer_ToHostBuffer(&tha);
    if (!emb_check(g_emb_api, err, "ToHostBuffer(embedding)")) {
        emb_destroy_buffer(buf_weight);
        emb_destroy_buffer(buf_tokens);
        emb_destroy_buffer(out_buf);
        return -1;
    }

    PJRT_Event_Await_Args ev{};
    ev.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    ev.event = tha.event;
    err = g_emb_api->PJRT_Event_Await(&ev);
    emb_check(g_emb_api, err, "Event_Await(out)");

    PJRT_Event_Destroy_Args eda{};
    eda.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    eda.event = tha.event;
    g_emb_api->PJRT_Event_Destroy(&eda);

    emb_destroy_buffer(buf_weight);
    emb_destroy_buffer(buf_tokens);
    emb_destroy_buffer(out_buf);
    return 0;
}

void xla_embedding_shutdown(void) {
    if (g_emb_exec) {
        PJRT_LoadedExecutable_Destroy_Args da{};
        da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        da.executable  = g_emb_exec;
        g_emb_api->PJRT_LoadedExecutable_Destroy(&da);
        g_emb_exec = nullptr;
    }
    if (g_emb_client) {
        PJRT_Client_Destroy_Args da{};
        da.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        da.client      = g_emb_client;
        g_emb_api->PJRT_Client_Destroy(&da);
        g_emb_client = nullptr;
    }
}

} // extern "C"
