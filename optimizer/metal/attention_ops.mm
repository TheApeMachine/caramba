#include <torch/extension.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

namespace fs = std::filesystem;

namespace {

struct AttnParams {
  uint32_t B;
  uint32_t H;
  uint32_t T;
  uint32_t D;
  float scale;
  uint32_t causal;
  float dropout_p;
  uint32_t seed;
  uint32_t q_stride_b;
  uint32_t q_stride_h;
  uint32_t q_stride_t;
  uint32_t k_stride_b;
  uint32_t k_stride_h;
  uint32_t k_stride_t;
  uint32_t v_stride_b;
  uint32_t v_stride_h;
  uint32_t v_stride_t;
  uint32_t o_stride_b;
  uint32_t o_stride_h;
  uint32_t o_stride_t;
  uint32_t lse_stride_b;
  uint32_t lse_stride_h;
  uint32_t lse_stride_t;
};

constexpr NSUInteger kThreadsPerThreadgroup = 256;

static id<MTLLibrary> g_lib = nil;
static id<MTLComputePipelineState> g_fwd = nil;
static id<MTLComputePipelineState> g_bwd_pre = nil;
static id<MTLComputePipelineState> g_bwd_dkv = nil;
static id<MTLComputePipelineState> g_bwd_dq = nil;
static std::mutex g_mutex;

static std::string metallib_path_for_this_module() {
  Dl_info info;
  if (dladdr((void*)&metallib_path_for_this_module, &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  fs::path so_path(info.dli_fname);
  fs::path lib_path = so_path.parent_path() / "caramba_attention_ops.metallib";
  return lib_path.string();
}

static void ensure_library_locked(id<MTLDevice> device) {
  if (g_lib != nil) {
    return;
  }
  const std::string lib_path = metallib_path_for_this_module();
  TORCH_CHECK(!lib_path.empty(), "caramba_metal_attention_ops: failed to locate extension path via dladdr()");
  NSString* ns_path = [NSString stringWithUTF8String:lib_path.c_str()];
  NSURL* url = [NSURL fileURLWithPath:ns_path];
  NSError* err = nil;
  g_lib = [device newLibraryWithURL:url error:&err];
  if (g_lib == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_attention_ops: failed to load metallib at ", lib_path, ": ", msg);
  }
}

static id<MTLComputePipelineState> ensure_pipeline(
    id<MTLDevice> device,
    id<MTLComputePipelineState> __strong* pipeline,
    const char* fn_name) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_library_locked(device);
  if (*pipeline != nil) {
    return *pipeline;
  }
  NSString* ns_fn = [NSString stringWithUTF8String:fn_name];
  id<MTLFunction> fn = [g_lib newFunctionWithName:ns_fn];
  TORCH_CHECK(fn != nil, "caramba_metal_attention_ops: function `", fn_name, "` not found in metallib");
  NSError* err = nil;
  *pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
  if (*pipeline == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_attention_ops: failed to create compute pipeline: ", msg);
  }
  TORCH_CHECK(
      (*pipeline).maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup,
      "caramba_metal_attention_ops: pipeline maxTotalThreadsPerThreadgroup (",
      (int)(*pipeline).maxTotalThreadsPerThreadgroup,
      ") < expected threads (",
      (int)kThreadsPerThreadgroup,
      ")");
  return *pipeline;
}

static inline NSUInteger simd_width_or_default(id<MTLComputePipelineState> pipeline) {
  // Metal reports the runtime SIMD-group width for this pipeline.
  // Older/edge toolchains may report 0; preserve prior behavior by falling back to 32.
  const NSUInteger w = pipeline.threadExecutionWidth;
  return w ? w : 32;
}

static inline id<MTLBuffer> storage_as_mtlbuffer(const at::Tensor& t) {
  const auto& dp = t.storage().data_ptr();
  void* ctx = dp.get_context();
  TORCH_CHECK(ctx != nullptr, "caramba_metal_attention_ops: expected MPS storage to provide an MTLBuffer context");
  return (__bridge id<MTLBuffer>)ctx;
}

static inline NSUInteger storage_offset_bytes(const at::Tensor& t) {
  return (NSUInteger)(t.storage_offset() * (int64_t)t.element_size());
}

static void check_attn_tensor(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, ": must be on MPS");
  TORCH_CHECK(t.dtype() == at::kHalf, name, ": must be fp16");
  TORCH_CHECK(t.dim() == 4, name, ": must be 4D (B,H,T,D)");
  TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
  TORCH_CHECK(t.stride(3) == 1, name, ": last dim must be contiguous (stride==1)");
}

static AttnParams make_params(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& lse_like,
    double scale,
    bool causal,
    double dropout_p,
    uint32_t seed) {
  constexpr int64_t kU32Max = (int64_t)std::numeric_limits<uint32_t>::max();
  const int64_t B = q.size(0);
  const int64_t H = q.size(1);
  const int64_t T = q.size(2);
  const int64_t D = q.size(3);
  TORCH_CHECK(D > 0 && D <= (int64_t)kThreadsPerThreadgroup, "attention_train: requires 1 <= head_dim <= 256, got ", D);
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "attention_train: q/k/v shapes must match");
  TORCH_CHECK(out.sizes() == q.sizes(), "attention_train: out shape mismatch");
  TORCH_CHECK(lse_like.dim() == 3 && lse_like.size(0) == B && lse_like.size(1) == H && lse_like.size(2) == T,
              "attention_train: lse/delta must be (B,H,T)");
  auto stride_u32 = [&](const at::Tensor& t, const char* tname, int idx) -> uint32_t {
    const int64_t s = t.stride(idx);
    TORCH_CHECK(
        s >= 0 && s <= kU32Max,
        "attention_train: ",
        tname,
        ".stride(",
        idx,
        ") out of range for uint32 (stride in elements must be within [0, UINT32_MAX]), got ",
        s);
    return static_cast<uint32_t>(s);
  };
  AttnParams p;
  p.B = (uint32_t)B;
  p.H = (uint32_t)H;
  p.T = (uint32_t)T;
  p.D = (uint32_t)D;
  p.scale = (float)scale;
  p.causal = causal ? 1u : 0u;
  p.dropout_p = (float)dropout_p;
  p.seed = seed;
  p.q_stride_b = stride_u32(q, "q", 0);
  p.q_stride_h = stride_u32(q, "q", 1);
  p.q_stride_t = stride_u32(q, "q", 2);
  p.k_stride_b = stride_u32(k, "k", 0);
  p.k_stride_h = stride_u32(k, "k", 1);
  p.k_stride_t = stride_u32(k, "k", 2);
  p.v_stride_b = stride_u32(v, "v", 0);
  p.v_stride_h = stride_u32(v, "v", 1);
  p.v_stride_t = stride_u32(v, "v", 2);
  p.o_stride_b = stride_u32(out, "out", 0);
  p.o_stride_h = stride_u32(out, "out", 1);
  p.o_stride_t = stride_u32(out, "out", 2);
  p.lse_stride_b = stride_u32(lse_like, "lse_like", 0);
  p.lse_stride_h = stride_u32(lse_like, "lse_like", 1);
  p.lse_stride_t = stride_u32(lse_like, "lse_like", 2);
  return p;
}

std::vector<torch::Tensor> attn_train_fwd(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    double scale,
    bool causal,
    double dropout_p,
    int64_t seed) {
  constexpr int64_t kU32Max = (int64_t)std::numeric_limits<uint32_t>::max();
  TORCH_CHECK(
      seed >= 0 && seed <= kU32Max,
      "attn_train_fwd: seed must be within [0, UINT32_MAX], got ",
      seed);
  const uint32_t seed_u32 = static_cast<uint32_t>(seed);
  check_attn_tensor(q, "attn_train_fwd: q");
  check_attn_tensor(k, "attn_train_fwd: k");
  check_attn_tensor(v, "attn_train_fwd: v");

  TORCH_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype(), "attn_train_fwd: q/k/v dtype mismatch");
  TORCH_CHECK(0.0 <= dropout_p && dropout_p < 1.0, "attn_train_fwd: dropout_p must be in [0,1)");

  auto out = torch::empty_like(q);
  auto lse = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(at::kFloat));

  const AttnParams p = make_params(q, k, v, out, lse, scale, causal, dropout_p, seed_u32);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = ensure_pipeline(device, &g_fwd, "attn_train_fwd_fp16");
  const NSUInteger simdWidth = simd_width_or_default(pipeline);
  // Invariant: simd_width_or_default(pipeline) returns >= 32 (fallback), and
  // kThreadsPerThreadgroup is 256, so simdgroups_per_threadgroup (ceil(256 / simdWidth)) is always >= 1.
  (void)simdWidth;

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "attn_train_fwd: failed to get current MPS stream");
  // IMPORTANT: MPSStream's Metal objects are synchronized on its serial queue.
  // Interacting with the command buffer/encoder off-queue can trigger Metal asserts.
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    TORCH_CHECK(encoder != nil, "attn_train_fwd: failed to get MTLComputeCommandEncoder from MPS stream");

    [encoder setComputePipelineState:pipeline];

    auto set_tensor = [&](const at::Tensor& t, int idx) {
      id<MTLBuffer> buf = storage_as_mtlbuffer(t);
      TORCH_CHECK(buf != nil, "attn_train_fwd: tensor has null MTLBuffer");
      [encoder setBuffer:buf offset:storage_offset_bytes(t) atIndex:(NSUInteger)idx];
    };

    set_tensor(q, 0);
    set_tensor(k, 1);
    set_tensor(v, 2);
    set_tensor(out, 3);
    set_tensor(lse, 4);
    [encoder setBytes:&p length:sizeof(AttnParams) atIndex:5];

    const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    const MTLSize grid = MTLSizeMake((NSUInteger)q.size(2), (NSUInteger)q.size(1), (NSUInteger)q.size(0));
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

    stream->endKernelCoalescing();
  });

  return {out, lse};
}

std::vector<torch::Tensor> attn_train_bwd(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor grad_out,
    double scale,
    bool causal,
    double dropout_p,
    int64_t seed) {
  constexpr int64_t kU32Max = (int64_t)std::numeric_limits<uint32_t>::max();
  TORCH_CHECK(
      seed >= 0 && seed <= kU32Max,
      "attn_train_bwd: seed must be within [0, UINT32_MAX], got ",
      seed);
  const uint32_t seed_u32 = static_cast<uint32_t>(seed);
  check_attn_tensor(q, "attn_train_bwd: q");
  check_attn_tensor(k, "attn_train_bwd: k");
  check_attn_tensor(v, "attn_train_bwd: v");
  check_attn_tensor(out, "attn_train_bwd: out");
  check_attn_tensor(grad_out, "attn_train_bwd: grad_out");
  TORCH_CHECK(lse.device().is_mps(), "attn_train_bwd: lse must be on MPS");
  TORCH_CHECK(lse.dtype() == at::kFloat, "attn_train_bwd: lse must be fp32");
  TORCH_CHECK(lse.is_contiguous(), "attn_train_bwd: lse must be contiguous");
  TORCH_CHECK(lse.dim() == 3, "attn_train_bwd: lse must be (B,H,T)");
  TORCH_CHECK(0.0 <= dropout_p && dropout_p < 1.0, "attn_train_bwd: dropout_p must be in [0,1)");

  auto delta = torch::empty_like(lse);
  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);

  const AttnParams p0 = make_params(q, k, v, out, lse, scale, causal, dropout_p, seed_u32);
  const AttnParams p1 = make_params(q, k, v, out, delta, scale, causal, dropout_p, seed_u32);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "attn_train_bwd: failed to get current MPS stream");
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    TORCH_CHECK(encoder != nil, "attn_train_bwd: failed to get MTLComputeCommandEncoder from MPS stream");

    auto set_tensor = [&](const at::Tensor& t, int idx) {
      id<MTLBuffer> buf = storage_as_mtlbuffer(t);
      TORCH_CHECK(buf != nil, "attn_train_bwd: tensor has null MTLBuffer");
      [encoder setBuffer:buf offset:storage_offset_bytes(t) atIndex:(NSUInteger)idx];
    };

    // 1) delta preprocess
    {
      id<MTLComputePipelineState> p_pre = ensure_pipeline(device, &g_bwd_pre, "attn_train_bwd_preprocess_fp16");
      const NSUInteger simdWidth = simd_width_or_default(p_pre);
      // See note above: simdWidth is guaranteed >= 32 (fallback), so simdgroups_per_threadgroup >= 1.
      (void)simdWidth;
      [encoder setComputePipelineState:p_pre];
      set_tensor(out, 0);
      set_tensor(grad_out, 1);
      set_tensor(delta, 2);
      [encoder setBytes:&p1 length:sizeof(AttnParams) atIndex:3];
      const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
      const MTLSize grid = MTLSizeMake((NSUInteger)q.size(2), (NSUInteger)q.size(1), (NSUInteger)q.size(0));
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }

    // 2) dK/dV
    {
      id<MTLComputePipelineState> p_dkv = ensure_pipeline(device, &g_bwd_dkv, "attn_train_bwd_dkv_fp16");
      const NSUInteger simdWidth = simd_width_or_default(p_dkv);
      // See note above: simdWidth is guaranteed >= 32 (fallback), so simdgroups_per_threadgroup >= 1.
      (void)simdWidth;
      [encoder setComputePipelineState:p_dkv];
      set_tensor(q, 0);
      set_tensor(k, 1);
      set_tensor(v, 2);
      set_tensor(grad_out, 3);
      set_tensor(lse, 4);
      set_tensor(delta, 5);
      set_tensor(dk, 6);
      set_tensor(dv, 7);
      [encoder setBytes:&p0 length:sizeof(AttnParams) atIndex:8];
      const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
      const MTLSize grid = MTLSizeMake((NSUInteger)q.size(2), (NSUInteger)q.size(1), (NSUInteger)q.size(0));
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }

    // 3) dQ
    {
      id<MTLComputePipelineState> p_dq = ensure_pipeline(device, &g_bwd_dq, "attn_train_bwd_dq_fp16");
      const NSUInteger simdWidth = simd_width_or_default(p_dq);
      // See note above: simdWidth is guaranteed >= 32 (fallback), so simdgroups_per_threadgroup >= 1.
      (void)simdWidth;
      [encoder setComputePipelineState:p_dq];
      set_tensor(q, 0);
      set_tensor(k, 1);
      set_tensor(v, 2);
      set_tensor(grad_out, 3);
      set_tensor(lse, 4);
      set_tensor(delta, 5);
      set_tensor(dq, 6);
      [encoder setBytes:&p0 length:sizeof(AttnParams) atIndex:7];
      const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
      const MTLSize grid = MTLSizeMake((NSUInteger)q.size(2), (NSUInteger)q.size(1), (NSUInteger)q.size(0));
      [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    }

    stream->endKernelCoalescing();
  });

  return {dq, dk, dv};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_train_fwd", &attn_train_fwd, "Attention Train Forward (Metal/MPS, fp16)");
  m.def("attn_train_bwd", &attn_train_bwd, "Attention Train Backward (Metal/MPS, fp16)");
}

