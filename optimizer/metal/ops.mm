#include <torch/extension.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <string>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace fs = std::filesystem;

namespace {

// Must match `DBAParams` in `dba_decode.metal` (layout + types).
struct DBAParams {
  uint32_t sem_head_dim;
  uint32_t geo_head_dim;
  uint32_t v_head_dim;
  uint32_t n_heads;
  uint32_t seq_len;
  float sem_scale;
  float geo_scale;
  uint32_t ksem_stride_b;
  uint32_t ksem_stride_t;
  uint32_t kgeo_stride_b;
  uint32_t kgeo_stride_t;
  uint32_t v_stride_b;
  uint32_t v_stride_t;
};

// Must match `RMSNormParams` in `rmsnorm.metal` (layout + types).
struct RMSNormParams {
  uint32_t d_model;
  float eps;
  uint32_t stride_row;
};

// Must match `RoPEParams` in `rope.metal` (layout + types).
struct RoPEParams {
  uint32_t d_model;
  uint32_t rot_dim;
  uint32_t half_rot;
  uint32_t seq_len;
};

// Must match `LionParams` in `lion.metal`.
struct LionParams {
  uint32_t n;
  float lr;
  float beta1;
  float weight_decay;
};

constexpr NSUInteger kThreadsPerThreadgroup = 256;

static id<MTLLibrary> g_lib = nil;
static id<MTLComputePipelineState> g_pipeline_dba_decode = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_noweight = nil;
static id<MTLComputePipelineState> g_pipeline_rope = nil;
static id<MTLComputePipelineState> g_pipeline_lion = nil;
static std::mutex g_pipeline_mutex;

static std::string metallib_path_for_this_module() {
  Dl_info info;
  if (dladdr((void*)&metallib_path_for_this_module, &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  fs::path so_path(info.dli_fname);
  fs::path lib_path = so_path.parent_path() / "caramba_ops.metallib";
  return lib_path.string();
}

static void ensure_library_locked(id<MTLDevice> device) {
  if (g_lib != nil) {
    return;
  }

  const std::string lib_path = metallib_path_for_this_module();
  TORCH_CHECK(!lib_path.empty(), "caramba_metal_ops: failed to locate extension path via dladdr()");

  NSString* ns_path = [NSString stringWithUTF8String:lib_path.c_str()];
  NSError* err = nil;
  g_lib = [device newLibraryWithFile:ns_path error:&err];
  if (g_lib == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_ops: failed to load metallib at ", lib_path, ": ", msg);
  }
}

static id<MTLComputePipelineState> ensure_pipeline(
    id<MTLDevice> device,
    id<MTLComputePipelineState>* pipeline,
    const char* fn_name) {
  std::lock_guard<std::mutex> lock(g_pipeline_mutex);
  ensure_library_locked(device);

  if (*pipeline != nil) {
    return *pipeline;
  }

  NSString* ns_fn = [NSString stringWithUTF8String:fn_name];
  id<MTLFunction> fn = [g_lib newFunctionWithName:ns_fn];
  TORCH_CHECK(fn != nil, "caramba_metal_ops: function `", fn_name, "` not found in metallib");

  NSError* err = nil;
  *pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
  if (*pipeline == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_ops: failed to create compute pipeline: ", msg);
  }

  // Basic sanity check against accidental dispatch mismatch.
  TORCH_CHECK(
      (*pipeline).maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup,
      "caramba_metal_ops: pipeline maxTotalThreadsPerThreadgroup (",
      (int)(*pipeline).maxTotalThreadsPerThreadgroup,
      ") < expected threads (",
      (int)kThreadsPerThreadgroup,
      ")");

  return *pipeline;
}

static inline id<MTLBuffer> storage_as_mtlbuffer(const at::Tensor& t) {
  // MPS tensors are backed by MTLBuffer allocations; the storage base pointer is
  // an opaque handle that is compatible with `id<MTLBuffer>` in practice.
  //
  // NOTE: This is intentionally low-level to avoid CPU staging/copies.
  const auto& dp = t.storage().data_ptr();
  void* ctx = dp.get_context();
  if (ctx == nullptr) {
    ctx = dp.get();
  }
  return (id<MTLBuffer>)ctx;
}

static inline NSUInteger storage_offset_bytes(const at::Tensor& t) {
  return (NSUInteger)(t.storage_offset() * (int64_t)t.element_size());
}

torch::Tensor dba_decode(
    at::Tensor q_sem, // (B,H,sem_hd) fp16 MPS
    at::Tensor k_sem, // (B,S,H*sem_hd) fp16 MPS
    at::Tensor q_geo, // (B,H,geo_hd) fp16 MPS
    at::Tensor k_geo, // (B,S,H*geo_hd) fp16 MPS
    at::Tensor v, // (B,S,H*v_hd) fp16 MPS
    double sem_scale,
    double geo_scale) {
  TORCH_CHECK(q_sem.device().is_mps(), "dba_decode: q_sem must be on MPS");
  TORCH_CHECK(q_geo.device().is_mps(), "dba_decode: q_geo must be on MPS");
  TORCH_CHECK(k_sem.device().is_mps(), "dba_decode: k_sem must be on MPS");
  TORCH_CHECK(k_geo.device().is_mps(), "dba_decode: k_geo must be on MPS");
  TORCH_CHECK(v.device().is_mps(), "dba_decode: v must be on MPS");

  TORCH_CHECK(q_sem.dtype() == at::kHalf, "dba_decode: q_sem must be fp16");
  TORCH_CHECK(q_geo.dtype() == at::kHalf, "dba_decode: q_geo must be fp16");
  TORCH_CHECK(k_sem.dtype() == at::kHalf, "dba_decode: k_sem must be fp16");
  TORCH_CHECK(k_geo.dtype() == at::kHalf, "dba_decode: k_geo must be fp16");
  TORCH_CHECK(v.dtype() == at::kHalf, "dba_decode: v must be fp16");

  TORCH_CHECK(q_sem.is_contiguous(), "dba_decode: q_sem must be contiguous");
  TORCH_CHECK(q_geo.is_contiguous(), "dba_decode: q_geo must be contiguous");
  // K/V are often views into preallocated cache buffers; batch stride may be larger
  // than (S * D). We only require the last dimension to be contiguous.
  TORCH_CHECK(k_sem.stride(2) == 1, "dba_decode: k_sem last dim must be contiguous (stride==1)");
  TORCH_CHECK(k_geo.stride(2) == 1, "dba_decode: k_geo last dim must be contiguous (stride==1)");
  TORCH_CHECK(v.stride(2) == 1, "dba_decode: v last dim must be contiguous (stride==1)");

  TORCH_CHECK(q_sem.dim() == 3, "dba_decode: q_sem must be (B,H,D)");
  TORCH_CHECK(q_geo.dim() == 3, "dba_decode: q_geo must be (B,H,D)");
  TORCH_CHECK(k_sem.dim() == 3, "dba_decode: k_sem must be (B,S,D)");
  TORCH_CHECK(k_geo.dim() == 3, "dba_decode: k_geo must be (B,S,D)");
  TORCH_CHECK(v.dim() == 3, "dba_decode: v must be (B,S,D)");

  const int64_t B = q_sem.size(0);
  const int64_t H = q_sem.size(1);
  const int64_t sem_hd = q_sem.size(2);
  const int64_t geo_hd = q_geo.size(2);

  TORCH_CHECK(q_geo.size(0) == B && q_geo.size(1) == H, "dba_decode: q_geo shape mismatch");
  TORCH_CHECK(k_sem.size(0) == B, "dba_decode: k_sem batch mismatch");
  TORCH_CHECK(k_geo.size(0) == B, "dba_decode: k_geo batch mismatch");
  TORCH_CHECK(v.size(0) == B, "dba_decode: v batch mismatch");

  const int64_t S = k_sem.size(1);
  TORCH_CHECK(k_geo.size(1) == S, "dba_decode: k_geo seq mismatch");
  TORCH_CHECK(v.size(1) == S, "dba_decode: v seq mismatch");

  TORCH_CHECK(k_sem.size(2) == H * sem_hd, "dba_decode: k_sem last dim must be H*sem_hd");
  TORCH_CHECK(k_geo.size(2) == H * geo_hd, "dba_decode: k_geo last dim must be H*geo_hd");
  TORCH_CHECK(v.size(2) % H == 0, "dba_decode: v last dim must be divisible by H");

  const int64_t v_hd = v.size(2) / H;

  // Output: (B,H,v_hd)
  auto out = torch::empty({B, H, v_hd}, q_sem.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_dba_decode, "dba_decode_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "dba_decode: failed to get current MPS stream");

  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "dba_decode: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "dba_decode: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(q_sem, 0);
  set_tensor(k_sem, 1);
  set_tensor(q_geo, 2);
  set_tensor(k_geo, 3);
  set_tensor(v, 4);
  set_tensor(out, 5);

  DBAParams params;
  params.sem_head_dim = (uint32_t)sem_hd;
  params.geo_head_dim = (uint32_t)geo_hd;
  params.v_head_dim = (uint32_t)v_hd;
  params.n_heads = (uint32_t)H;
  params.seq_len = (uint32_t)S;
  params.sem_scale = (float)sem_scale;
  params.geo_scale = (float)geo_scale;
  params.ksem_stride_b = (uint32_t)k_sem.stride(0);
  params.ksem_stride_t = (uint32_t)k_sem.stride(1);
  params.kgeo_stride_b = (uint32_t)k_geo.stride(0);
  params.kgeo_stride_t = (uint32_t)k_geo.stride(1);
  params.v_stride_b = (uint32_t)v.stride(0);
  params.v_stride_t = (uint32_t)v.stride(1);

  [encoder setBytes:&params length:sizeof(DBAParams) atIndex:6];

  // Threadgroups: (1, H, B), one threadgroup per (batch, head).
  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake(1, (NSUInteger)H, (NSUInteger)B);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return out;
}

torch::Tensor rmsnorm(
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor weight, // (D,) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "rmsnorm: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "rmsnorm: weight must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf, "rmsnorm: x must be fp16");
  TORCH_CHECK(weight.dtype() == at::kHalf, "rmsnorm: weight must be fp16");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "rmsnorm: weight must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm: invalid last dim");
  TORCH_CHECK(weight.numel() == D, "rmsnorm: weight must have numel == x.size(-1)");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm: x.numel must be divisible by D");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_rmsnorm, "rmsnorm_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(out, 2);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:3];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return out;
}

torch::Tensor rmsnorm_noweight(
    at::Tensor x, // (..., D) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "rmsnorm_noweight: x must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf, "rmsnorm_noweight: x must be fp16");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_noweight: x must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm_noweight: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_noweight: invalid last dim");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_noweight: x.numel must be divisible by D");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_rmsnorm_noweight, "rmsnorm_noweight_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm_noweight: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm_noweight: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm_noweight: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(out, 1);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:2];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return out;
}

torch::Tensor rope(
    at::Tensor x,   // (B,H,T,D) fp16 MPS contiguous
    at::Tensor cos, // (T, rot/2) fp16 MPS contiguous
    at::Tensor sin, // (T, rot/2) fp16 MPS contiguous
    int64_t rot_dim) {
  TORCH_CHECK(x.device().is_mps(), "rope: x must be on MPS");
  TORCH_CHECK(cos.device().is_mps(), "rope: cos must be on MPS");
  TORCH_CHECK(sin.device().is_mps(), "rope: sin must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf, "rope: x must be fp16");
  TORCH_CHECK(cos.dtype() == at::kHalf, "rope: cos must be fp16");
  TORCH_CHECK(sin.dtype() == at::kHalf, "rope: sin must be fp16");
  TORCH_CHECK(x.is_contiguous(), "rope: x must be contiguous");
  TORCH_CHECK(cos.is_contiguous(), "rope: cos must be contiguous");
  TORCH_CHECK(sin.is_contiguous(), "rope: sin must be contiguous");

  TORCH_CHECK(x.dim() == 4, "rope: x must be (B,H,T,D)");
  TORCH_CHECK(cos.dim() == 2, "rope: cos must be (T, rot/2)");
  TORCH_CHECK(sin.dim() == 2, "rope: sin must be (T, rot/2)");

  TORCH_CHECK(rot_dim > 0, "rope: rot_dim must be > 0");
  TORCH_CHECK((rot_dim % 2) == 0, "rope: rot_dim must be even");

  const int64_t B = x.size(0);
  const int64_t H = x.size(1);
  const int64_t T = x.size(2);
  const int64_t D = x.size(3);
  TORCH_CHECK(rot_dim <= D, "rope: rot_dim must be <= head_dim");

  const int64_t half_rot = rot_dim / 2;
  TORCH_CHECK(cos.size(0) == T && cos.size(1) == half_rot, "rope: cos shape mismatch");
  TORCH_CHECK(sin.size(0) == T && sin.size(1) == half_rot, "rope: sin shape mismatch");

  auto out = torch::empty_like(x);
  const int64_t n_vec = B * H * T;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_rope, "rope_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rope: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rope: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rope: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(cos, 1);
  set_tensor(sin, 2);
  set_tensor(out, 3);

  RoPEParams params;
  params.d_model = (uint32_t)D;
  params.rot_dim = (uint32_t)rot_dim;
  params.half_rot = (uint32_t)half_rot;
  params.seq_len = (uint32_t)T;
  [encoder setBytes:&params length:sizeof(RoPEParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)n_vec, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return out;
}

torch::Tensor lion_step(
    at::Tensor p,    // fp16 MPS contiguous (updated in-place)
    at::Tensor grad, // fp16 MPS contiguous
    at::Tensor m,    // fp16 MPS contiguous (updated in-place)
    double lr,
    double beta1,
    double weight_decay) {
  TORCH_CHECK(p.device().is_mps(), "lion_step: p must be on MPS");
  TORCH_CHECK(grad.device().is_mps(), "lion_step: grad must be on MPS");
  TORCH_CHECK(m.device().is_mps(), "lion_step: m must be on MPS");

  TORCH_CHECK(p.dtype() == at::kHalf, "lion_step: p must be fp16");
  TORCH_CHECK(grad.dtype() == at::kHalf, "lion_step: grad must be fp16");
  TORCH_CHECK(m.dtype() == at::kHalf, "lion_step: m must be fp16");

  TORCH_CHECK(p.is_contiguous(), "lion_step: p must be contiguous");
  TORCH_CHECK(grad.is_contiguous(), "lion_step: grad must be contiguous");
  TORCH_CHECK(m.is_contiguous(), "lion_step: m must be contiguous");

  TORCH_CHECK(p.numel() == grad.numel(), "lion_step: p/grad numel mismatch");
  TORCH_CHECK(p.numel() == m.numel(), "lion_step: p/m numel mismatch");

  const int64_t n = p.numel();

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_lion, "lion_step_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "lion_step: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "lion_step: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "lion_step: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(p, 0);
  set_tensor(grad, 1);
  set_tensor(m, 2);

  LionParams prm;
  prm.n = (uint32_t)n;
  prm.lr = (float)lr;
  prm.beta1 = (float)beta1;
  prm.weight_decay = (float)weight_decay;
  [encoder setBytes:&prm length:sizeof(LionParams) atIndex:3];

  const NSUInteger tg_n = (NSUInteger)((n + (int64_t)kThreadsPerThreadgroup - 1) / (int64_t)kThreadsPerThreadgroup);
  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake(tg_n, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return p;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dba_decode", &dba_decode, "DBA Decode Forward (Metal/MPS, fp16)");
  m.def("rmsnorm", &rmsnorm, "RMSNorm Forward (Metal/MPS, fp16)");
  m.def("rmsnorm_noweight", &rmsnorm_noweight, "RMSNorm Forward (no weight, Metal/MPS, fp16)");
  m.def("rope", &rope, "RoPE Apply (Metal/MPS, fp16)");
  m.def("lion_step", &lion_step, "Lion step update (Metal/MPS, fp16)");
}

