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

// Must match `RMSNormGradWParams` in `rmsnorm.metal`.
struct RMSNormGradWParams {
  uint32_t d_model;
  uint32_t rows;
  uint32_t stride_row;
};

// Must match `LayerNormParams` in `layernorm.metal` (layout + types).
struct LayerNormParams {
  uint32_t d_model;
  float eps;
  uint32_t stride_row;
};

// Must match `LayerNormGradWParams` in `layernorm.metal`.
struct LayerNormGradWParams {
  uint32_t d_model;
  uint32_t rows;
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

// Must match `AdamWParams` in `adamw.metal`.
struct AdamWParams {
  uint32_t n;
  float step_size;
  float beta1;
  float beta2;
  float eps;
  float lr_wd;
};

// Must match `SSMScanParams` in `ssm_scan.metal`.
struct SSMScanParams {
  uint32_t B;
  uint32_t T;
  uint32_t D_inner;
  uint32_t D_state;
};

constexpr NSUInteger kThreadsPerThreadgroup = 256;

static id<MTLLibrary> g_lib = nil;
static id<MTLComputePipelineState> g_pipeline_dba_decode = nil;
static id<MTLComputePipelineState> g_pipeline_dba_decode_null = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_noweight = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_noweight_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_fwd_inv = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_fwd_inv_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_noweight_fwd_inv = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_noweight_fwd_inv_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_bwd = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_bwd_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_bwd_noweight = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_bwd_noweight_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_gradw = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_gradw_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_weight = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_weight_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_noweight = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_noweight_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_fwd_stats = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_fwd_stats_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_weight_fwd_stats = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_weight_fwd_stats_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_noweight_fwd_stats = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_noweight_fwd_stats_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_bwd_x = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_bwd_x_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_bwd_x_noweight = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_bwd_x_noweight_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_gradw = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_gradw_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_gradb = nil;
static id<MTLComputePipelineState> g_pipeline_layernorm_gradb_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rope = nil;
static id<MTLComputePipelineState> g_pipeline_rope_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rope_bwd = nil;
static id<MTLComputePipelineState> g_pipeline_rope_bwd_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_lion = nil;
static id<MTLComputePipelineState> g_pipeline_lion_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_adamw_master = nil;
static id<MTLComputePipelineState> g_pipeline_adamw_master_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_ssm_fwd = nil;
static id<MTLComputePipelineState> g_pipeline_ssm_bwd_g = nil;
static id<MTLComputePipelineState> g_pipeline_ssm_gradB = nil;
static id<MTLComputePipelineState> g_pipeline_ssm_gradC = nil;
static id<MTLComputePipelineState> g_pipeline_ssm_gradD = nil;
static id<MTLComputePipelineState> g_pipeline_ssm_gradA = nil;
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
  NSURL* url = [NSURL fileURLWithPath:ns_path];
  NSError* err = nil;
  // newLibraryWithFile:error: is deprecated; use URL variant on newer macOS.
  g_lib = [device newLibraryWithURL:url error:&err];
  if (g_lib == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_ops: failed to load metallib at ", lib_path, ": ", msg);
  }
}

static id<MTLComputePipelineState> ensure_pipeline(
    id<MTLDevice> device,
    // Important: this is a STRONG out-parameter. If left un-annotated under ARC,
    // Clang treats it as __autoreleasing and rejects passing addresses of globals.
    id<MTLComputePipelineState> __strong* pipeline,
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
  TORCH_CHECK(
      ctx != nullptr,
      "caramba_metal_ops: expected MPS storage to provide an MTLBuffer context (got null). "
      "This usually indicates a non-standard tensor storage backend.");
  // Under ARC we must use a bridged cast from void* to ObjC object.
  return (__bridge id<MTLBuffer>)ctx;
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
  TORCH_CHECK(
      v_hd <= (int64_t)kThreadsPerThreadgroup,
      "dba_decode: v_head_dim must be <= ",
      (int)kThreadsPerThreadgroup,
      " (got v_head_dim=",
      v_hd,
      ").");

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

torch::Tensor dba_decode_null(
    at::Tensor q_sem,      // (B,H,sem_hd) fp16 MPS
    at::Tensor k_sem,      // (B,S,H*sem_hd) fp16 MPS
    at::Tensor q_geo,      // (B,H,geo_hd) fp16 MPS
    at::Tensor k_geo,      // (B,S,H*geo_hd) fp16 MPS
    at::Tensor v,          // (B,S,H*v_hd) fp16 MPS
    at::Tensor k_sem_null, // (B,H,sem_hd) fp16 MPS
    at::Tensor k_geo_null, // (B,H,geo_hd) fp16 MPS
    at::Tensor v_null,     // (B,H,v_hd) fp16 MPS
    double sem_scale,
    double geo_scale) {
  TORCH_CHECK(q_sem.device().is_mps(), "dba_decode_null: q_sem must be on MPS");
  TORCH_CHECK(q_geo.device().is_mps(), "dba_decode_null: q_geo must be on MPS");
  TORCH_CHECK(k_sem.device().is_mps(), "dba_decode_null: k_sem must be on MPS");
  TORCH_CHECK(k_geo.device().is_mps(), "dba_decode_null: k_geo must be on MPS");
  TORCH_CHECK(v.device().is_mps(), "dba_decode_null: v must be on MPS");
  TORCH_CHECK(k_sem_null.device().is_mps(), "dba_decode_null: k_sem_null must be on MPS");
  TORCH_CHECK(k_geo_null.device().is_mps(), "dba_decode_null: k_geo_null must be on MPS");
  TORCH_CHECK(v_null.device().is_mps(), "dba_decode_null: v_null must be on MPS");

  TORCH_CHECK(q_sem.dtype() == at::kHalf, "dba_decode_null: q_sem must be fp16");
  TORCH_CHECK(q_geo.dtype() == at::kHalf, "dba_decode_null: q_geo must be fp16");
  TORCH_CHECK(k_sem.dtype() == at::kHalf, "dba_decode_null: k_sem must be fp16");
  TORCH_CHECK(k_geo.dtype() == at::kHalf, "dba_decode_null: k_geo must be fp16");
  TORCH_CHECK(v.dtype() == at::kHalf, "dba_decode_null: v must be fp16");
  TORCH_CHECK(k_sem_null.dtype() == at::kHalf, "dba_decode_null: k_sem_null must be fp16");
  TORCH_CHECK(k_geo_null.dtype() == at::kHalf, "dba_decode_null: k_geo_null must be fp16");
  TORCH_CHECK(v_null.dtype() == at::kHalf, "dba_decode_null: v_null must be fp16");

  TORCH_CHECK(q_sem.is_contiguous(), "dba_decode_null: q_sem must be contiguous");
  TORCH_CHECK(q_geo.is_contiguous(), "dba_decode_null: q_geo must be contiguous");
  TORCH_CHECK(k_sem_null.is_contiguous(), "dba_decode_null: k_sem_null must be contiguous");
  TORCH_CHECK(k_geo_null.is_contiguous(), "dba_decode_null: k_geo_null must be contiguous");
  TORCH_CHECK(v_null.is_contiguous(), "dba_decode_null: v_null must be contiguous");
  TORCH_CHECK(k_sem.stride(2) == 1, "dba_decode_null: k_sem last dim must be contiguous (stride==1)");
  TORCH_CHECK(k_geo.stride(2) == 1, "dba_decode_null: k_geo last dim must be contiguous (stride==1)");
  TORCH_CHECK(v.stride(2) == 1, "dba_decode_null: v last dim must be contiguous (stride==1)");

  TORCH_CHECK(q_sem.dim() == 3, "dba_decode_null: q_sem must be (B,H,D)");
  TORCH_CHECK(q_geo.dim() == 3, "dba_decode_null: q_geo must be (B,H,D)");
  TORCH_CHECK(k_sem.dim() == 3, "dba_decode_null: k_sem must be (B,S,D)");
  TORCH_CHECK(k_geo.dim() == 3, "dba_decode_null: k_geo must be (B,S,D)");
  TORCH_CHECK(v.dim() == 3, "dba_decode_null: v must be (B,S,D)");
  TORCH_CHECK(k_sem_null.dim() == 3, "dba_decode_null: k_sem_null must be (B,H,D)");
  TORCH_CHECK(k_geo_null.dim() == 3, "dba_decode_null: k_geo_null must be (B,H,D)");
  TORCH_CHECK(v_null.dim() == 3, "dba_decode_null: v_null must be (B,H,D)");

  const int64_t B = q_sem.size(0);
  const int64_t H = q_sem.size(1);
  const int64_t sem_hd = q_sem.size(2);
  const int64_t geo_hd = q_geo.size(2);

  TORCH_CHECK(q_geo.size(0) == B && q_geo.size(1) == H, "dba_decode_null: q_geo shape mismatch");
  TORCH_CHECK(k_sem.size(0) == B, "dba_decode_null: k_sem batch mismatch");
  TORCH_CHECK(k_geo.size(0) == B, "dba_decode_null: k_geo batch mismatch");
  TORCH_CHECK(v.size(0) == B, "dba_decode_null: v batch mismatch");
  TORCH_CHECK(k_sem_null.size(0) == B && k_sem_null.size(1) == H, "dba_decode_null: k_sem_null shape mismatch");
  TORCH_CHECK(k_geo_null.size(0) == B && k_geo_null.size(1) == H, "dba_decode_null: k_geo_null shape mismatch");
  TORCH_CHECK(v_null.size(0) == B && v_null.size(1) == H, "dba_decode_null: v_null shape mismatch");

  const int64_t S = k_sem.size(1);
  TORCH_CHECK(k_geo.size(1) == S, "dba_decode_null: k_geo seq mismatch");
  TORCH_CHECK(v.size(1) == S, "dba_decode_null: v seq mismatch");

  TORCH_CHECK(k_sem.size(2) == H * sem_hd, "dba_decode_null: k_sem last dim must be H*sem_hd");
  TORCH_CHECK(k_geo.size(2) == H * geo_hd, "dba_decode_null: k_geo last dim must be H*geo_hd");
  TORCH_CHECK(v.size(2) % H == 0, "dba_decode_null: v last dim must be divisible by H");

  TORCH_CHECK(k_sem_null.size(2) == sem_hd, "dba_decode_null: k_sem_null last dim must be sem_hd");
  TORCH_CHECK(k_geo_null.size(2) == geo_hd, "dba_decode_null: k_geo_null last dim must be geo_hd");

  const int64_t v_hd = v.size(2) / H;
  TORCH_CHECK(v_null.size(2) == v_hd, "dba_decode_null: v_null last dim must be v_hd");
  TORCH_CHECK(
      v_hd <= (int64_t)kThreadsPerThreadgroup,
      "dba_decode_null: v_head_dim must be <= ",
      (int)kThreadsPerThreadgroup,
      " (got v_head_dim=",
      v_hd,
      ").");

  auto out = torch::empty({B, H, v_hd}, q_sem.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_dba_decode_null, "dba_decode_fp16_null");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "dba_decode_null: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "dba_decode_null: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "dba_decode_null: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(q_sem, 0);
  set_tensor(k_sem, 1);
  set_tensor(q_geo, 2);
  set_tensor(k_geo, 3);
  set_tensor(v, 4);
  set_tensor(k_sem_null, 5);
  set_tensor(k_geo_null, 6);
  set_tensor(v_null, 7);
  set_tensor(out, 8);

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

  [encoder setBytes:&params length:sizeof(DBAParams) atIndex:9];

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
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rmsnorm: x must be fp16 or fp32");
  TORCH_CHECK(weight.dtype() == x.dtype(), "rmsnorm: weight dtype must match x");
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
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_fp32, "rmsnorm_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm, "rmsnorm_fp16");

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
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rmsnorm_noweight: x must be fp16 or fp32");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_noweight: x must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm_noweight: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_noweight: invalid last dim");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_noweight: x.numel must be divisible by D");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_noweight_fp32, "rmsnorm_noweight_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm_noweight, "rmsnorm_noweight_fp16");

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

std::vector<torch::Tensor> rmsnorm_forward_with_inv(
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor weight, // (D,) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "rmsnorm_forward_with_inv: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "rmsnorm_forward_with_inv: weight must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rmsnorm_forward_with_inv: x must be fp16 or fp32");
  TORCH_CHECK(weight.dtype() == x.dtype(), "rmsnorm_forward_with_inv: weight dtype must match x");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_forward_with_inv: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "rmsnorm_forward_with_inv: weight must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm_forward_with_inv: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_forward_with_inv: invalid last dim");
  TORCH_CHECK(weight.numel() == D, "rmsnorm_forward_with_inv: weight must have numel == x.size(-1)");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_forward_with_inv: x.numel must be divisible by D");
  auto inv = torch::empty({rows}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_fwd_inv_fp32, "rmsnorm_fwd_inv_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm_fwd_inv, "rmsnorm_fwd_inv_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm_forward_with_inv: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm_forward_with_inv: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm_forward_with_inv: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(out, 2);
  set_tensor(inv, 3);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return {out, inv};
}

std::vector<torch::Tensor> rmsnorm_noweight_forward_with_inv(
    at::Tensor x, // (..., D) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "rmsnorm_noweight_forward_with_inv: x must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rmsnorm_noweight_forward_with_inv: x must be fp16 or fp32");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_noweight_forward_with_inv: x must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm_noweight_forward_with_inv: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_noweight_forward_with_inv: invalid last dim");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_noweight_forward_with_inv: x.numel must be divisible by D");
  auto inv = torch::empty({rows}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_noweight_fwd_inv_fp32, "rmsnorm_noweight_fwd_inv_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm_noweight_fwd_inv, "rmsnorm_noweight_fwd_inv_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm_noweight_forward_with_inv: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm_noweight_forward_with_inv: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm_noweight_forward_with_inv: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(out, 1);
  set_tensor(inv, 2);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:3];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return {out, inv};
}

torch::Tensor rmsnorm_backward_x(
    at::Tensor grad_y, // (..., D) fp16 MPS contiguous
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor weight, // (D,) fp16 MPS contiguous
    at::Tensor inv     // (rows,) fp16 MPS contiguous
) {
  TORCH_CHECK(grad_y.device().is_mps(), "rmsnorm_backward_x: grad_y must be on MPS");
  TORCH_CHECK(x.device().is_mps(), "rmsnorm_backward_x: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "rmsnorm_backward_x: weight must be on MPS");
  TORCH_CHECK(inv.device().is_mps(), "rmsnorm_backward_x: inv must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat, "rmsnorm_backward_x: grad_y must be fp16 or fp32");
  TORCH_CHECK(x.dtype() == grad_y.dtype(), "rmsnorm_backward_x: x dtype must match grad_y");
  TORCH_CHECK(weight.dtype() == x.dtype(), "rmsnorm_backward_x: weight dtype must match x");
  TORCH_CHECK(inv.dtype() == x.dtype(), "rmsnorm_backward_x: inv dtype must match x");
  TORCH_CHECK(grad_y.is_contiguous(), "rmsnorm_backward_x: grad_y must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_backward_x: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "rmsnorm_backward_x: weight must be contiguous");
  TORCH_CHECK(inv.is_contiguous(), "rmsnorm_backward_x: inv must be contiguous");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_backward_x: invalid last dim");
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_backward_x: x.numel must be divisible by D");
  TORCH_CHECK(inv.numel() == rows, "rmsnorm_backward_x: inv shape mismatch");
  TORCH_CHECK(weight.numel() == D, "rmsnorm_backward_x: weight shape mismatch");

  auto grad_x = torch::empty_like(x);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_bwd_fp32, "rmsnorm_bwd_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm_bwd, "rmsnorm_bwd_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm_backward_x: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm_backward_x: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm_backward_x: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(inv, 2);
  set_tensor(grad_y, 3);
  set_tensor(grad_x, 4);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = 0.0f;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:5];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return grad_x;
}

torch::Tensor rmsnorm_backward_x_noweight(
    at::Tensor grad_y, // (..., D) fp16 MPS contiguous
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor inv     // (rows,) fp16 MPS contiguous
) {
  TORCH_CHECK(grad_y.device().is_mps(), "rmsnorm_backward_x_noweight: grad_y must be on MPS");
  TORCH_CHECK(x.device().is_mps(), "rmsnorm_backward_x_noweight: x must be on MPS");
  TORCH_CHECK(inv.device().is_mps(), "rmsnorm_backward_x_noweight: inv must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat, "rmsnorm_backward_x_noweight: grad_y must be fp16 or fp32");
  TORCH_CHECK(x.dtype() == grad_y.dtype(), "rmsnorm_backward_x_noweight: x dtype must match grad_y");
  TORCH_CHECK(inv.dtype() == x.dtype(), "rmsnorm_backward_x_noweight: inv dtype must match x");
  TORCH_CHECK(grad_y.is_contiguous(), "rmsnorm_backward_x_noweight: grad_y must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_backward_x_noweight: x must be contiguous");
  TORCH_CHECK(inv.is_contiguous(), "rmsnorm_backward_x_noweight: inv must be contiguous");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_backward_x_noweight: invalid last dim");
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_backward_x_noweight: x.numel must be divisible by D");
  TORCH_CHECK(inv.numel() == rows, "rmsnorm_backward_x_noweight: inv shape mismatch");

  auto grad_x = torch::empty_like(x);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_bwd_noweight_fp32, "rmsnorm_bwd_noweight_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm_bwd_noweight, "rmsnorm_bwd_noweight_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm_backward_x_noweight: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm_backward_x_noweight: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm_backward_x_noweight: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(inv, 1);
  set_tensor(grad_y, 2);
  set_tensor(grad_x, 3);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = 0.0f;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return grad_x;
}

torch::Tensor rmsnorm_backward_w(
    at::Tensor grad_y, // (..., D) fp16 MPS contiguous
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor inv     // (rows,) fp16 MPS contiguous
) {
  TORCH_CHECK(grad_y.device().is_mps(), "rmsnorm_backward_w: grad_y must be on MPS");
  TORCH_CHECK(x.device().is_mps(), "rmsnorm_backward_w: x must be on MPS");
  TORCH_CHECK(inv.device().is_mps(), "rmsnorm_backward_w: inv must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat, "rmsnorm_backward_w: grad_y must be fp16 or fp32");
  TORCH_CHECK(x.dtype() == grad_y.dtype(), "rmsnorm_backward_w: x dtype must match grad_y");
  TORCH_CHECK(inv.dtype() == x.dtype(), "rmsnorm_backward_w: inv dtype must match x");
  TORCH_CHECK(grad_y.is_contiguous(), "rmsnorm_backward_w: grad_y must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_backward_w: x must be contiguous");
  TORCH_CHECK(inv.is_contiguous(), "rmsnorm_backward_w: inv must be contiguous");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_backward_w: invalid last dim");
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_backward_w: x.numel must be divisible by D");
  TORCH_CHECK(inv.numel() == rows, "rmsnorm_backward_w: inv shape mismatch");

  auto grad_w = torch::empty({D}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_gradw_fp32, "rmsnorm_gradw_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm_gradw, "rmsnorm_gradw_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm_backward_w: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm_backward_w: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm_backward_w: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(inv, 1);
  set_tensor(grad_y, 2);
  set_tensor(grad_w, 3);

  RMSNormGradWParams prm;
  prm.d_model = (uint32_t)D;
  prm.rows = (uint32_t)rows;
  prm.stride_row = (uint32_t)D;
  [encoder setBytes:&prm length:sizeof(RMSNormGradWParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)D, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return grad_w;
}

torch::Tensor layernorm(
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor weight, // (D,) fp16 MPS contiguous
    at::Tensor bias,   // (D,) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "layernorm: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "layernorm: weight must be on MPS");
  TORCH_CHECK(bias.device().is_mps(), "layernorm: bias must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "layernorm: x must be fp16 or fp32");
  TORCH_CHECK(weight.dtype() == x.dtype(), "layernorm: weight dtype match x");
  TORCH_CHECK(bias.dtype() == x.dtype(), "layernorm: bias dtype match x");
  TORCH_CHECK(x.is_contiguous(), "layernorm: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "layernorm: weight must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "layernorm: bias must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "layernorm: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "layernorm: invalid last dim");
  TORCH_CHECK(weight.numel() == D, "layernorm: weight must have numel == x.size(-1)");
  TORCH_CHECK(bias.numel() == D, "layernorm: bias must have numel == x.size(-1)");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "layernorm: x.numel must be divisible by D");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_fp32, "layernorm_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm, "layernorm_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(bias, 2);
  set_tensor(out, 3);

  LayerNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(LayerNormParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return out;
}

torch::Tensor layernorm_weight(
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor weight, // (D,) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "layernorm_weight: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "layernorm_weight: weight must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "layernorm_weight: x must be fp16 or fp32");
  TORCH_CHECK(weight.dtype() == x.dtype(), "layernorm_weight: weight dtype match x");
  TORCH_CHECK(x.is_contiguous(), "layernorm_weight: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "layernorm_weight: weight must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "layernorm_weight: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "layernorm_weight: invalid last dim");
  TORCH_CHECK(weight.numel() == D, "layernorm_weight: weight must have numel == x.size(-1)");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "layernorm_weight: x.numel must be divisible by D");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_weight_fp32, "layernorm_weight_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm_weight, "layernorm_weight_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm_weight: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm_weight: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm_weight: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(out, 2);

  LayerNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(LayerNormParams) atIndex:3];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return out;
}

torch::Tensor layernorm_noweight(
    at::Tensor x, // (..., D) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "layernorm_noweight: x must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "layernorm_noweight: x must be fp16 or fp32");
  TORCH_CHECK(x.is_contiguous(), "layernorm_noweight: x must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "layernorm_noweight: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "layernorm_noweight: invalid last dim");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "layernorm_noweight: x.numel must be divisible by D");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_noweight_fp32, "layernorm_noweight_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm_noweight, "layernorm_noweight_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm_noweight: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm_noweight: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm_noweight: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(out, 1);

  LayerNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(LayerNormParams) atIndex:2];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return out;
}

std::vector<torch::Tensor> layernorm_forward_with_stats(
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor weight, // (D,) fp16 MPS contiguous
    at::Tensor bias,   // (D,) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "layernorm_forward_with_stats: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "layernorm_forward_with_stats: weight must be on MPS");
  TORCH_CHECK(bias.device().is_mps(), "layernorm_forward_with_stats: bias must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "layernorm_forward_with_stats: x must be fp16 or fp32");
  TORCH_CHECK(weight.dtype() == x.dtype(), "layernorm_forward_with_stats: weight dtype match x");
  TORCH_CHECK(bias.dtype() == x.dtype(), "layernorm_forward_with_stats: bias dtype match x");
  TORCH_CHECK(x.is_contiguous(), "layernorm_forward_with_stats: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "layernorm_forward_with_stats: weight must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "layernorm_forward_with_stats: bias must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "layernorm_forward_with_stats: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "layernorm_forward_with_stats: invalid last dim");
  TORCH_CHECK(weight.numel() == D, "layernorm_forward_with_stats: weight must have numel == x.size(-1)");
  TORCH_CHECK(bias.numel() == D, "layernorm_forward_with_stats: bias must have numel == x.size(-1)");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "layernorm_forward_with_stats: x.numel must be divisible by D");
  auto mean = torch::empty({rows}, x.options());
  auto inv = torch::empty({rows}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_fwd_stats_fp32, "layernorm_fwd_stats_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm_fwd_stats, "layernorm_fwd_stats_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm_forward_with_stats: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm_forward_with_stats: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm_forward_with_stats: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(bias, 2);
  set_tensor(out, 3);
  set_tensor(mean, 4);
  set_tensor(inv, 5);

  LayerNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(LayerNormParams) atIndex:6];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return {out, mean, inv};
}

std::vector<torch::Tensor> layernorm_weight_forward_with_stats(
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor weight, // (D,) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "layernorm_weight_forward_with_stats: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "layernorm_weight_forward_with_stats: weight must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "layernorm_weight_forward_with_stats: x must be fp16 or fp32");
  TORCH_CHECK(weight.dtype() == x.dtype(), "layernorm_weight_forward_with_stats: weight dtype match x");
  TORCH_CHECK(x.is_contiguous(), "layernorm_weight_forward_with_stats: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "layernorm_weight_forward_with_stats: weight must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "layernorm_weight_forward_with_stats: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "layernorm_weight_forward_with_stats: invalid last dim");
  TORCH_CHECK(weight.numel() == D, "layernorm_weight_forward_with_stats: weight must have numel == x.size(-1)");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "layernorm_weight_forward_with_stats: x.numel must be divisible by D");
  auto mean = torch::empty({rows}, x.options());
  auto inv = torch::empty({rows}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_weight_fwd_stats_fp32, "layernorm_weight_fwd_stats_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm_weight_fwd_stats, "layernorm_weight_fwd_stats_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm_weight_forward_with_stats: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm_weight_forward_with_stats: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm_weight_forward_with_stats: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(out, 2);
  set_tensor(mean, 3);
  set_tensor(inv, 4);

  LayerNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(LayerNormParams) atIndex:5];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return {out, mean, inv};
}

std::vector<torch::Tensor> layernorm_noweight_forward_with_stats(
    at::Tensor x, // (..., D) fp16 MPS contiguous
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "layernorm_noweight_forward_with_stats: x must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "layernorm_noweight_forward_with_stats: x must be fp16 or fp32");
  TORCH_CHECK(x.is_contiguous(), "layernorm_noweight_forward_with_stats: x must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "layernorm_noweight_forward_with_stats: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "layernorm_noweight_forward_with_stats: invalid last dim");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "layernorm_noweight_forward_with_stats: x.numel must be divisible by D");
  auto mean = torch::empty({rows}, x.options());
  auto inv = torch::empty({rows}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_noweight_fwd_stats_fp32, "layernorm_noweight_fwd_stats_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm_noweight_fwd_stats, "layernorm_noweight_fwd_stats_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm_noweight_forward_with_stats: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm_noweight_forward_with_stats: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm_noweight_forward_with_stats: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(out, 1);
  set_tensor(mean, 2);
  set_tensor(inv, 3);

  LayerNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(LayerNormParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return {out, mean, inv};
}

torch::Tensor layernorm_backward_x(
    at::Tensor grad_y, // (..., D) fp16 MPS contiguous
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor weight, // (D,) fp16 MPS contiguous
    at::Tensor mean,   // (rows,) fp16 MPS contiguous
    at::Tensor inv     // (rows,) fp16 MPS contiguous
) {
  TORCH_CHECK(grad_y.device().is_mps(), "layernorm_backward_x: grad_y must be on MPS");
  TORCH_CHECK(x.device().is_mps(), "layernorm_backward_x: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "layernorm_backward_x: weight must be on MPS");
  TORCH_CHECK(mean.device().is_mps(), "layernorm_backward_x: mean must be on MPS");
  TORCH_CHECK(inv.device().is_mps(), "layernorm_backward_x: inv must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat, "layernorm_backward_x: grad_y must be fp16 or fp32");
  TORCH_CHECK(x.dtype() == grad_y.dtype(), "layernorm_backward_x: x dtype match grad_y");
  TORCH_CHECK(weight.dtype() == x.dtype(), "layernorm_backward_x: weight dtype match x");
  TORCH_CHECK(mean.dtype() == x.dtype(), "layernorm_backward_x: mean must dtype match x");
  TORCH_CHECK(inv.dtype() == x.dtype(), "layernorm_backward_x: inv dtype match x");
  TORCH_CHECK(grad_y.is_contiguous(), "layernorm_backward_x: grad_y must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "layernorm_backward_x: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "layernorm_backward_x: weight must be contiguous");
  TORCH_CHECK(mean.is_contiguous(), "layernorm_backward_x: mean must be contiguous");
  TORCH_CHECK(inv.is_contiguous(), "layernorm_backward_x: inv must be contiguous");
  TORCH_CHECK(x.sizes() == grad_y.sizes(), "layernorm_backward_x: x/grad_y shape mismatch");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "layernorm_backward_x: invalid last dim");
  TORCH_CHECK(weight.numel() == D, "layernorm_backward_x: weight shape mismatch");

  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "layernorm_backward_x: x.numel must be divisible by D");
  TORCH_CHECK(mean.numel() == rows, "layernorm_backward_x: mean shape mismatch");
  TORCH_CHECK(inv.numel() == rows, "layernorm_backward_x: inv shape mismatch");

  auto grad_x = torch::empty_like(x);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_bwd_x_fp32, "layernorm_bwd_x_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm_bwd_x, "layernorm_bwd_x_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm_backward_x: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm_backward_x: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm_backward_x: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(mean, 2);
  set_tensor(inv, 3);
  set_tensor(grad_y, 4);
  set_tensor(grad_x, 5);

  LayerNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = 0.0f;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(LayerNormParams) atIndex:6];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return grad_x;
}

torch::Tensor layernorm_backward_x_noweight(
    at::Tensor grad_y, // (..., D) fp16 MPS contiguous
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor mean,   // (rows,) fp16 MPS contiguous
    at::Tensor inv     // (rows,) fp16 MPS contiguous
) {
  TORCH_CHECK(grad_y.device().is_mps(), "layernorm_backward_x_noweight: grad_y must be on MPS");
  TORCH_CHECK(x.device().is_mps(), "layernorm_backward_x_noweight: x must be on MPS");
  TORCH_CHECK(mean.device().is_mps(), "layernorm_backward_x_noweight: mean must be on MPS");
  TORCH_CHECK(inv.device().is_mps(), "layernorm_backward_x_noweight: inv must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat, "layernorm_backward_x_noweight: grad_y must be fp16 or fp32");
  TORCH_CHECK(x.dtype() == grad_y.dtype(), "layernorm_backward_x_noweight: x dtype match grad_y");
  TORCH_CHECK(mean.dtype() == x.dtype(), "layernorm_backward_x_noweight: mean dtype match x");
  TORCH_CHECK(inv.dtype() == x.dtype(), "layernorm_backward_x_noweight: inv dtype match x");
  TORCH_CHECK(grad_y.is_contiguous(), "layernorm_backward_x_noweight: grad_y must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "layernorm_backward_x_noweight: x must be contiguous");
  TORCH_CHECK(mean.is_contiguous(), "layernorm_backward_x_noweight: mean must be contiguous");
  TORCH_CHECK(inv.is_contiguous(), "layernorm_backward_x_noweight: inv must be contiguous");
  TORCH_CHECK(x.sizes() == grad_y.sizes(), "layernorm_backward_x_noweight: x/grad_y shape mismatch");

  const int64_t D = x.size(-1);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(D > 0, "layernorm_backward_x_noweight: invalid last dim");
  TORCH_CHECK(rows * D == x.numel(), "layernorm_backward_x_noweight: x.numel must be divisible by D");
  TORCH_CHECK(mean.numel() == rows, "layernorm_backward_x_noweight: mean shape mismatch");
  TORCH_CHECK(inv.numel() == rows, "layernorm_backward_x_noweight: inv shape mismatch");

  auto grad_x = torch::empty_like(x);

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_bwd_x_noweight_fp32, "layernorm_bwd_x_noweight_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm_bwd_x_noweight, "layernorm_bwd_x_noweight_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm_backward_x_noweight: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm_backward_x_noweight: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm_backward_x_noweight: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(mean, 1);
  set_tensor(inv, 2);
  set_tensor(grad_y, 3);
  set_tensor(grad_x, 4);

  LayerNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = 0.0f;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(LayerNormParams) atIndex:5];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return grad_x;
}

torch::Tensor layernorm_backward_w(
    at::Tensor grad_y, // (..., D) fp16 MPS contiguous
    at::Tensor x,      // (..., D) fp16 MPS contiguous
    at::Tensor mean,   // (rows,) fp16 MPS contiguous
    at::Tensor inv     // (rows,) fp16 MPS contiguous
) {
  TORCH_CHECK(grad_y.device().is_mps(), "layernorm_backward_w: grad_y must be on MPS");
  TORCH_CHECK(x.device().is_mps(), "layernorm_backward_w: x must be on MPS");
  TORCH_CHECK(mean.device().is_mps(), "layernorm_backward_w: mean must be on MPS");
  TORCH_CHECK(inv.device().is_mps(), "layernorm_backward_w: inv must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat, "layernorm_backward_w: grad_y must be fp16 or fp32");
  TORCH_CHECK(x.dtype() == grad_y.dtype(), "layernorm_backward_w: x dtype match grad_y");
  TORCH_CHECK(mean.dtype() == x.dtype(), "layernorm_backward_w: mean dtype match x");
  TORCH_CHECK(inv.dtype() == x.dtype(), "layernorm_backward_w: inv dtype match x");
  TORCH_CHECK(grad_y.is_contiguous(), "layernorm_backward_w: grad_y must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "layernorm_backward_w: x must be contiguous");
  TORCH_CHECK(mean.is_contiguous(), "layernorm_backward_w: mean must be contiguous");
  TORCH_CHECK(inv.is_contiguous(), "layernorm_backward_w: inv must be contiguous");
  TORCH_CHECK(x.sizes() == grad_y.sizes(), "layernorm_backward_w: x/grad_y shape mismatch");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "layernorm_backward_w: invalid last dim");
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "layernorm_backward_w: x.numel must be divisible by D");
  TORCH_CHECK(mean.numel() == rows, "layernorm_backward_w: mean shape mismatch");
  TORCH_CHECK(inv.numel() == rows, "layernorm_backward_w: inv shape mismatch");

  auto grad_w = torch::empty({D}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_gradw_fp32, "layernorm_gradw_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm_gradw, "layernorm_gradw_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm_backward_w: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm_backward_w: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm_backward_w: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(mean, 1);
  set_tensor(inv, 2);
  set_tensor(grad_y, 3);
  set_tensor(grad_w, 4);

  LayerNormGradWParams prm;
  prm.d_model = (uint32_t)D;
  prm.rows = (uint32_t)rows;
  prm.stride_row = (uint32_t)D;
  [encoder setBytes:&prm length:sizeof(LayerNormGradWParams) atIndex:5];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)D, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return grad_w;
}

torch::Tensor layernorm_backward_b(
    at::Tensor grad_y // (..., D) fp16 MPS contiguous
) {
  TORCH_CHECK(grad_y.device().is_mps(), "layernorm_backward_b: grad_y must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat, "layernorm_backward_b: grad_y must be fp16 or fp32");
  TORCH_CHECK(grad_y.is_contiguous(), "layernorm_backward_b: grad_y must be contiguous");

  const int64_t D = grad_y.size(-1);
  TORCH_CHECK(D > 0, "layernorm_backward_b: invalid last dim");
  const int64_t rows = grad_y.numel() / D;
  TORCH_CHECK(rows * D == grad_y.numel(), "layernorm_backward_b: grad_y.numel must be divisible by D");

  auto grad_b = torch::empty({D}, grad_y.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (grad_y.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_layernorm_gradb_fp32, "layernorm_gradb_fp32")
      : ensure_pipeline(device, &g_pipeline_layernorm_gradb, "layernorm_gradb_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "layernorm_backward_b: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "layernorm_backward_b: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "layernorm_backward_b: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(grad_y, 0);
  set_tensor(grad_b, 1);

  LayerNormGradWParams prm;
  prm.d_model = (uint32_t)D;
  prm.rows = (uint32_t)rows;
  prm.stride_row = (uint32_t)D;
  [encoder setBytes:&prm length:sizeof(LayerNormGradWParams) atIndex:2];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)D, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return grad_b;
}

torch::Tensor rope(
    at::Tensor x,   // (B,H,T,D) fp16 MPS contiguous
    at::Tensor cos, // (T, rot/2) fp16 MPS contiguous
    at::Tensor sin, // (T, rot/2) fp16 MPS contiguous
    int64_t rot_dim) {
  TORCH_CHECK(x.device().is_mps(), "rope: x must be on MPS");
  TORCH_CHECK(cos.device().is_mps(), "rope: cos must be on MPS");
  TORCH_CHECK(sin.device().is_mps(), "rope: sin must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rope: x must be fp16 or fp32");
  TORCH_CHECK(cos.dtype() == x.dtype(), "rope: cos dtype must match x");
  TORCH_CHECK(sin.dtype() == x.dtype(), "rope: sin dtype must match x");
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
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rope_fp32, "rope_fp32")
      : ensure_pipeline(device, &g_pipeline_rope, "rope_fp16");

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

torch::Tensor rope_backward(
    at::Tensor grad_y, // (B,H,T,D) fp16 MPS contiguous
    at::Tensor cos,    // (T, rot/2) fp16 MPS contiguous
    at::Tensor sin,    // (T, rot/2) fp16 MPS contiguous
    int64_t rot_dim) {
  TORCH_CHECK(grad_y.device().is_mps(), "rope_backward: grad_y must be on MPS");
  TORCH_CHECK(cos.device().is_mps(), "rope_backward: cos must be on MPS");
  TORCH_CHECK(sin.device().is_mps(), "rope_backward: sin must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat, "rope_backward: grad_y must be fp16 or fp32");
  TORCH_CHECK(cos.dtype() == grad_y.dtype(), "rope_backward: cos dtype must match grad_y");
  TORCH_CHECK(sin.dtype() == grad_y.dtype(), "rope_backward: sin dtype must match grad_y");
  TORCH_CHECK(grad_y.is_contiguous(), "rope_backward: grad_y must be contiguous");
  TORCH_CHECK(cos.is_contiguous(), "rope_backward: cos must be contiguous");
  TORCH_CHECK(sin.is_contiguous(), "rope_backward: sin must be contiguous");

  TORCH_CHECK(grad_y.dim() == 4, "rope_backward: grad_y must be (B,H,T,D)");
  TORCH_CHECK(cos.dim() == 2, "rope_backward: cos must be (T, rot/2)");
  TORCH_CHECK(sin.dim() == 2, "rope_backward: sin must be (T, rot/2)");

  TORCH_CHECK(rot_dim > 0, "rope_backward: rot_dim must be > 0");
  TORCH_CHECK((rot_dim % 2) == 0, "rope_backward: rot_dim must be even");

  const int64_t B = grad_y.size(0);
  const int64_t H = grad_y.size(1);
  const int64_t T = grad_y.size(2);
  const int64_t D = grad_y.size(3);
  TORCH_CHECK(rot_dim <= D, "rope_backward: rot_dim must be <= head_dim");

  const int64_t half_rot = rot_dim / 2;
  TORCH_CHECK(cos.size(0) == T && cos.size(1) == half_rot, "rope_backward: cos shape mismatch");
  TORCH_CHECK(sin.size(0) == T && sin.size(1) == half_rot, "rope_backward: sin shape mismatch");

  auto grad_x = torch::empty_like(grad_y);
  const int64_t n_vec = B * H * T;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (grad_y.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rope_bwd_fp32, "rope_bwd_fp32")
      : ensure_pipeline(device, &g_pipeline_rope_bwd, "rope_bwd_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rope_backward: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rope_backward: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rope_backward: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(grad_y, 0);
  set_tensor(cos, 1);
  set_tensor(sin, 2);
  set_tensor(grad_x, 3);

  RoPEParams params;
  params.d_model = (uint32_t)D;
  params.rot_dim = (uint32_t)rot_dim;
  params.half_rot = (uint32_t)half_rot;
  params.seq_len = (uint32_t)T;
  [encoder setBytes:&params length:sizeof(RoPEParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)n_vec, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return grad_x;
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

  TORCH_CHECK(p.dtype() == at::kHalf || p.dtype() == at::kFloat, "lion_step: p must be fp16 or fp32");
  TORCH_CHECK(grad.dtype() == p.dtype(), "lion_step: grad dtype match p");
  TORCH_CHECK(m.dtype() == p.dtype(), "lion_step: m dtype match p");

  TORCH_CHECK(p.is_contiguous(), "lion_step: p must be contiguous");
  TORCH_CHECK(grad.is_contiguous(), "lion_step: grad must be contiguous");
  TORCH_CHECK(m.is_contiguous(), "lion_step: m must be contiguous");

  TORCH_CHECK(p.numel() == grad.numel(), "lion_step: p/grad numel mismatch");
  TORCH_CHECK(p.numel() == m.numel(), "lion_step: p/m numel mismatch");

  const int64_t n = p.numel();

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (p.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_lion_fp32, "lion_step_fp32")
      : ensure_pipeline(device, &g_pipeline_lion, "lion_step_fp16");

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

torch::Tensor adamw_master_step(
    at::Tensor p,          // (n,) fp16 MPS contiguous (updated)
    at::Tensor grad,       // (n,) fp16 MPS contiguous
    at::Tensor master,     // (n,) fp32 MPS contiguous (updated)
    at::Tensor exp_avg,    // (n,) fp32 MPS contiguous (updated)
    at::Tensor exp_avg_sq, // (n,) fp32 MPS contiguous (updated)
    double step_size,
    double beta1,
    double beta2,
    double eps,
    double lr_wd) {
  TORCH_CHECK(p.device().is_mps(), "adamw_master_step: p must be on MPS");
  TORCH_CHECK(grad.device().is_mps(), "adamw_master_step: grad must be on MPS");
  TORCH_CHECK(master.device().is_mps(), "adamw_master_step: master must be on MPS");
  TORCH_CHECK(exp_avg.device().is_mps(), "adamw_master_step: exp_avg must be on MPS");
  TORCH_CHECK(exp_avg_sq.device().is_mps(), "adamw_master_step: exp_avg_sq must be on MPS");

  TORCH_CHECK(p.dtype() == at::kHalf || p.dtype() == at::kFloat, "adamw_master_step: p must be fp16 or fp32");
  TORCH_CHECK(grad.dtype() == p.dtype(), "adamw_master_step: grad dtype match p");
  TORCH_CHECK(master.dtype() == at::kFloat, "adamw_master_step: master must be fp32");
  TORCH_CHECK(exp_avg.dtype() == at::kFloat, "adamw_master_step: exp_avg must be fp32");
  TORCH_CHECK(exp_avg_sq.dtype() == at::kFloat, "adamw_master_step: exp_avg_sq must be fp32");

  TORCH_CHECK(p.is_contiguous(), "adamw_master_step: p must be contiguous");
  TORCH_CHECK(grad.is_contiguous(), "adamw_master_step: grad must be contiguous");
  TORCH_CHECK(master.is_contiguous(), "adamw_master_step: master must be contiguous");
  TORCH_CHECK(exp_avg.is_contiguous(), "adamw_master_step: exp_avg must be contiguous");
  TORCH_CHECK(exp_avg_sq.is_contiguous(), "adamw_master_step: exp_avg_sq must be contiguous");

  TORCH_CHECK(p.numel() == grad.numel(), "adamw_master_step: p/grad numel mismatch");
  TORCH_CHECK(p.numel() == master.numel(), "adamw_master_step: p/master numel mismatch");
  TORCH_CHECK(p.numel() == exp_avg.numel(), "adamw_master_step: p/exp_avg numel mismatch");
  TORCH_CHECK(p.numel() == exp_avg_sq.numel(), "adamw_master_step: p/exp_avg_sq numel mismatch");

  const int64_t n = p.numel();

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (p.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_adamw_master_fp32, "adamw_master_step_fp32")
      : ensure_pipeline(device, &g_pipeline_adamw_master, "adamw_master_step_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "adamw_master_step: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "adamw_master_step: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "adamw_master_step: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(p, 0);
  set_tensor(grad, 1);
  set_tensor(master, 2);
  set_tensor(exp_avg, 3);
  set_tensor(exp_avg_sq, 4);

  AdamWParams prm;
  prm.n = (uint32_t)n;
  prm.step_size = (float)step_size;
  prm.beta1 = (float)beta1;
  prm.beta2 = (float)beta2;
  prm.eps = (float)eps;
  prm.lr_wd = (float)lr_wd;
  [encoder setBytes:&prm length:sizeof(AdamWParams) atIndex:5];

  const NSUInteger tg_n = (NSUInteger)((n + (int64_t)kThreadsPerThreadgroup - 1) / (int64_t)kThreadsPerThreadgroup);
  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake(tg_n, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return p;
}

static void check_contig_3d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
  TORCH_CHECK(t.dim() == 3, name, ": must be 3D");
  TORCH_CHECK(t.stride(2) == 1, name, ": last dim must be contiguous (stride==1)");
}

static void check_contig_2d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
  TORCH_CHECK(t.dim() == 2, name, ": must be 2D");
  TORCH_CHECK(t.stride(1) == 1, name, ": last dim must be contiguous (stride==1)");
}

static void check_contig_1d(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
  TORCH_CHECK(t.dim() == 1, name, ": must be 1D");
  TORCH_CHECK(t.stride(0) == 1, name, ": must be contiguous (stride==1)");
}

std::vector<torch::Tensor> ssm_scan_forward(
    at::Tensor x,   // (B,T,D_inner) fp16 MPS contiguous
    at::Tensor dt,  // (B,T,D_inner) fp16 MPS contiguous
    at::Tensor A,   // (D_inner,D_state) fp16 MPS contiguous
    at::Tensor Bv,  // (B,T,D_state) fp16 MPS contiguous
    at::Tensor Cv,  // (B,T,D_state) fp16 MPS contiguous
    at::Tensor D    // (D_inner,) fp16 MPS contiguous
) {
  TORCH_CHECK(x.device().is_mps(), "ssm_scan_forward: x must be on MPS");
  TORCH_CHECK(dt.device().is_mps(), "ssm_scan_forward: dt must be on MPS");
  TORCH_CHECK(A.device().is_mps(), "ssm_scan_forward: A must be on MPS");
  TORCH_CHECK(Bv.device().is_mps(), "ssm_scan_forward: B must be on MPS");
  TORCH_CHECK(Cv.device().is_mps(), "ssm_scan_forward: C must be on MPS");
  TORCH_CHECK(D.device().is_mps(), "ssm_scan_forward: D must be on MPS");

  TORCH_CHECK(x.dtype() == at::kHalf, "ssm_scan_forward: x must be fp16");
  TORCH_CHECK(dt.dtype() == at::kHalf, "ssm_scan_forward: dt must be fp16");
  TORCH_CHECK(A.dtype() == at::kHalf, "ssm_scan_forward: A must be fp16");
  TORCH_CHECK(Bv.dtype() == at::kHalf, "ssm_scan_forward: B must be fp16");
  TORCH_CHECK(Cv.dtype() == at::kHalf, "ssm_scan_forward: C must be fp16");
  TORCH_CHECK(D.dtype() == at::kHalf, "ssm_scan_forward: D must be fp16");

  check_contig_3d(x, "ssm_scan_forward: x");
  check_contig_3d(dt, "ssm_scan_forward: dt");
  check_contig_2d(A, "ssm_scan_forward: A");
  check_contig_3d(Bv, "ssm_scan_forward: B");
  check_contig_3d(Cv, "ssm_scan_forward: C");
  check_contig_1d(D, "ssm_scan_forward: D");

  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t D_inner = x.size(2);
  TORCH_CHECK(dt.size(0) == B && dt.size(1) == T && dt.size(2) == D_inner, "ssm_scan_forward: dt shape mismatch");
  TORCH_CHECK(A.size(0) == D_inner, "ssm_scan_forward: A.shape[0] must match D_inner");
  const int64_t D_state = A.size(1);
  TORCH_CHECK(D_state > 0 && D_state <= 32, "ssm_scan_forward: D_state must be in [1,32]");
  TORCH_CHECK(Bv.size(0) == B && Bv.size(1) == T && Bv.size(2) == D_state, "ssm_scan_forward: B shape mismatch");
  TORCH_CHECK(Cv.size(0) == B && Cv.size(1) == T && Cv.size(2) == D_state, "ssm_scan_forward: C shape mismatch");
  TORCH_CHECK(D.numel() == D_inner, "ssm_scan_forward: D shape mismatch");

  auto y = torch::empty({B, T, D_inner}, x.options());
  auto h_hist = torch::empty({B, T, D_inner, D_state}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline =
      ensure_pipeline(device, &g_pipeline_ssm_fwd, "ssm_scan_fwd_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "ssm_scan_forward: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "ssm_scan_forward: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "ssm_scan_forward: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(dt, 1);
  set_tensor(A, 2);
  set_tensor(Bv, 3);
  set_tensor(Cv, 4);
  set_tensor(D, 5);
  set_tensor(y, 6);
  set_tensor(h_hist, 7);

  SSMScanParams prm;
  prm.B = (uint32_t)B;
  prm.T = (uint32_t)T;
  prm.D_inner = (uint32_t)D_inner;
  prm.D_state = (uint32_t)D_state;
  [encoder setBytes:&prm length:sizeof(SSMScanParams) atIndex:8];

  const MTLSize tg = MTLSizeMake(32, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)(B * D_inner), 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];

  return {y, h_hist};
}

std::vector<torch::Tensor> ssm_scan_backward(
    at::Tensor grad_y, // (B,T,D_inner) fp16 MPS contiguous
    at::Tensor x,      // (B,T,D_inner) fp16 MPS contiguous
    at::Tensor dt,     // (B,T,D_inner) fp16 MPS contiguous
    at::Tensor A,      // (D_inner,D_state) fp16 MPS contiguous
    at::Tensor Bv,     // (B,T,D_state) fp16 MPS contiguous
    at::Tensor Cv,     // (B,T,D_state) fp16 MPS contiguous
    at::Tensor D,      // (D_inner,) fp16 MPS contiguous
    at::Tensor h_hist  // (B,T,D_inner,D_state) fp16 MPS contiguous
) {
  TORCH_CHECK(grad_y.device().is_mps(), "ssm_scan_backward: grad_y must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf, "ssm_scan_backward: grad_y must be fp16");
  check_contig_3d(grad_y, "ssm_scan_backward: grad_y");

  check_contig_3d(x, "ssm_scan_backward: x");
  check_contig_3d(dt, "ssm_scan_backward: dt");
  check_contig_2d(A, "ssm_scan_backward: A");
  check_contig_3d(Bv, "ssm_scan_backward: B");
  check_contig_3d(Cv, "ssm_scan_backward: C");
  check_contig_1d(D, "ssm_scan_backward: D");
  TORCH_CHECK(h_hist.is_contiguous(), "ssm_scan_backward: h_hist must be contiguous");
  TORCH_CHECK(h_hist.dim() == 4, "ssm_scan_backward: h_hist must be 4D");

  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t D_inner = x.size(2);
  const int64_t D_state = A.size(1);
  TORCH_CHECK(grad_y.size(0) == B && grad_y.size(1) == T && grad_y.size(2) == D_inner, "ssm_scan_backward: grad_y shape mismatch");
  TORCH_CHECK(h_hist.size(0) == B && h_hist.size(1) == T && h_hist.size(2) == D_inner && h_hist.size(3) == D_state, "ssm_scan_backward: h_hist shape mismatch");

  auto g_hist = torch::empty({B, T, D_inner, D_state}, x.options());
  auto grad_x = torch::empty({B, T, D_inner}, x.options());
  auto grad_dt = torch::empty({B, T, D_inner}, x.options());
  auto gradA_partial = torch::empty({B, D_inner, D_state}, x.options());

  auto grad_B = torch::empty({B, T, D_state}, x.options());
  auto grad_C = torch::empty({B, T, D_state}, x.options());
  auto grad_D = torch::empty({D_inner}, x.options());
  auto grad_A = torch::empty({D_inner, D_state}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "ssm_scan_backward: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "ssm_scan_backward: failed to get MTLComputeCommandEncoder from MPS stream");

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "ssm_scan_backward: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  SSMScanParams prm;
  prm.B = (uint32_t)B;
  prm.T = (uint32_t)T;
  prm.D_inner = (uint32_t)D_inner;
  prm.D_state = (uint32_t)D_state;

  // 1) Compute g_hist + grad_x + grad_dt + gradA_partial
  {
    id<MTLComputePipelineState> p0 =
        ensure_pipeline(device, &g_pipeline_ssm_bwd_g, "ssm_scan_bwd_g_fp16");
    [encoder setComputePipelineState:p0];
    set_tensor(x, 0);
    set_tensor(dt, 1);
    set_tensor(A, 2);
    set_tensor(Bv, 3);
    set_tensor(Cv, 4);
    set_tensor(D, 5);
    set_tensor(grad_y, 6);
    set_tensor(h_hist, 7);
    set_tensor(g_hist, 8);
    set_tensor(grad_x, 9);
    set_tensor(grad_dt, 10);
    set_tensor(gradA_partial, 11);
    [encoder setBytes:&prm length:sizeof(SSMScanParams) atIndex:12];

    const MTLSize tg = MTLSizeMake(32, 1, 1);
    const MTLSize grid = MTLSizeMake((NSUInteger)(B * D_inner), 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  }

  // 2) grad_B reduce over D_inner
  {
    id<MTLComputePipelineState> p1 =
        ensure_pipeline(device, &g_pipeline_ssm_gradB, "ssm_gradB_reduce_fp16");
    [encoder setComputePipelineState:p1];
    set_tensor(x, 0);
    set_tensor(dt, 1);
    set_tensor(g_hist, 2);
    set_tensor(grad_B, 3);
    [encoder setBytes:&prm length:sizeof(SSMScanParams) atIndex:4];

    const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    const MTLSize grid = MTLSizeMake((NSUInteger)(B * T * D_state), 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  }

  // 3) grad_C reduce over D_inner
  {
    id<MTLComputePipelineState> p2 =
        ensure_pipeline(device, &g_pipeline_ssm_gradC, "ssm_gradC_reduce_fp16");
    [encoder setComputePipelineState:p2];
    set_tensor(grad_y, 0);
    set_tensor(h_hist, 1);
    set_tensor(grad_C, 2);
    [encoder setBytes:&prm length:sizeof(SSMScanParams) atIndex:3];

    const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    const MTLSize grid = MTLSizeMake((NSUInteger)(B * T * D_state), 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  }

  // 4) grad_D reduce over (B,T)
  {
    id<MTLComputePipelineState> p3 =
        ensure_pipeline(device, &g_pipeline_ssm_gradD, "ssm_gradD_reduce_fp16");
    [encoder setComputePipelineState:p3];
    set_tensor(x, 0);
    set_tensor(grad_y, 1);
    set_tensor(grad_D, 2);
    [encoder setBytes:&prm length:sizeof(SSMScanParams) atIndex:3];

    const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    const MTLSize grid = MTLSizeMake((NSUInteger)(D_inner), 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  }

  // 5) grad_A reduce over batch
  {
    id<MTLComputePipelineState> p4 =
        ensure_pipeline(device, &g_pipeline_ssm_gradA, "ssm_gradA_reduce_fp16");
    [encoder setComputePipelineState:p4];
    set_tensor(gradA_partial, 0);
    set_tensor(grad_A, 1);
    [encoder setBytes:&prm length:sizeof(SSMScanParams) atIndex:2];

    const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
    const MTLSize grid = MTLSizeMake((NSUInteger)(D_inner * D_state), 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  }

  return {grad_x, grad_dt, grad_A, grad_B, grad_C, grad_D};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dba_decode", &dba_decode, "DBA Decode Forward (Metal/MPS, fp16)");
  m.def("dba_decode_null", &dba_decode_null, "DBA Decode Forward with null KV (Metal/MPS, fp16)");
  m.def("rmsnorm", &rmsnorm, "RMSNorm Forward (Metal/MPS, fp16)");
  m.def("rmsnorm_noweight", &rmsnorm_noweight, "RMSNorm Forward (no weight, Metal/MPS, fp16)");
  m.def("rmsnorm_forward_with_inv", &rmsnorm_forward_with_inv, "RMSNorm Forward with inv cache (Metal/MPS, fp16)");
  m.def("rmsnorm_noweight_forward_with_inv", &rmsnorm_noweight_forward_with_inv, "RMSNorm Forward (no weight) with inv cache (Metal/MPS, fp16)");
  m.def("rmsnorm_backward_x", &rmsnorm_backward_x, "RMSNorm Backward grad_x (Metal/MPS, fp16)");
  m.def("rmsnorm_backward_x_noweight", &rmsnorm_backward_x_noweight, "RMSNorm Backward grad_x (no weight, Metal/MPS, fp16)");
  m.def("rmsnorm_backward_w", &rmsnorm_backward_w, "RMSNorm Backward grad_w (Metal/MPS, fp16)");
  m.def("layernorm", &layernorm, "LayerNorm Forward (Metal/MPS, fp16)");
  m.def("layernorm_weight", &layernorm_weight, "LayerNorm Forward (weight only, Metal/MPS, fp16)");
  m.def("layernorm_noweight", &layernorm_noweight, "LayerNorm Forward (no weight/bias, Metal/MPS, fp16)");
  m.def("layernorm_forward_with_stats", &layernorm_forward_with_stats, "LayerNorm Forward with (mean,inv) cache (Metal/MPS, fp16)");
  m.def("layernorm_weight_forward_with_stats", &layernorm_weight_forward_with_stats, "LayerNorm Forward (weight only) with (mean,inv) cache (Metal/MPS, fp16)");
  m.def("layernorm_noweight_forward_with_stats", &layernorm_noweight_forward_with_stats, "LayerNorm Forward (no weight) with (mean,inv) cache (Metal/MPS, fp16)");
  m.def("layernorm_backward_x", &layernorm_backward_x, "LayerNorm Backward grad_x (Metal/MPS, fp16)");
  m.def("layernorm_backward_x_noweight", &layernorm_backward_x_noweight, "LayerNorm Backward grad_x (no weight, Metal/MPS, fp16)");
  m.def("layernorm_backward_w", &layernorm_backward_w, "LayerNorm Backward grad_w (Metal/MPS, fp16)");
  m.def("layernorm_backward_b", &layernorm_backward_b, "LayerNorm Backward grad_b (Metal/MPS, fp16)");
  m.def("rope", &rope, "RoPE Apply (Metal/MPS, fp16)");
  m.def("rope_backward", &rope_backward, "RoPE Backward (Metal/MPS, fp16)");
  m.def("lion_step", &lion_step, "Lion step update (Metal/MPS, fp16)");
  m.def("adamw_master_step", &adamw_master_step, "AdamW master step update (Metal/MPS, fp16 params + fp32 state)");
  m.def("ssm_scan_forward", &ssm_scan_forward, "SSM selective scan forward (Metal/MPS, fp16)");
  m.def("ssm_scan_backward", &ssm_scan_backward, "SSM selective scan backward (Metal/MPS, fp16)");
}
