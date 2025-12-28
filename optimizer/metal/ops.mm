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

constexpr NSUInteger kThreadsPerThreadgroup = 256;

static id<MTLComputePipelineState> g_pipeline = nil;
static std::mutex g_pipeline_mutex;

static std::string metallib_path_for_this_module() {
  Dl_info info;
  if (dladdr((void*)&metallib_path_for_this_module, &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  fs::path so_path(info.dli_fname);
  fs::path lib_path = so_path.parent_path() / "dba_decode.metallib";
  return lib_path.string();
}

static void ensure_pipeline(id<MTLDevice> device) {
  std::lock_guard<std::mutex> lock(g_pipeline_mutex);
  if (g_pipeline != nil) {
    return;
  }

  const std::string lib_path = metallib_path_for_this_module();
  TORCH_CHECK(!lib_path.empty(), "caramba_metal_ops: failed to locate extension path via dladdr()");

  NSString* ns_path = [NSString stringWithUTF8String:lib_path.c_str()];
  NSError* err = nil;
  id<MTLLibrary> lib = [device newLibraryWithFile:ns_path error:&err];
  if (lib == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_ops: failed to load metallib at ", lib_path, ": ", msg);
  }

  id<MTLFunction> fn = [lib newFunctionWithName:@"dba_decode_fp16"];
  TORCH_CHECK(fn != nil, "caramba_metal_ops: function `dba_decode_fp16` not found in metallib: ", lib_path);

  g_pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
  if (g_pipeline == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_ops: failed to create compute pipeline: ", msg);
  }

  // Basic sanity check against accidental dispatch mismatch.
  TORCH_CHECK(
      g_pipeline.maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup,
      "caramba_metal_ops: pipeline maxTotalThreadsPerThreadgroup (",
      (int)g_pipeline.maxTotalThreadsPerThreadgroup,
      ") < expected threads (",
      (int)kThreadsPerThreadgroup,
      ")");
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
  ensure_pipeline(device);

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "dba_decode: failed to get current MPS stream");

  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "dba_decode: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:g_pipeline];

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

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dba_decode", &dba_decode, "DBA Decode Forward (Metal/MPS, fp16)");
}

