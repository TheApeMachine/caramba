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

struct ResonantUpdateParams {
  uint32_t n;
  uint32_t D;
  uint32_t H;
  float inv_D;
  float scale;
  float damping;
  uint32_t zero_diag;
};

constexpr NSUInteger kThreadsPerThreadgroup = 256;

static id<MTLLibrary> g_lib = nil;
static id<MTLComputePipelineState> g_fwd_fp32 = nil;
static id<MTLComputePipelineState> g_bwd_fp32 = nil;
static std::mutex g_mutex;

static std::string metallib_path_for_this_module() {
  Dl_info info;
  if (dladdr((void*)&metallib_path_for_this_module, &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  fs::path so_path(info.dli_fname);
  fs::path lib_path = so_path.parent_path() / "caramba_resonant_ops.metallib";
  return lib_path.string();
}

static void ensure_library_locked(id<MTLDevice> device) {
  if (g_lib != nil) {
    return;
  }
  const std::string lib_path = metallib_path_for_this_module();
  TORCH_CHECK(!lib_path.empty(), "caramba_metal_resonant_ops: failed to locate extension path via dladdr()");
  NSString* ns_path = [NSString stringWithUTF8String:lib_path.c_str()];
  NSURL* url = [NSURL fileURLWithPath:ns_path];
  NSError* err = nil;
  g_lib = [device newLibraryWithURL:url error:&err];
  if (g_lib == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_resonant_ops: failed to load metallib at ", lib_path, ": ", msg);
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
  TORCH_CHECK(fn != nil, "caramba_metal_resonant_ops: function `", fn_name, "` not found in metallib");
  NSError* err = nil;
  *pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
  if (*pipeline == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "caramba_metal_resonant_ops: failed to create compute pipeline: ", msg);
  }
  TORCH_CHECK(
      (*pipeline).maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup,
      "caramba_metal_resonant_ops: pipeline maxTotalThreadsPerThreadgroup (",
      (int)(*pipeline).maxTotalThreadsPerThreadgroup,
      ") < expected threads (",
      (int)kThreadsPerThreadgroup,
      ")");
  return *pipeline;
}

static inline id<MTLBuffer> storage_as_mtlbuffer(const at::Tensor& t) {
  const auto& dp = t.storage().data_ptr();
  void* ctx = dp.get_context();
  TORCH_CHECK(ctx != nullptr, "caramba_metal_resonant_ops: expected MPS storage to provide an MTLBuffer context");
  return (__bridge id<MTLBuffer>)ctx;
}

static inline id<MTLCommandBuffer> command_buffer() {
  auto* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream, "caramba_metal_resonant_ops: no current MPS stream available");
  return stream->commandBuffer();
}

static inline id<MTLDevice> mtl_device() {
  // Match the rest of caramba's Metal extensions: use MPSDevice singleton.
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  TORCH_CHECK(device != nil, "caramba_metal_resonant_ops: no default MPS device available");
  return device;
}

static void check_fp32_mps(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, ": tensor must be on MPS");
  TORCH_CHECK(t.dtype() == at::kFloat, name, ": tensor must be fp32");
  TORCH_CHECK(t.is_contiguous(), name, ": tensor must be contiguous");
}

static at::Tensor resonant_update_forward_fp32(
    at::Tensor x,
    at::Tensor y,
    at::Tensor vr,
    at::Tensor vi,
    at::Tensor diag,
    double scale,
    double damping,
    bool zero_diag) {
  check_fp32_mps(x, "x");
  check_fp32_mps(y, "y");
  check_fp32_mps(vr, "vr");
  check_fp32_mps(vi, "vi");
  check_fp32_mps(diag, "diag");
  TORCH_CHECK(x.sizes() == y.sizes() && x.sizes() == vr.sizes() && x.sizes() == vi.sizes(), "x/y/vr/vi shapes must match");
  TORCH_CHECK(x.dim() == 3, "x must be (BT,H,D)");
  const auto BT = (uint32_t)x.size(0);
  const auto H = (uint32_t)x.size(1);
  const auto D = (uint32_t)x.size(2);
  TORCH_CHECK(diag.dim() == 2, "diag must be (H,D)");
  TORCH_CHECK((uint32_t)diag.size(0) == H && (uint32_t)diag.size(1) == D, "diag shape mismatch");

  const uint32_t n = BT * H * D;
  auto xo = at::empty_like(x);
  auto yo = at::empty_like(y);
  auto a = at::empty_like(x);
  auto b = at::empty_like(y);
  auto inv_r = at::empty_like(x);

  id<MTLDevice> device = mtl_device();
  id<MTLComputePipelineState> pipe = ensure_pipeline(device, &g_fwd_fp32, "resonant_update_fwd_fp32");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "caramba_metal_resonant_ops: failed to get current MPS stream");
  // Ensure no prior encoder is left open before requesting a compute encoder.
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(enc != nil, "caramba_metal_resonant_ops: failed to get MTLComputeCommandEncoder from MPS stream");
  [enc setComputePipelineState:pipe];

  [enc setBuffer:storage_as_mtlbuffer(x) offset:x.storage_offset() * sizeof(float) atIndex:0];
  [enc setBuffer:storage_as_mtlbuffer(y) offset:y.storage_offset() * sizeof(float) atIndex:1];
  [enc setBuffer:storage_as_mtlbuffer(vr) offset:vr.storage_offset() * sizeof(float) atIndex:2];
  [enc setBuffer:storage_as_mtlbuffer(vi) offset:vi.storage_offset() * sizeof(float) atIndex:3];
  [enc setBuffer:storage_as_mtlbuffer(diag) offset:diag.storage_offset() * sizeof(float) atIndex:4];
  [enc setBuffer:storage_as_mtlbuffer(xo) offset:xo.storage_offset() * sizeof(float) atIndex:5];
  [enc setBuffer:storage_as_mtlbuffer(yo) offset:yo.storage_offset() * sizeof(float) atIndex:6];
  [enc setBuffer:storage_as_mtlbuffer(a) offset:a.storage_offset() * sizeof(float) atIndex:7];
  [enc setBuffer:storage_as_mtlbuffer(b) offset:b.storage_offset() * sizeof(float) atIndex:8];
  [enc setBuffer:storage_as_mtlbuffer(inv_r) offset:inv_r.storage_offset() * sizeof(float) atIndex:9];

  ResonantUpdateParams p;
  p.n = n;
  p.D = D;
  p.H = H;
  p.inv_D = (float)(1.0 / (double)D);
  p.scale = (float)scale;
  p.damping = (float)damping;
  p.zero_diag = zero_diag ? 1u : 0u;
  [enc setBytes:&p length:sizeof(ResonantUpdateParams) atIndex:10];

  const NSUInteger tg = kThreadsPerThreadgroup;
  const NSUInteger grid = ((NSUInteger)n + tg - 1) / tg;
  [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  // Ask MPSStream to end the current coalesced encoder safely.
  // (Do NOT call [enc endEncoding] directly; MPSStream owns encoder lifetime.)
  stream->endKernelCoalescing();
  return at::stack({xo, yo, a, b, inv_r}, 0);
}

static std::vector<at::Tensor> resonant_update_backward_fp32(
    at::Tensor grad_xo,
    at::Tensor grad_yo,
    at::Tensor x,
    at::Tensor y,
    at::Tensor diag,
    at::Tensor a,
    at::Tensor b,
    at::Tensor inv_r,
    double scale,
    double damping,
    bool zero_diag) {
  check_fp32_mps(grad_xo, "grad_xo");
  check_fp32_mps(grad_yo, "grad_yo");
  check_fp32_mps(x, "x");
  check_fp32_mps(y, "y");
  check_fp32_mps(diag, "diag");
  check_fp32_mps(a, "a");
  check_fp32_mps(b, "b");
  check_fp32_mps(inv_r, "inv_r");

  TORCH_CHECK(x.sizes() == grad_xo.sizes() && x.sizes() == grad_yo.sizes(), "grad shapes must match x");
  TORCH_CHECK(x.dim() == 3, "x must be (BT,H,D)");
  const auto BT = (uint32_t)x.size(0);
  const auto H = (uint32_t)x.size(1);
  const auto D = (uint32_t)x.size(2);
  const uint32_t n = BT * H * D;

  auto gvr = at::empty_like(x);
  auto gvi = at::empty_like(x);
  auto gx = at::empty_like(x);
  auto gy = at::empty_like(x);

  id<MTLDevice> device = mtl_device();
  id<MTLComputePipelineState> pipe = ensure_pipeline(device, &g_bwd_fp32, "resonant_update_bwd_fp32");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "caramba_metal_resonant_ops: failed to get current MPS stream");
  // Ensure no prior encoder is left open before requesting a compute encoder.
  stream->endKernelCoalescing();
  id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(enc != nil, "caramba_metal_resonant_ops: failed to get MTLComputeCommandEncoder from MPS stream");
  [enc setComputePipelineState:pipe];

  [enc setBuffer:storage_as_mtlbuffer(grad_xo) offset:grad_xo.storage_offset() * sizeof(float) atIndex:0];
  [enc setBuffer:storage_as_mtlbuffer(grad_yo) offset:grad_yo.storage_offset() * sizeof(float) atIndex:1];
  [enc setBuffer:storage_as_mtlbuffer(x) offset:x.storage_offset() * sizeof(float) atIndex:2];
  [enc setBuffer:storage_as_mtlbuffer(y) offset:y.storage_offset() * sizeof(float) atIndex:3];
  [enc setBuffer:storage_as_mtlbuffer(diag) offset:diag.storage_offset() * sizeof(float) atIndex:4];
  [enc setBuffer:storage_as_mtlbuffer(a) offset:a.storage_offset() * sizeof(float) atIndex:5];
  [enc setBuffer:storage_as_mtlbuffer(b) offset:b.storage_offset() * sizeof(float) atIndex:6];
  [enc setBuffer:storage_as_mtlbuffer(inv_r) offset:inv_r.storage_offset() * sizeof(float) atIndex:7];
  [enc setBuffer:storage_as_mtlbuffer(gvr) offset:gvr.storage_offset() * sizeof(float) atIndex:8];
  [enc setBuffer:storage_as_mtlbuffer(gvi) offset:gvi.storage_offset() * sizeof(float) atIndex:9];
  [enc setBuffer:storage_as_mtlbuffer(gx) offset:gx.storage_offset() * sizeof(float) atIndex:10];
  [enc setBuffer:storage_as_mtlbuffer(gy) offset:gy.storage_offset() * sizeof(float) atIndex:11];

  ResonantUpdateParams p;
  p.n = n;
  p.D = D;
  p.H = H;
  p.inv_D = (float)(1.0 / (double)D);
  p.scale = (float)scale;
  p.damping = (float)damping;
  p.zero_diag = zero_diag ? 1u : 0u;
  [enc setBytes:&p length:sizeof(ResonantUpdateParams) atIndex:12];

  const NSUInteger tg = kThreadsPerThreadgroup;
  const NSUInteger grid = ((NSUInteger)n + tg - 1) / tg;
  [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  // Ask MPSStream to end the current coalesced encoder safely.
  // (Do NOT call [enc endEncoding] directly; MPSStream owns encoder lifetime.)
  stream->endKernelCoalescing();

  return {gx, gy, gvr, gvi};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("resonant_update_forward_fp32", &resonant_update_forward_fp32, "Resonant update forward (fp32, Metal/MPS)");
  m.def("resonant_update_backward_fp32", &resonant_update_backward_fp32, "Resonant update backward (fp32, Metal/MPS)");
}

