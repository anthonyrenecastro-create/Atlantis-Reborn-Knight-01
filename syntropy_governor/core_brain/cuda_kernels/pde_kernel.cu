#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
  return max(lo, min(v, hi));
}

__device__ __forceinline__ float idx4(const float* x, int n, int c, int h, int w, int C, int H, int W) {
  return x[((n * C + c) * H + h) * W + w];
}

__device__ __forceinline__ float laplacian(const float* x, int n, int c, int h, int w, int C, int H, int W) {
  const int hm = clampi(h - 1, 0, H - 1);
  const int hp = clampi(h + 1, 0, H - 1);
  const int wm = clampi(w - 1, 0, W - 1);
  const int wp = clampi(w + 1, 0, W - 1);

  float center = idx4(x, n, c, h, w, C, H, W);
  float up = idx4(x, n, c, hm, w, C, H, W);
  float down = idx4(x, n, c, hp, w, C, H, W);
  float left = idx4(x, n, c, h, wm, C, H, W);
  float right = idx4(x, n, c, h, wp, C, H, W);

  return up + down + left + right - 4.0f * center;
}

// 13-point stencil approximation of biharmonic operator in 2D for unit spacing.
__device__ __forceinline__ float biharmonic13(const float* x, int n, int c, int h, int w, int C, int H, int W) {
  int hm2 = clampi(h - 2, 0, H - 1);
  int hm1 = clampi(h - 1, 0, H - 1);
  int hp1 = clampi(h + 1, 0, H - 1);
  int hp2 = clampi(h + 2, 0, H - 1);

  int wm2 = clampi(w - 2, 0, W - 1);
  int wm1 = clampi(w - 1, 0, W - 1);
  int wp1 = clampi(w + 1, 0, W - 1);
  int wp2 = clampi(w + 2, 0, W - 1);

  float center = idx4(x, n, c, h, w, C, H, W);

  float axial2 = idx4(x, n, c, hm2, w, C, H, W) + idx4(x, n, c, hp2, w, C, H, W) +
                 idx4(x, n, c, h, wm2, C, H, W) + idx4(x, n, c, h, wp2, C, H, W);

  float diagonal = idx4(x, n, c, hm1, wm1, C, H, W) + idx4(x, n, c, hm1, wp1, C, H, W) +
                   idx4(x, n, c, hp1, wm1, C, H, W) + idx4(x, n, c, hp1, wp1, C, H, W);

  float axial1 = idx4(x, n, c, hm1, w, C, H, W) + idx4(x, n, c, hp1, w, C, H, W) +
                 idx4(x, n, c, h, wm1, C, H, W) + idx4(x, n, c, h, wp1, C, H, W);

  return axial2 + 2.0f * diagonal - 8.0f * axial1 + 20.0f * center;
}

__global__ void pde_step_kernel(
    const float* phi1,
    const float* phi5,
    const float* Phi,
    float* out_phi1,
    float* out_phi5,
    int N,
    int C,
    int H,
    int W,
    float alpha,
    float zeta,
    float gamma1,
    float beta,
    float omega,
    float theta,
    float gamma5,
    float dt) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * H * W;
  if (idx >= total) {
    return;
  }

  int w = idx % W;
  int t = idx / W;
  int h = t % H;
  t = t / H;
  int c = t % C;
  int n = t / C;

  float p1 = idx4(phi1, n, c, h, w, C, H, W);
  float p5 = idx4(phi5, n, c, h, w, C, H, W);
  float P = idx4(Phi, n, c, h, w, C, H, W);

  float lap_phi1 = laplacian(phi1, n, c, h, w, C, H, W);
  float bi_lap_phi1 = biharmonic13(phi1, n, c, h, w, C, H, W);
  float lap_phi5 = laplacian(phi5, n, c, h, w, C, H, W);
  float lap_Phi = laplacian(Phi, n, c, h, w, C, H, W);

  float reaction = -alpha * p1 * p1 * p1 + zeta * lap_Phi;
  float diffusion = -gamma1 * lap_phi1 + beta * bi_lap_phi1;
  float d_phi1 = diffusion + reaction;

  float eps = 1e-6f;
  float log_term = omega * p5 * logf(fabsf(p5) + eps);
  float advection = theta * (lap_phi5 * P + p5 * lap_Phi);
  float d_phi5 = -gamma5 * lap_phi5 + log_term + advection;

  float next_phi1 = p1 + d_phi1 * dt;
  float next_phi5 = p5 + d_phi5 * dt;

  // Keep same stability bounds as Python fallback.
  next_phi1 = fminf(5.0f, fmaxf(-5.0f, next_phi1));
  next_phi5 = fminf(10.0f, fmaxf(-10.0f, next_phi5));

  out_phi1[idx] = next_phi1;
  out_phi5[idx] = next_phi5;
}

} // namespace

std::vector<torch::Tensor> pde_step_cuda(
    torch::Tensor phi1,
    torch::Tensor phi5,
    torch::Tensor Phi,
    float alpha,
    float zeta,
    float gamma1,
    float beta,
    float omega,
    float theta,
    float gamma5,
    float dt) {

  auto out_phi1 = torch::zeros_like(phi1);
  auto out_phi5 = torch::zeros_like(phi5);

  const int N = phi1.size(0);
  const int C = phi1.size(1);
  const int H = phi1.size(2);
  const int W = phi1.size(3);

  const int total = N * C * H * W;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  pde_step_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      phi1.data_ptr<float>(),
      phi5.data_ptr<float>(),
      Phi.data_ptr<float>(),
      out_phi1.data_ptr<float>(),
      out_phi5.data_ptr<float>(),
      N,
      C,
      H,
      W,
      alpha,
      zeta,
      gamma1,
      beta,
      omega,
      theta,
      gamma5,
      dt);

  return {out_phi1, out_phi5};
}
