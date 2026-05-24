#include <torch/extension.h>
#include <vector>

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
    float dt);

std::vector<torch::Tensor> pde_step(
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
  TORCH_CHECK(phi1.is_cuda(), "phi1 must be CUDA tensor");
  TORCH_CHECK(phi5.is_cuda(), "phi5 must be CUDA tensor");
  TORCH_CHECK(Phi.is_cuda(), "Phi must be CUDA tensor");
  TORCH_CHECK(phi1.scalar_type() == torch::kFloat32, "phi1 must be float32");
  TORCH_CHECK(phi5.scalar_type() == torch::kFloat32, "phi5 must be float32");
  TORCH_CHECK(Phi.scalar_type() == torch::kFloat32, "Phi must be float32");
  TORCH_CHECK(phi1.is_contiguous(), "phi1 must be contiguous");
  TORCH_CHECK(phi5.is_contiguous(), "phi5 must be contiguous");
  TORCH_CHECK(Phi.is_contiguous(), "Phi must be contiguous");
  TORCH_CHECK(phi1.sizes() == phi5.sizes(), "phi1 and phi5 shape mismatch");
  TORCH_CHECK(phi1.sizes() == Phi.sizes(), "phi1 and Phi shape mismatch");
  TORCH_CHECK(phi1.dim() == 4, "expected shape (N,C,H,W)");

  return pde_step_cuda(phi1, phi5, Phi, alpha, zeta, gamma1, beta, omega, theta, gamma5, dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pde_step", &pde_step, "Field PDE step (CUDA)");
}
