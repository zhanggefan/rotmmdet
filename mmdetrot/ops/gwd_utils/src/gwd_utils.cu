#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

namespace {
int const threadsPerBlock = 512;
int const maxGridDim = 50000;
} // namespace

namespace gwd {
template <typename T>
__host__ __device__ __forceinline__ void eig2x2sym(const T *in, T *eigval,
                                                   T *eigmat) {
  T x00 = in[0], x0110 = in[1], x11 = in[2];
  T b = -x00 - x11, c = x00 * x11 - x0110 * x0110;
  T r0 = (-b + sqrt(b * b - 4 * c)) / (T)2;
  T r1 = (-b - sqrt(b * b - 4 * c)) / (T)2;
  eigval[0] = r0;
  eigval[1] = r1;
  T v00 = -x0110, v10 = x00 - r0;
  T n = sqrt(v00 * v00 + v10 * v10);
  if (n == (T)0.0) {
    v00 = x11 - r0, v10 = -x0110;
    n = sqrt(v00 * v00 + v10 * v10);
  }
  if (n == (T)0.0) {
    v00 = (T)1.0, v10 = (T)0.0;
    n = (T)1.0;
  }
  eigmat[0] = v00 / n;
  eigmat[2] = v10 / n;
  eigmat[1] = -v10 / n;
  eigmat[3] = v00 / n;
}

template <typename T>
__global__ void matsqrt2x2sym_fwd_kernel(const T *in, T *out, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gridDim.x * blockDim.x) {
    T eigval[2];
    T eigmat[4];
    eig2x2sym(in + i * 3, eigval, eigmat);
    T r0_h = sqrt(eigval[0]), r1_h = sqrt(eigval[1]);
    out += i * 3;
    out[0] = eigmat[0] * eigmat[0] * r0_h + eigmat[1] * eigmat[1] * r1_h;
    out[1] = eigmat[0] * eigmat[2] * r0_h + eigmat[1] * eigmat[3] * r1_h;
    out[2] = eigmat[2] * eigmat[2] * r0_h + eigmat[3] * eigmat[3] * r1_h;
  }
}

template <typename T>
__host__ __device__ __forceinline__ void gauss_jordan(T *mat, T *res, int N) {
  for (int i = 0; i < N; i++) {
    T best = (T)0.0;
    int best_ind = -1;
    for (int j = i; j < N; j++) {
      if (abs(mat[j * (N + 1) + i]) != (T)0.0)
        if (best_ind < 0 || abs(abs(mat[j * (N + 1) + i]) - (T)1.0) < best) {
          best_ind = j;
          best = abs(abs(mat[j * (N + 1) + i]) - (T)1.0);
        }
    }
    assert(best_ind >= 0);
    for (int k = 0; k < N + 1; k++) {
      T t = mat[i * (N + 1) + k];
      mat[i * (N + 1) + k] = mat[best_ind * (N + 1) + k];
      mat[best_ind * (N + 1) + k] = t;
    }
    for (int j = i + 1; j < N; j++) {
      T t = mat[j * (N + 1) + i] / mat[i * (N + 1) + i];
      for (int k = i; k < N + 1; k++)
        mat[j * (N + 1) + k] -= mat[i * (N + 1) + k] * t;
    }
  }
  for (int i = N - 1; i >= 0; i--) {
    for (int j = i + 1; j < N; j++)
      mat[i * (N + 1) + N] -= mat[i * (N + 1) + j] * res[j];
    res[i] = mat[i * (N + 1) + N] / mat[i * (N + 1) + i];
  }
}

template <typename T>
__global__ void matsqrt2x2sym_bwd_kernel(const T *out, const T *dout, T *din,
                                         const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += gridDim.x * blockDim.x) {
    /**
     *      / 2*y00    2*y0110  0     \ T -1  / dy00 \
     *      | y0110   y00+y11   y0110 |     * |dy0110|
     *      \ 0       2*y0110   2*y11 /       \ dy11 /
     */
    out += i * 3;
    dout += i * 3;
    din += i * 3;
    const T y00 = out[0], y0110 = out[1], y11 = out[2];
    const T dy00 = dout[0], dy0110 = dout[1], dy11 = dout[2];
    T buf3x4[3][4] = {{2 * y00, y0110, 0, dy00},
                      {2 * y0110, y00 + y11, 2 * y0110, dy0110},
                      {0, y0110, 2 * y11, dy11}};
    gauss_jordan(&buf3x4[0][0], din, 3);
  }
}

at::Tensor matsqrt2x2sym_fwd_cuda(const at::Tensor &in) {
  CHECK_INPUT(in);
  int N = in.numel() / 3;
  at::Tensor out = at::empty_like(in);

  AT_DISPATCH_FLOATING_TYPES(
      in.scalar_type(), "matsqrt2x2sym_fwd_cuda", ([&] {
        dim3 blocks(
            std::min(at::cuda::ATenCeilDiv(N, threadsPerBlock), maxGridDim));
        dim3 threads(threadsPerBlock);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        matsqrt2x2sym_fwd_kernel<<<blocks, threads, 0, stream>>>(
            in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), N);
      }));
  return out;
}

at::Tensor matsqrt2x2sym_bwd_cuda(const at::Tensor &out,
                                  const at::Tensor &dout) {
  CHECK_INPUT(out);
  int N = out.numel() / 3;
  at::Tensor din = at::empty_like(dout);

  AT_DISPATCH_FLOATING_TYPES(
      dout.scalar_type(), "matsqrt2x2sym_bwd_cuda", ([&] {
        dim3 blocks(
            std::min(at::cuda::ATenCeilDiv(N, threadsPerBlock), maxGridDim));
        dim3 threads(threadsPerBlock);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        matsqrt2x2sym_bwd_kernel<<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), dout.data_ptr<scalar_t>(),
            din.data_ptr<scalar_t>(), N);
      }));
  return din;
}

at::Tensor matsqrt2x2sym_fwd_cpu(const at::Tensor &in) {
  CHECK_CONTIGUOUS(in);
  int N = in.numel() / 3;
  at::Tensor out = at::empty_like(in);

  AT_DISPATCH_FLOATING_TYPES(
      in.scalar_type(), "matsqrt2x2_fwd_cpu", ([&] {
        for (int i = 0; i < N; i++) {
          auto in_off = in.data_ptr<scalar_t>() + i * 3;
          auto out_off = out.data_ptr<scalar_t>() + i * 3;
          scalar_t eigval[2], eigmat[4];
          eig2x2sym(in_off, eigval, eigmat);
          scalar_t r0_h = sqrt(eigval[0]), r1_h = sqrt(eigval[1]);
          out_off[0] =
              eigmat[0] * eigmat[0] * r0_h + eigmat[1] * eigmat[1] * r1_h;
          out_off[1] =
              eigmat[0] * eigmat[2] * r0_h + eigmat[1] * eigmat[3] * r1_h;
          out_off[2] =
              eigmat[2] * eigmat[2] * r0_h + eigmat[3] * eigmat[3] * r1_h;
        }
      }));
  return out;
}

at::Tensor matsqrt2x2sym_bwd_cpu(const at::Tensor &out,
                                 const at::Tensor &dout) {
  CHECK_CONTIGUOUS(out);
  int N = out.numel() / 3;
  at::Tensor din = at::empty_like(dout);

  AT_DISPATCH_FLOATING_TYPES(
      dout.scalar_type(), "matsqrt2x2_bwd_cpu", ([&] {
        for (int i = 0; i < N; i++) {
          auto out_off = out.data_ptr<scalar_t>() + i * 3;
          auto dout_off = dout.data_ptr<scalar_t>() + i * 3;
          auto din_off = din.data_ptr<scalar_t>() + i * 3;

          const scalar_t y00 = out_off[0], y0110 = out_off[1], y11 = out_off[2];
          const scalar_t dy00 = dout_off[0], dy0110 = dout_off[1],
                         dy11 = dout_off[2];
          scalar_t buf3x4[3][4] = {{2 * y00, y0110, 0, dy00},
                                   {2 * y0110, y00 + y11, 2 * y0110, dy0110},
                                   {0, y0110, 2 * y11, dy11}};
          gauss_jordan(&buf3x4[0][0], din_off, 3);
        }
      }));
  return din;
}

at::Tensor matsqrt2x2sym_fwd(const at::Tensor &in) {
  if (!in.is_contiguous())
    in.contiguous();
  if (in.is_cuda())
    return matsqrt2x2sym_fwd_cuda(in);
  else
    return matsqrt2x2sym_fwd_cpu(in);
}

at::Tensor matsqrt2x2sym_bwd(const at::Tensor &out, const at::Tensor &dout) {
  if (!out.is_contiguous())
    out.contiguous();
  if (!dout.is_contiguous())
    dout.contiguous();
  if (out.is_cuda())
    return matsqrt2x2sym_bwd_cuda(out, dout);
  else
    return matsqrt2x2sym_bwd_cpu(out, dout);
}

} // namespace gwd