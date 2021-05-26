#include <torch/extension.h>
namespace gwd {

at::Tensor matsqrt2x2sym_fwd(const at::Tensor &in);
at::Tensor matsqrt2x2sym_bwd(const at::Tensor &out, const at::Tensor &dout);

using namespace pybind11::literals;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matsqrt2x2sym_fwd", &matsqrt2x2sym_fwd, "in"_a,
        "2x2 symmetrical matrix sqrt forward");
  m.def("matsqrt2x2sym_bwd", &matsqrt2x2sym_bwd, "out"_a, "dout"_a,
        "2x2 symmetrical matrix sqrt backward");
}
} // namespace gwd