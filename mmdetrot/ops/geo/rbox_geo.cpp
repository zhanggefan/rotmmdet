#include "rbox_geo_utils.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

py::array_t<double, py::array::c_style>
rboxes_iou(const py::array_t<double, py::array::c_style> &rboxes_1_,
           const py::array_t<double, py::array::c_style> &rboxes_2_, bool iof) {

  auto rboxes_1 = rboxes_1_.unchecked<2>();
  auto rboxes_2 = rboxes_2_.unchecked<2>();

  int N = rboxes_1.shape(0);
  int M = rboxes_2.shape(0);

  auto iou_mat_ = py::array_t<double, py::array::c_style>({N, M});
  auto iou_mat = iou_mat_.mutable_unchecked<2>();

  const int mode_flag = iof ? 1 : 0;
  for (int i = 0; i < N; i++) {
    auto rbox_1 = rboxes_1.data(i, 0);
    for (int j = 0; j < M; j++) {
      auto rbox_2 = rboxes_2.data(j, 0);
      iou_mat(i, j) = geo::rotated_boxes_iou<double>(rbox_1, rbox_2, mode_flag);
    }
  }
  return iou_mat_;
}

std::vector<std::vector<py::array_t<double, py::array::c_style>>>
rboxes_intersection(const py::array_t<double, py::array::c_style> &rboxes_1_,
                    const py::array_t<double, py::array::c_style> &rboxes_2_) {

  auto rboxes_1 = rboxes_1_.unchecked<2>();
  auto rboxes_2 = rboxes_2_.unchecked<2>();

  int N = rboxes_1.shape(0);
  int M = rboxes_2.shape(0);

  std::vector<std::vector<py::array_t<double, py::array::c_style>>>
      intersection;
  intersection.resize(N);

  for (int i = 0; i < N; i++) {
    intersection[i].resize(M);
    auto rbox_1 = rboxes_1.data(i, 0);
    for (int j = 0; j < M; j++) {
      auto rbox_2 = rboxes_2.data(j, 0);
      geo::Point<double> intersect_pts[24];
      int num_pts = geo::rotated_boxes_intersection<double>(rbox_1, rbox_2,
                                                            intersect_pts);
      auto intersect_pts_mat_ =
          py::array_t<double, py::array::c_style>({num_pts, 2});
      auto intersect_pts_mat = intersect_pts_mat_.mutable_unchecked<2>();
      for (int k = 0; k < num_pts; k++) {
        intersect_pts_mat(k, 0) = intersect_pts[k].x;
        intersect_pts_mat(k, 1) = intersect_pts[k].y;
      }
      intersection[i].push_back(intersect_pts_mat_);
    }
  }
  return intersection;
}

py::array_t<double, py::array::c_style>
rboxes_truncate(const py::array_t<double, py::array::c_style> &rboxes_,
                const py::array_t<double, py::array::c_style> &rlimit_) {
  auto rboxes = rboxes_.unchecked<2>();
  auto rlimit = rlimit_.unchecked<1>();

  int N = rboxes.shape(0);

  auto truncated_ = py::array_t<double, py::array::c_style>({N, 5});
  auto truncated = truncated_.mutable_unchecked<2>();

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < 5; j++) {
      truncated(i, j) = rboxes(i, j);
    }
    geo::rotated_boxes_truncate((double *)truncated.data(i, 0), rlimit.data(0));
  }
  return truncated_;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rboxes_iou", &rboxes_iou, py::arg("rboxes_1"), py::arg("rboxes_2"),
        py::arg("iof") = false)
      .def("rboxes_truncate", &rboxes_truncate, py::arg("rboxes"),
           py::arg("rlimit"))
      .def("rboxes_intersection", &rboxes_intersection, py::arg("rboxes_1"),
           py::arg("rboxes_2"));
}
