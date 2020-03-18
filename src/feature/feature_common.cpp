//!
//! \file feature_common.cpp
//! \brief
//!
//! \author Yu Jin
//! \version
//! \date Mar 18, 2020
//!

#include "feature_common.h"
#include "opencv2/opencv.hpp"

template <typename _Tp>
_Tp _read(const cv::FileStorage& fs, const std::string& key,
          const _Tp& default_value) {
  if (!fs.isOpened()) {
    return default_value;
  }
  auto node = fs[key];
  if (node.type() == cv::FileNode::NONE) {
    return default_value;
  }
  _Tp x;
  fs[key] >> x;
  return x;
}

FeatureParameters FeatureParameters::from_yaml(const std::string& filename) {
  FeatureParameters params;
  cv::FileStorage fs{filename, cv::FileStorage::READ};
  std::string type = _read(fs, "max_corners", std::string{""});
  if (type == "cv_cuda") {
    params.type = Type::CV_CUDA;
  } else if (type == "nvx") {
    params.type = Type::NVX;
  } else {
    params.type = Type::CV;
  }
  params.max_corners = _read(fs, "max_corners", 500);
  params.height = _read(fs, "height", 720);
  params.width = _read(fs, "width", 1280);
  params.quality_level = _read(fs, "quality_level", 0.01);
  params.min_distance = _read(fs, "min_distance", 30.0);
  params.block_size = _read(fs, "block_size", 3);
  params.use_harris_detector = _read(fs, "use_harris_detector", false);
  params.k = _read(fs, "k", 0.04);
  params.fast_type = _read(fs, "fast_type", 10);
  params.fast_threshold = _read(fs, "fast_threshold", 25.0);
  params.harris_threshold = _read(fs, "harris_threshold", 2000.0);
  params.cell_size = _read(fs, "cell_size", 30);
  params.win_size = _read(fs, "win_size", 21);
  params.max_level = _read(fs, "max_level", 3);
  params.num_iters = _read(fs, "num_iters", 30);
  params.epsilon = _read(fs, "epsilon", 0.01);
  params.use_initial_flow = _read(fs, "use_initial_flow", false);
  return params;
}