//!
//! \file feature_common.h
//! \brief
//!
//! \author Yu Jin
//! \version
//! \date Mar 18, 2020
//!

#ifndef _FEATURE_COMMON_H_
#define _FEATURE_COMMON_H_

#include <string>

struct FeatureParameters {
  enum Type { CV = 0, CV_CUDA = 1, NVX = 2 };

  // common
  Type type = Type::CV;
  int max_corners = 500;
  int height = 720;
  int width = 1280;

  // corner detect
  double quality_level = 0.01;
  double min_distance = 30.0;
  int block_size = 3;
  bool use_harris_detector = false;
  double k = 0.04;

  int fast_type = 10;  // 9, 10, 11, 12
  double fast_threshold = 25.0;
  double harris_threshold = 2000.0;
  int cell_size = 30;

  // optical flow
  int win_size = 21;
  int max_level = 3;
  int num_iters = 30;
  double epsilon = 0.01;
  bool use_initial_flow = false;

  static FeatureParameters from_yaml(const std::string& filename);
};

#endif  // _FEATURE_COMMON_H_
