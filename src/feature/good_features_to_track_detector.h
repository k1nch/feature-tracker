//!
//! \file good_features_to_track_detector.h
//! \brief
//!
//! \author Yu Jin
//! \version
//! \date Mar 18, 2020
//!

#ifndef _GOOD_FEATURES_TO_TRACK_DETECTOR_H_
#define _GOOD_FEATURES_TO_TRACK_DETECTOR_H_

#include <memory>
#include "feature_common.h"
#include "opencv2/opencv.hpp"
#if HAVE_NVX
#include <NVX/nvxcu.h>
#include <OVX/UtilityOVX.hpp>
#endif

class GoodFeaturesToTrackDetector {
 public:
  static std::shared_ptr<GoodFeaturesToTrackDetector> create(
      const FeatureParameters& params);

  virtual void detect(cv::InputArray image, cv::OutputArray corners,
                      cv::InputArray mask = cv::noArray()) = 0;

 protected:
  GoodFeaturesToTrackDetector(const FeatureParameters& params)
      : params_{params} {}

 protected:
  FeatureParameters params_;
};

class GoodFeaturesToTrackDetectorCv : public GoodFeaturesToTrackDetector {
 public:
  GoodFeaturesToTrackDetectorCv(const FeatureParameters& params)
      : GoodFeaturesToTrackDetector{params} {}

  void detect(cv::InputArray image, cv::OutputArray corners,
              cv::InputArray mask = cv::noArray());
};

#ifdef HAVE_OPENCV_CUDAIMGPROC
class GoodFeaturesToTrackDetectorCvCuda : public GoodFeaturesToTrackDetector {
 public:
  GoodFeaturesToTrackDetectorCvCuda(const FeatureParameters& params);

  void detect(cv::InputArray image, cv::OutputArray corners,
              cv::InputArray mask = cv::noArray());

 private:
  cv::Ptr<cv::cuda::CornersDetector> ptr_;
  cv::cuda::GpuMat image_d_;
  cv::cuda::GpuMat corners_d_;
  cv::cuda::GpuMat mask_d_;
};
#endif

#if HAVE_NVX
class GoodFeaturesToTrackDetectorNvx : public GoodFeaturesToTrackDetector {
 public:
  GoodFeaturesToTrackDetectorNvx(const FeatureParameters& params);
  ~GoodFeaturesToTrackDetectorNvx();

  void detect(cv::InputArray image, cv::OutputArray corners,
              cv::InputArray mask = cv::noArray());

 private:
  ovxio::ContextGuard context_;
  vx_array pts_ = nullptr;
};
#endif

#endif  // _GOOD_FEATURES_TO_TRACK_DETECTOR_H_
