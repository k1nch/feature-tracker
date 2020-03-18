//!
//! \file good_features_to_track_detector.cpp
//! \brief
//!
//! \author Yu Jin
//! \version
//! \date Mar 18, 2020
//!

#include "good_features_to_track_detector.h"
#if HAVE_NVX
#include <NVX/Utility.hpp>
#include <NVX/nvx_opencv_interop.hpp>
#endif

std::shared_ptr<GoodFeaturesToTrackDetector>
GoodFeaturesToTrackDetector::create(const FeatureParameters& params) {
  switch (params.type) {
    case FeatureParameters::CV:
      return std::make_shared<GoodFeaturesToTrackDetectorCv>(params);
#ifdef HAVE_OPENCV_CUDAIMGPROC
    case FeatureParameters::CV_CUDA:
      return std::make_shared<GoodFeaturesToTrackDetectorCvCuda>(params);
#endif
#if HAVE_NVX
    case FeatureParameters::NVX:
      return std::make_shared<GoodFeaturesToTrackDetectorNvx>(params);
#endif
    default:
      break;
  }
  return nullptr;
}

void GoodFeaturesToTrackDetectorCv::detect(cv::InputArray image,
                                           cv::OutputArray corners,
                                           cv::InputArray mask) {
  cv::Mat image_gray;
  if (image.channels() == 1) {
    image_gray = image.getMat();
  } else {
    cv::cvtColor(image.getMat(), image_gray, CV_BGR2GRAY);
  }
  cv::goodFeaturesToTrack(image_gray, corners, params_.max_corners,
                          params_.quality_level, params_.min_distance, mask,
                          params_.block_size, params_.use_harris_detector,
                          params_.k);
}

#ifdef HAVE_OPENCV_CUDAIMGPROC
GoodFeaturesToTrackDetectorCvCuda::GoodFeaturesToTrackDetectorCvCuda(
    const FeatureParameters& params)
    : GoodFeaturesToTrackDetector{params} {
  ptr_ = cv::cuda::createGoodFeaturesToTrackDetector(
      CV_8UC1, params_.max_corners, params_.quality_level, params_.min_distance,
      params_.block_size, params_.use_harris_detector, params_.k);
}

void GoodFeaturesToTrackDetectorCvCuda::detect(cv::InputArray image,
                                               cv::OutputArray corners,
                                               cv::InputArray mask) {
  image_d_.upload(image);
  cv::cuda::GpuMat image_gray;
  if (image.channels() == 1) {
    image_gray = image_d_;
  } else {
    cv::cuda::cvtColor(image_d_, image_gray, CV_BGR2GRAY);
  }
  mask_d_.upload(mask);
  ptr_->detect(image_gray, corners_d_, mask_d_, cv::cuda::Stream::Null());
  corners_d_.download(corners);
}
#endif

#if HAVE_NVX
GoodFeaturesToTrackDetectorNvx::GoodFeaturesToTrackDetectorNvx(
    const FeatureParameters& params)
    : GoodFeaturesToTrackDetector{params} {
  pts_ = vxCreateArray(context_, NVX_TYPE_POINT2F, params_.max_corners);
}

GoodFeaturesToTrackDetectorNvx::~GoodFeaturesToTrackDetectorNvx() {
  vxReleaseArray(&pts_);
  pts_ = nullptr;
}

void GoodFeaturesToTrackDetectorNvx::detect(cv::InputArray image,
                                            cv::OutputArray corners,
                                            cv::InputArray mask) {
  cv::Mat image_mat = image.getMat();
  vx_image image_vx = nvx_cv::createVXImageFromCVMat(context_, image.getMat());
  vx_image mask_vx =
      mask.empty() ? nullptr
                   : nvx_cv::createVXImageFromCVMat(context_, mask.getMat());
  vx_image image_vx_gray;
  if (image.channels() == 1) {
    image_vx_gray = image_vx;
  } else {
    image_vx_gray =
        vxCreateImage(context_, params_.width, params_.height, VX_DF_IMAGE_U8);
    vxuColorConvert(context_, image_vx, image_vx_gray);
  }

  if (params_.use_harris_detector) {
    NVXIO_SAFE_CALL(nvxuHarrisTrack(
        context_, image_vx_gray, pts_, mask_vx, nullptr, params_.k,
        params_.harris_threshold, params_.cell_size, nullptr));
  } else {
    NVXIO_SAFE_CALL(nvxuFastTrack(
        context_, image_vx_gray, pts_, mask_vx, nullptr, params_.fast_type,
        params_.fast_threshold, params_.cell_size, nullptr));
  }

  auto fill = [&](vx_array vx, std::vector<cv::Point2f>* vec) {
    vx_size num_items = 0;
    vxQueryArray(vx, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items,
                 sizeof(num_items));
    vx_size stride = sizeof(vx_size);
    void* base = NULL;
    vxAccessArrayRange(vx, 0, num_items, &stride, &base, VX_READ_ONLY);

    vec->clear();
    for (vx_size i = 0; i < num_items; i++) {
      nvx_point2f_t* p = (nvx_point2f_t*)base;
      vec->emplace_back(cv::Point2f{p[i].x, p[i].y});
    }

    vxCommitArrayRange(vx, 0, num_items, &base);
  };

  std::vector<cv::Point2f>* result =
      (std::vector<cv::Point2f>*)corners.getObj();
  fill(pts_, result);

  vxReleaseImage(&image_vx);
  vxReleaseImage(&mask_vx);
  if (image.channels() != 1) {
    vxReleaseImage(&image_vx_gray);
  }
}
#endif
