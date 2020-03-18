
//!
//! \file sparse_pyr_lk_optical_flow.h
//! \brief
//!
//! \author Yu Jin
//! \version
//! \date Mar 18, 2020
//!

#ifndef _SPARSE_PYR_LK_OPTICAL_FLOW_H_
#define _SPARSE_PYR_LK_OPTICAL_FLOW_H_

#include <memory>
#include "feature_common.h"
#include "opencv2/opencv.hpp"
#if HAVE_NVX
#include <NVX/nvxcu.h>
#include <OVX/UtilityOVX.hpp>
#endif

class SparsePyrLKOpticalFlow {
 public:
  static std::shared_ptr<SparsePyrLKOpticalFlow> create(
      const FeatureParameters& params);

  virtual void calc(cv::InputArray prev_img, cv::InputArray next_img,
                    cv::InputArray prev_pts, cv::InputOutputArray next_pts,
                    cv::OutputArray status, cv::OutputArray err) = 0;

 protected:
  SparsePyrLKOpticalFlow(const FeatureParameters& params) : params_{params} {}

 protected:
  FeatureParameters params_;
};

class SparsePyrLKOpticalFlowCv : public SparsePyrLKOpticalFlow {
 public:
  SparsePyrLKOpticalFlowCv(const FeatureParameters& params)
      : SparsePyrLKOpticalFlow{params} {}

  void calc(cv::InputArray prev_img, cv::InputArray next_img,
            cv::InputArray prev_pts, cv::InputOutputArray next_pts,
            cv::OutputArray status, cv::OutputArray err);
};

#ifdef HAVE_OPENCV_CUDAOPTFLOW
class SparsePyrLKOpticalFlowCvCuda : public SparsePyrLKOpticalFlow {
 public:
  SparsePyrLKOpticalFlowCvCuda(const FeatureParameters& params);

  void calc(cv::InputArray prev_img, cv::InputArray next_img,
            cv::InputArray prev_pts, cv::InputOutputArray next_pts,
            cv::OutputArray status, cv::OutputArray err);

 private:
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> ptr_;
  cv::cuda::GpuMat prev_img_d_;
  cv::cuda::GpuMat next_img_d_;
  cv::cuda::GpuMat prev_pts_d_;
  cv::cuda::GpuMat next_pts_d_;
  cv::cuda::GpuMat status_d_;
  cv::cuda::GpuMat err_d_;
};
#endif

#if HAVE_NVX
class SparsePyrLKOpticalFlowNvx : public SparsePyrLKOpticalFlow {
 public:
  SparsePyrLKOpticalFlowNvx(const FeatureParameters& params);
  ~SparsePyrLKOpticalFlowNvx();

  void calc(cv::InputArray prev_img, cv::InputArray next_img,
            cv::InputArray prev_pts, cv::InputOutputArray next_pts,
            cv::OutputArray status, cv::OutputArray err);

 private:
  ovxio::ContextGuard context_;
  vx_array prev_pts_ = nullptr;
  vx_array next_pts_ = nullptr;
  vx_array new_points_estimates_ = nullptr;
  vx_pyramid prev_pyr_ = nullptr;
  vx_pyramid next_pyr_ = nullptr;
};
#endif

#endif  // _SPARSE_PYR_LK_OPTICAL_FLOW_H_
