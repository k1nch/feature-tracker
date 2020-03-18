//!
//! \file sparse_pyr_lk_optical_flow.cpp
//! \brief
//!
//! \author Yu Jin
//! \version
//! \date Mar 18, 2020
//!

#include "sparse_pyr_lk_optical_flow.h"
#if HAVE_NVX
#include <NVX/Utility.hpp>
#include <NVX/nvx_opencv_interop.hpp>
#endif

std::shared_ptr<SparsePyrLKOpticalFlow> SparsePyrLKOpticalFlow::create(
    const FeatureParameters& params) {
  switch (params.type) {
    case FeatureParameters::CV:
      return std::make_shared<SparsePyrLKOpticalFlowCv>(params);
#ifdef HAVE_OPENCV_CUDAOPTFLOW
    case FeatureParameters::CV_CUDA:
      return std::make_shared<SparsePyrLKOpticalFlowCvCuda>(params);
#endif
#if HAVE_NVX
    case FeatureParameters::NVX:
      return std::make_shared<SparsePyrLKOpticalFlowNvx>(params);
#endif
    default:
      break;
  }
  return nullptr;
}

void SparsePyrLKOpticalFlowCv::calc(cv::InputArray prev_img,
                                    cv::InputArray next_img,
                                    cv::InputArray prev_pts,
                                    cv::InputOutputArray next_pts,
                                    cv::OutputArray status,
                                    cv::OutputArray err) {
  cv::Mat prev_gray, next_gray;
  if (prev_img.channels() == 1) {
    prev_gray = prev_img.getMat();
  } else {
    cv::cvtColor(prev_img.getMat(), prev_gray, CV_BGR2GRAY);
  }
  if (next_img.channels() == 1) {
    next_gray = next_img.getMat();
  } else {
    cv::cvtColor(next_img.getMat(), next_gray, CV_BGR2GRAY);
  }
  cv::calcOpticalFlowPyrLK(
      prev_gray, next_gray, prev_pts, next_pts, status, err,
      cv::Size{params_.win_size, params_.win_size}, params_.max_level,
      cv::TermCriteria{cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                       params_.num_iters, params_.epsilon},
      params_.use_initial_flow, 1e-4);
}

#ifdef HAVE_OPENCV_CUDAOPTFLOW
SparsePyrLKOpticalFlowCvCuda::SparsePyrLKOpticalFlowCvCuda(
    const FeatureParameters& params)
    : SparsePyrLKOpticalFlow{params} {
  ptr_ = cv::cuda::SparsePyrLKOpticalFlow::create(
      cv::Size{params_.win_size, params_.win_size}, params_.max_level,
      params_.num_iters, params_.use_initial_flow);
}

void SparsePyrLKOpticalFlowCvCuda::calc(cv::InputArray prev_img,
                                        cv::InputArray next_img,
                                        cv::InputArray prev_pts,
                                        cv::InputOutputArray next_pts,
                                        cv::OutputArray status,
                                        cv::OutputArray err) {
  prev_img_d_.upload(prev_img);
  next_img_d_.upload(next_img);
  cv::cuda::GpuMat prev_gray;
  if (prev_img.channels() == 1) {
    prev_gray = prev_img_d_;
  } else {
    cv::cuda::cvtColor(prev_img_d_, prev_gray, CV_BGR2GRAY);
  }
  cv::cuda::GpuMat next_gray;
  if (next_img.channels() == 1) {
    next_gray = next_img_d_;
  } else {
    cv::cuda::cvtColor(next_img_d_, next_gray, CV_BGR2GRAY);
  }
  prev_pts_d_.upload(prev_pts);
  if (params_.use_initial_flow) {
    next_pts_d_.upload(next_pts);
  }
  prev_pts_d_ = prev_pts_d_.reshape(2, 1);
  if (next_pts_d_.cols < prev_pts_d_.cols) {
    auto sz = prev_pts_d_.size();
    next_pts_d_.create(sz, CV_32FC2);
    status_d_.create(sz, CV_8UC1);
    if (!err.empty()) {
      err_d_.create(sz, CV_32FC1);
    }
  }

  ptr_->calc(prev_gray, next_gray, prev_pts_d_, next_pts_d_, status_d_, err_d_,
             cv::cuda::Stream::Null());

  next_pts_d_.download(next_pts);
  status_d_.download(status);
  if (!err.empty()) {
    err_d_.download(err);
  }
}
#endif

#if HAVE_NVX
SparsePyrLKOpticalFlowNvx::SparsePyrLKOpticalFlowNvx(
    const FeatureParameters& params)
    : SparsePyrLKOpticalFlow{params} {
  prev_pts_ = vxCreateArray(context_, NVX_TYPE_POINT2F, params_.max_corners);
  next_pts_ = vxCreateArray(context_, NVX_TYPE_POINT2F, params_.max_corners);
  new_points_estimates_ =
      vxCreateArray(context_, NVX_TYPE_POINT2F, params_.max_corners);
  prev_pyr_ =
      vxCreatePyramid(context_, params_.max_level, VX_SCALE_PYRAMID_HALF,
                      params_.width, params_.height, VX_DF_IMAGE_U8);
  next_pyr_ =
      vxCreatePyramid(context_, params_.max_level, VX_SCALE_PYRAMID_HALF,
                      params_.width, params_.height, VX_DF_IMAGE_U8);
}

SparsePyrLKOpticalFlowNvx::~SparsePyrLKOpticalFlowNvx() {
  vxReleaseArray(&prev_pts_);
  prev_pts_ = nullptr;
  vxReleaseArray(&next_pts_);
  next_pts_ = nullptr;
  vxReleaseArray(&new_points_estimates_);
  new_points_estimates_ = nullptr;
  vxReleasePyramid(&prev_pyr_);
  prev_pyr_ = nullptr;
  vxReleasePyramid(&next_pyr_);
  next_pyr_ = nullptr;
}

void SparsePyrLKOpticalFlowNvx::calc(cv::InputArray prev_img,
                                     cv::InputArray next_img,
                                     cv::InputArray prev_pts,
                                     cv::InputOutputArray next_pts,
                                     cv::OutputArray status,
                                     cv::OutputArray err) {
  vx_image prev_img_vx =
      nvx_cv::createVXImageFromCVMat(context_, prev_img.getMat());
  vx_image next_img_vx =
      nvx_cv::createVXImageFromCVMat(context_, next_img.getMat());
  vx_image prev_vx_gray;
  if (prev_img.channels() == 1) {
    prev_vx_gray = prev_img_vx;
  } else {
    prev_vx_gray =
        vxCreateImage(context_, params_.width, params_.height, VX_DF_IMAGE_U8);
    vxuColorConvert(context_, prev_img_vx, prev_vx_gray);
  }
  vx_image next_vx_gray;
  if (next_img.channels() == 1) {
    next_vx_gray = next_img_vx;
  } else {
    next_vx_gray =
        vxCreateImage(context_, params_.width, params_.height, VX_DF_IMAGE_U8);
    vxuColorConvert(context_, next_img_vx, next_vx_gray);
  }

  auto fill_input = [](vx_array pts, std::vector<cv::Point2f>* vec) {
    vxTruncateArray(pts, 0);
    vxAddArrayItems(pts, vec->size(), vec->data(), sizeof(nvx_point2f_t));
  };
  {
    std::vector<cv::Point2f>* ptr =
        (std::vector<cv::Point2f>*)prev_pts.getObj();
    fill_input(prev_pts_, ptr);
  }
  if (params_.use_initial_flow) {
    std::vector<cv::Point2f>* ptr =
        (std::vector<cv::Point2f>*)next_pts.getObj();
    fill_input(new_points_estimates_, ptr);
  }

  vxuGaussianPyramid(context_, prev_vx_gray, prev_pyr_);
  vxuGaussianPyramid(context_, next_vx_gray, next_pyr_);

  vx_float32 epsilon = params_.epsilon;
  vx_scalar epsilon_vx = vxCreateScalar(context_, VX_TYPE_FLOAT32, &epsilon);

  vx_uint32 num_iters = params_.num_iters;
  vx_scalar num_iters_vx = vxCreateScalar(context_, VX_TYPE_UINT32, &num_iters);

  vx_bool use_init_est = params_.use_initial_flow ? vx_true_e : vx_false_e;
  vx_scalar use_init_est_vx =
      vxCreateScalar(context_, VX_TYPE_BOOL, &use_init_est);

  vxuOpticalFlowPyrLK(context_, prev_pyr_, next_pyr_, prev_pts_,
                      new_points_estimates_, next_pts_, VX_TERM_CRITERIA_BOTH,
                      epsilon_vx, num_iters_vx, use_init_est_vx,
                      params_.win_size);

  auto fill_output = [](vx_array vx, std::vector<cv::Point2f>* vec) {
    vx_size num_items = 0;
    vxQueryArray(vx, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items,
                 sizeof(num_items));
    vx_size stride = sizeof(vx_size);
    void* base = nullptr;
    vxAccessArrayRange(vx, 0, num_items, &stride, &base, VX_READ_ONLY);

    vec->clear();
    for (vx_size i = 0; i < num_items; i++) {
      nvx_point2f_t* p = (nvx_point2f_t*)base;
      vec->emplace_back(cv::Point2f{p[i].x, p[i].y});
    }

    vxCommitArrayRange(vx, 0, num_items, &base);
  };

  auto result = (std::vector<cv::Point2f>*)next_pts.getObj();
  fill_output(next_pts_, result);

  auto status_ptr = (std::vector<uint8_t>*)status.getObj();
  status_ptr->resize(result->size());
  for (int i = 0; i < result->size(); ++i) {
    auto& p = (*result)[i];
    (*status_ptr)[i] =
        p.x >= 0 && p.x < params_.width && p.y >= 0 && p.y < params_.height;
  }

  vxReleaseImage(&prev_img_vx);
  vxReleaseImage(&next_img_vx);
  vxReleaseScalar(&epsilon_vx);
  vxReleaseScalar(&num_iters_vx);
  vxReleaseScalar(&use_init_est_vx);
  if (prev_img.channels() != 1) {
    vxReleaseImage(&prev_vx_gray);
  }
  if (next_img.channels() != 1) {
    vxReleaseImage(&next_vx_gray);
  }
}
#endif