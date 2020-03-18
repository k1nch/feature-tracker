//!
//! \file main.cpp
//! \brief
//!
//! \author Yu Jin
//! \version
//! \date Mar 18, 2020
//!

#include <chrono>
#include <iostream>
#include <memory>
#include "feature/good_features_to_track_detector.h"
#include "feature/sparse_pyr_lk_optical_flow.h"
#include "opencv2/opencv.hpp"

class Timer {
  template <typename _D1, typename _D2>
  class _duration_divide {
    using rd = std::ratio_divide<typename _D1::period, typename _D2::period>;

   public:
    static constexpr double value =
        static_cast<double>(rd::num) / static_cast<double>(rd::den);
  };

 public:
  Timer(const std::string& msg = "time") : msg_(msg) {
    tp_ = std::chrono::steady_clock::now();
  }
  ~Timer() {
    std::cout << msg_ << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - tp_)
                         .count() *
                     _duration_divide<std::chrono::microseconds,
                                      std::chrono::seconds>::value
              << " seconds" << std::endl;
  }

 private:
  std::string msg_;
  std::chrono::steady_clock::time_point tp_;
};

int main(int argc, char* argv[]) {
  cv::VideoCapture capture{argv[1]};
  cv::Mat img_bgr, img_prev, img_next;
  std::vector<std::vector<cv::Point2f>> pt_prev, pt_prev_back, pt_next;
  std::vector<std::shared_ptr<GoodFeaturesToTrackDetector>> detector;
  std::vector<std::shared_ptr<SparsePyrLKOpticalFlow>> tracker, tracker_back;
  std::vector<std::vector<uint8_t>> status, status_back;
  std::vector<std::vector<float>> err;

  FeatureParameters params =
      argc > 2 ? FeatureParameters::from_yaml(argv[2]) : FeatureParameters{};
  capture.read(img_bgr);
  cv::Size sz{params.width, params.height};
  cv::resize(img_bgr, img_bgr, sz);
  cv::cvtColor(img_bgr, img_prev, CV_BGR2GRAY);

  cv::Mat mask = cv::Mat::ones(sz, CV_8UC1);

  for (int i = 0; i < 3; ++i) {
    params.type = (FeatureParameters::Type)i;
    detector.emplace_back(GoodFeaturesToTrackDetector::create(params));
    params.use_initial_flow = false;
    tracker.emplace_back(SparsePyrLKOpticalFlow::create(params));
    params.use_initial_flow = true;
    tracker_back.emplace_back(SparsePyrLKOpticalFlow::create(params));
  }
  pt_prev.resize(detector.size());
  pt_prev_back.resize(detector.size());
  pt_next.resize(detector.size());
  status.resize(detector.size());
  status_back.resize(detector.size());
  err.resize(detector.size());

  cv::namedWindow("optical flow", CV_WINDOW_NORMAL);
  while (capture.read(img_bgr)) {
    cv::resize(img_bgr, img_bgr, sz);
    img_next = img_bgr.clone();
    // cv::cvtColor(img_bgr, img_next, CV_BGR2GRAY);

    for (int i = 0; i < detector.size(); ++i) {
      if (!detector[i] || !tracker[i]) {
        continue;
      }
      Timer _{std::to_string(i)};
      detector[i]->detect(img_prev, pt_prev[i], mask);
      pt_prev_back[i] = pt_prev[i];
      tracker[i]->calc(img_prev, img_next, pt_prev[i], pt_next[i], status[i],
                       err[i]);
      tracker_back[i]->calc(img_next, img_prev, pt_next[i], pt_prev_back[i],
                            status_back[i], err[i]);
    }

    for (int i = 0; i < detector.size(); ++i) {
      if (!detector[i] || !tracker[i]) {
        continue;
      }
      for (int j = 0; j < pt_prev[0].size(); ++j) {
        cv::Scalar color{0, 0, 0};
        color[i] = 255;
        if (status[i][j] && (j >= status_back[i].size() || status_back[i][j])) {
          cv::arrowedLine(img_bgr, pt_prev[i][j], pt_next[i][j], color, 1, 8,
                          0);
        }
      }
    }

    cv::imshow("optical flow", img_bgr);
    if (cv::waitKey(0) == 'q') {
      break;
    }

    std::swap(img_prev, img_next);
    for (int i = 0; i < pt_prev.size(); ++i) {
      pt_prev[i].clear();
      pt_next[i].clear();
    }
  }

  return 0;
}
