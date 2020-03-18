project(feature-tracker)

option(HAVE_NVX "nvidia visionworks" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11")

if(HAVE_NVX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_NVX=1")
endif()

find_package(CUDA)
find_package(OpenCV)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(3rdparty/nvxio/include)

set(SOURCES
  src/main.cpp
  src/feature/feature_common.cpp
  src/feature/good_features_to_track_detector.cpp
  src/feature/sparse_pyr_lk_optical_flow.cpp
)

if(HAVE_NVX)
  set(SOURCES
    ${SOURCES}
    3rdparty/nvxio/src/NVX/Utility.cpp
    3rdparty/nvxio/src/OVX/UtilityOVX.cpp
  )
endif()

add_executable(feature-tracker
  ${SOURCES}
)

target_link_libraries(feature-tracker
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
  dl pthread rt stdc++
)

if(HAVE_NVX)
  target_link_libraries(feature-tracker
    visionworks
  )
endif()