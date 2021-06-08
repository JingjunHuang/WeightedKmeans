cmake_minimum_required(VERSION 3.10.2)
project(kmeans)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-o2")
set(OpenCV_INCLUDE_DIRS "/opt/homebrew/Cellar/opencv/4.5.2_4")

find_package(OpenCV REQUIRED)

MESSAGE ( STATUS "OpenCV library status:" )
MESSAGE ( STATUS "version ${OpenCV_VERSION}" )
MESSAGE ( STATUS "libraries ${OpenCV_LIBS}" )
MESSAGE ( STATUS "include path ${OpenCV_INCLUDE_DIRS}" )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(k_means ${CMAKE_SOURCE_DIR}/k_means.cpp)


add_executable(test_k_means ${CMAKE_SOURCE_DIR}/test_k_means.cpp)
target_link_libraries(test_k_means ${OpenCV_LIBS} k_means)