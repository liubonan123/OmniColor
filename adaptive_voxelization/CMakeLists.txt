cmake_minimum_required(VERSION 2.8.3)
project(adaptive_voxel)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
     ${PCL_INCLUDE_DIRS}
)

SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(adaptive_voxel src/adaptive_voxelization.cpp)
target_link_libraries(adaptive_voxel ${PCL_LIBRARIES} ${OpenCV_LIBS} )