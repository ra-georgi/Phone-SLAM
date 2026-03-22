cmake_minimum_required(VERSION 3.28)
project(Phone_SLAM)
set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/results/bin)
include_directories(include/)
add_executable(plot_gyro src/plot_gyro.cpp)