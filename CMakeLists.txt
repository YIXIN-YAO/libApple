cmake_minimum_required(VERSION 3.10)

project(yolox)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local//cuda/lib64)
# cudnn
include_directories(/usr/local/cudnn/include)
link_directories(/usr/local/cudnn/lib64)
# tensorrt
include_directories(/home/server/user/TensorRT-7/include)
link_directories(/home/server/user/TensorRT-7/lib)
link_directories(${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


add_library(Apple SHARED libApple.cpp libApple.h common.cpp common.h yolox.cpp yolox.h Realsense.cpp Realsense.h AppleDetect.cpp AppleDetect.h)
target_link_libraries(Apple nvinfer)
target_link_libraries(Apple cudart)
target_link_libraries(Apple ${OpenCV_LIBS} realsense2)

add_executable(testlib ${PROJECT_SOURCE_DIR}/testLib.cpp ${PROJECT_SOURCE_DIR}/yolox.cpp common.cpp common.h)
target_link_libraries(testlib nvinfer)
target_link_libraries(testlib cudart)
target_link_libraries(testlib ${OpenCV_LIBS} realsense2 Apple)
add_definitions(-O0 -pthread)

