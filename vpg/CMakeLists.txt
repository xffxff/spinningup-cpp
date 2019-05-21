cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(vpg)

find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(vpg run.cpp cartpole.h agent.h model.h)

target_link_libraries(vpg "${TORCH_LIBRARIES}")
set_property(TARGET vpg PROPERTY CXX_STANDARD 11)