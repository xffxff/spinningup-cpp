cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ppo)

find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

FILE(GLOB HEAD_RIGEL_EDITOR ./*.h)
FILE(GLOB_RECURSE SRCS_RIGEL_EDITOR ./*.cpp)
add_executable(ppo ${HEAD_RIGEL_EDITOR} ${SRCS_RIGEL_EDITOR})

target_link_libraries(ppo "${TORCH_LIBRARIES}")
set_property(TARGET ppo PROPERTY CXX_STANDARD 11)
