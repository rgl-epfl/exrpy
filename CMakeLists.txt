cmake_minimum_required(VERSION 2.8.12)
project(exrpy)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  add_definitions(/D _CRT_SECURE_NO_WARNINGS)
endif()

add_subdirectory(ext/pybind11)
add_subdirectory(ext ext_build)
include_directories(SYSTEM ${OPENEXR_INCLUDE_DIRS} ext/pybind11)
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
endif()

pybind11_add_module(exrpy src/main.cpp)

target_link_libraries(exrpy LINK_PRIVATE IlmImf)

