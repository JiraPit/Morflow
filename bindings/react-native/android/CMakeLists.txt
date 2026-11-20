# The C++ module, built into the app's native library by React Native's
# autolinking (see react-native.config.js).
cmake_minimum_required(VERSION 3.13)
project(react_native_morflow)

set(MORFLOW_CPP ${CMAKE_CURRENT_LIST_DIR}/../cpp)

add_library(react_native_morflow STATIC
  ${MORFLOW_CPP}/MorflowCapi.cpp
  ${MORFLOW_CPP}/MorflowHost.cpp
  ${MORFLOW_CPP}/MorflowModule.cpp
)

target_include_directories(react_native_morflow PUBLIC ${MORFLOW_CPP})

target_link_libraries(react_native_morflow
  jsi
  reactnative
  react_codegen_RNMorflowSpec
  dl
)

target_compile_reactnative_options(react_native_morflow PRIVATE)
