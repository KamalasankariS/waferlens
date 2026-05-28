cmake_minimum_required(VERSION 3.18)
project(waferlens VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(WAFERLENS_BUILD_TESTS   "Build C++ unit tests"        ON)
option(WAFERLENS_BUILD_PYTHON  "Build pybind11 Python module" ON)

# --- CUDA detection (optional) -----------------------------------------------

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(WAFERLENS_CUDA_FOUND TRUE)
    message(STATUS "waferlens: CUDA enabled (${CMAKE_CUDA_COMPILER_VERSION})")
else()
    set(WAFERLENS_CUDA_FOUND FALSE)
    message(STATUS "waferlens: CUDA not found -- building CPU-only")
endif()

# --- Core library -------------------------------------------------------------

set(CPU_SOURCES
    src/cpu/radial_density.cpp
    src/cpu/angular_histogram.cpp
    src/cpu/edge_concentration.cpp
    src/cpu/ring_score.cpp
    src/cpu/connected_components.cpp
    src/cpu/scratch_linearity.cpp
    src/cpu/density_gradient.cpp
    src/wafer_analyzer.cpp
)

if(WAFERLENS_CUDA_FOUND)
    set(CUDA_SOURCES
        src/kernels/radial_density.cu
        src/kernels/angular_histogram.cu
        src/kernels/connected_components.cu
        src/kernels/scratch_linearity.cu
        src/kernels/density_gradient.cu
        src/kernels/cuda_dispatch.cu
    )
    add_library(waferlens ${CPU_SOURCES} ${CUDA_SOURCES})
    target_compile_definitions(waferlens PUBLIC WAFERLENS_CUDA_ENABLED)
    set_target_properties(waferlens PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    add_library(waferlens ${CPU_SOURCES})
endif()

target_include_directories(waferlens
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Python bindings (pybind11) -----------------------------------------------

if(WAFERLENS_BUILD_PYTHON)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.12.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(_waferlens_core python/waferlens/_core.cpp)
    target_link_libraries(_waferlens_core PRIVATE waferlens)
    target_include_directories(_waferlens_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    install(TARGETS _waferlens_core
        LIBRARY DESTINATION waferlens
    )
endif()

# --- C++ tests ----------------------------------------------------------------

if(WAFERLENS_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(waferlens_tests tests/cpp/test_features.cpp)
    target_link_libraries(waferlens_tests PRIVATE waferlens GTest::gtest_main)
    target_include_directories(waferlens_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    include(GoogleTest)
    gtest_discover_tests(waferlens_tests)
endif()

# --- Install ------------------------------------------------------------------

install(TARGETS waferlens
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/waferlens DESTINATION include)
