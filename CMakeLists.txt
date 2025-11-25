cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mnist_inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# Find Dependencies
# =============================================================================

# Find LibTorch
# Set TORCH_DIR to your LibTorch installation path if not in standard location
# Example: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

if(TORCH_FOUND)
    message(STATUS "Found LibTorch: ${TORCH_INSTALL_PREFIX}")
    message(STATUS "  Version: ${Torch_VERSION}")
    message(STATUS "  Libraries: ${TORCH_LIBRARIES}")
else()
    message(FATAL_ERROR "LibTorch not found. Please set CMAKE_PREFIX_PATH to LibTorch installation.")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_DIR}")
    message(STATUS "  Version: ${OpenCV_VERSION}")
    message(STATUS "  Libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV.")
endif()

# =============================================================================
# Compiler Flags
# =============================================================================

# Enable warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# LibTorch requires these flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# =============================================================================
# Include Directories
# =============================================================================

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# =============================================================================
# Source Files
# =============================================================================

set(SOURCES
    src/inference.cpp
    src/main.cpp
)

# =============================================================================
# Build Targets
# =============================================================================

# Main executable
add_executable(mnist_inference ${SOURCES})

# Link libraries
target_link_libraries(mnist_inference
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# Batch inference executable
add_executable(batch_inference_cpp
    src/inference.cpp
    src/batch_inference.cpp
)

# Link libraries
target_link_libraries(batch_inference_cpp
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# =============================================================================
# Platform-specific Settings
# =============================================================================

# Copy DLLs to output directory on Windows
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET mnist_inference
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:mnist_inference>
    )
    add_custom_command(TARGET batch_inference_cpp
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:batch_inference_cpp>
    )
endif()

# Set RPATH for Linux
if(UNIX AND NOT APPLE)
    set_target_properties(mnist_inference PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    set_target_properties(batch_inference_cpp PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS mnist_inference batch_inference_cpp
    RUNTIME DESTINATION bin
)

# =============================================================================
# Print Configuration Summary
# =============================================================================

message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project:        ${PROJECT_NAME}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  LibTorch:       ${TORCH_INSTALL_PREFIX}")
message(STATUS "  OpenCV:         ${OpenCV_VERSION}")
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  CXX Compiler:   ${CMAKE_CXX_COMPILER}")
message(STATUS "  CXX Flags:      ${CMAKE_CXX_FLAGS}")
message(STATUS "")
