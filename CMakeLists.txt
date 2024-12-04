cmake_minimum_required(VERSION 3.16)
project(Quadrature VERSION 1.0.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the library
add_library(quadrature STATIC
    source/gauss_jacobi.cpp
    source/gauss_legendre.cpp
    source/gauss_lobatto.cpp
)

# Specify the include directory for the public header
target_include_directories(quadrature PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add optimization flags
target_compile_options(quadrature PRIVATE -O3)

# Link LAPACK
find_package(LAPACK REQUIRED)
target_link_libraries(quadrature PRIVATE LAPACK::LAPACK)

# Enable testing
enable_testing()

# Add tests
file(GLOB TEST_SOURCES tests/*.cpp)
foreach(TEST_FILE ${TEST_SOURCES})
    # Get the name of the test from the filename
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE quadrature)
    set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_BINARY_DIR}/tests/${TEST_NAME})
endforeach()

# Optional: Install targets
install(TARGETS quadrature
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES quadrature.hpp DESTINATION include)
