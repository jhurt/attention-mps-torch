cmake_minimum_required(VERSION 4.2)
project(attention_mps_torch LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD 20)
set(LIB_NAME attention_mps_torch_lib)

add_compile_options(-Wno-elaborated-enum-base)

# Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# site-packages
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Found site-packages at: ${SITE_PACKAGES}")

# torch
set(Torch_DIR ${SITE_PACKAGES}/torch/share/cmake/Torch)
find_package(Torch REQUIRED)

# mlx
include(FetchContent)
FetchContent_Declare(
        mlx
        GIT_REPOSITORY https://github.com/jhurt/mlx.git
        GIT_TAG main
)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(CMAKE_INSTALL_LIBDIR attention_mps)
FetchContent_MakeAvailable(mlx)

add_library(${LIB_NAME} SHARED
        src/library.mm
)

set_target_properties(${LIB_NAME} PROPERTIES
        PREFIX ""
        SUFFIX ".so"
)

target_link_libraries(${LIB_NAME} PUBLIC
        ${TORCH_LIBRARIES}
        Python3::Python
        mlx
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework MetalPerformanceShadersGraph"
)
target_compile_definitions(${LIB_NAME} PUBLIC TORCH_EXTENSION_NAME=${LIB_NAME})

install(TARGETS ${LIB_NAME}
        DESTINATION attention_mps)
