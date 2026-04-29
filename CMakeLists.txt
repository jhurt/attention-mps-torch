cmake_minimum_required(VERSION 4.2)
project(attention_mps_torch LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)

set(LIB_NAME attention_mps_torch_lib)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
        OUTPUT_VARIABLE SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Found site-packages at: ${SITE_PACKAGES}")

set(Torch_DIR ${SITE_PACKAGES}/torch/share/cmake/Torch)
find_package(Torch REQUIRED)

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
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework MetalPerformanceShadersGraph"
)
target_compile_definitions(${LIB_NAME} PUBLIC TORCH_EXTENSION_NAME=${LIB_NAME})

install(TARGETS ${LIB_NAME}
        DESTINATION attention_mps)