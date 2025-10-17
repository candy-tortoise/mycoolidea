cmake_minimum_required(VERSION 3.25)
project(Mine LANGUAGES CXX CUDA)

enable_language(CXX)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
	set(CUDA_OPENMP_FLAGS "-Xcompiler=${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(OpenFHE REQUIRED)
find_package(FIDESlib REQUIRED)

if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else ()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})
set(CMAKE_CUDA_FLAGS "-I${OpenFHE_INCLUDE}/..")

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

add_executable(log-reg log-reg.cu)
target_link_libraries(log-reg PRIVATE FIDESlib::fideslib)

target_compile_options(log-reg PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -fchar8_t>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-fchar8_t,-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPENMP_FLAGS}>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G -g>
)
