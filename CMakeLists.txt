cmake_minimum_required(VERSION 3.18)
project(MAP VERSION 0.2.0)

set(CMAKE_GENERATOR "Unix Makefiles")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

####################################

add_compile_options(-march=armv7-a -mtune=cortex-a8 -mfloat-abi=hard -mfpu=neon -Wno-psabi)
add_compile_options(-O3 -g -fPIC -ftree-vectorize -ffast-math)
add_compile_definitions(DXENOMAI_SKIN_posix)

####################################
set(BELA_ROOT "${CMAKE_SYSROOT}/root/Bela") 
set(SYS_ROOT "${CMAKE_SYSROOT}")
set(TENSORFLOW_ROOT /opt/tensorflow )
set(TFLITE_LIB_ROOT /opt/tensorflow/build )
set(FLATBUFFERS_ROOT /opt/flatbuffers-arm32 )

find_library(COBALT_LIB REQUIRED
    NAMES cobalt libcobalt
    HINTS "${CMAKE_SYSROOT}/usr/xenomai/lib"
)

find_library(NEON_LIB REQUIRED
    NAMES NE10 libNE10
    HINTS "${CMAKE_SYSROOT}/usr/lib"
)

find_library(MATHNEON_LIB REQUIRED
    NAMES mathneon libmathneon
    HINTS "${CMAKE_SYSROOT}/usr/lib"
)

set(ENABLE_TFLITE_FRONTEND ON CACHE BOOL "ENABLE_TFLITE_FRONTEND" FORCE)
# set(ENABLE_ARMNN_FRONTEND ON CACHE BOOL "ENABLE_ARMNN_FRONTEND" FORCE)
# set(ENABLE_RTNEURAL_FRONTEND ON CACHE BOOL "ENABLE_RTNEURAL_FRONTEND" FORCE)

if(ENABLE_TFLITE_FRONTEND)
    add_compile_definitions(ENABLE_TFLITE_FRONTEND)
endif()

if(ENABLE_ARMNN_FRONTEND)
    add_compile_definitions(ENABLE_ARMNN_FRONTEND)
endif()

if(ENABLE_RTNEURAL_FRONTEND)
    include(ExternalProject)
    ExternalProject_Add(RTNeuralBuild
        GIT_REPOSITORY    git@github.com:jatinchowdhury18/RTNeural.git
        GIT_TAG           origin/main
        INSTALL_COMMAND   ""
        UPDATE_COMMAND    ""
        LOG_BUILD         ON
    )

    ExternalProject_Get_property(RTNeuralBuild SOURCE_DIR INSTALL_DIR BINARY_DIR)
    add_compile_definitions(ENABLE_RTNEURAL_FRONTEND)
    add_compile_definitions(RTNEURAL_DYNAMIC)
    
    add_library(RTNeural STATIC IMPORTED GLOBAL)

    set_target_properties(RTNeural PROPERTIES IMPORTED_LOCATION ${BINARY_DIR}/RTNeural/libRTNeural.a)
    set(RTNEURAL_INCLUDE_DIR ${SOURCE_DIR})
    # set_target_properties(RTNeural PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${SOURCE_DIR})
    # target_include_directories(RTNeural INTERFACE ${SOURCE_DIR})

    add_dependencies(RTNeural RTNeuralBuild)

endif()


add_subdirectory(DL4Bela/lib/Utils)
add_subdirectory(DL4Bela/lib/NNLib)

add_subdirectory(DL4Bela/samples/wavetable)
add_subdirectory(DL4Bela/samples/benchmark)

if(ENABLE_ARMNN_FRONTEND)
    add_subdirectory(DL4Bela/samples/keyword)
endif()
