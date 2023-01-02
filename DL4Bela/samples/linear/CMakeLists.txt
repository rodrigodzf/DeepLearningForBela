set(EXE_NAME linear)
file(GLOB SRC_FILES *.cpp)
add_executable(${EXE_NAME} ${SRC_FILES})
target_include_directories(${EXE_NAME} PRIVATE ${BELA_ROOT} ${BELA_ROOT}/include)
if (ENABLE_TFLITE_FRONTEND)
    ## Add TfLite dependency
    find_package(TfLiteSrc REQUIRED MODULE)
    find_package(TfLite REQUIRED MODULE)
    ## Add Flatbuffers dependency
    find_package(Flatbuffers REQUIRED MODULE)
    # include/flatbuffers/flatbuffers.h is not warning clean
    # We can't change compilation flags on header files directly, so we need to add them to an interface library first
    add_library(flatbuffer_headers INTERFACE)
    target_include_directories(flatbuffer_headers INTERFACE $<BUILD_INTERFACE:${Flatbuffers_INCLUDE_DIR}>
                                                    $<INSTALL_INTERFACE:include/flatbuffer_headers>)
    target_compile_options(flatbuffer_headers INTERFACE -Wno-sign-conversion)
    add_library(tflite_headers INTERFACE)
    target_include_directories(tflite_headers INTERFACE $<BUILD_INTERFACE:${TfLite_INCLUDE_DIR}>
                                                        $<INSTALL_INTERFACE:include/tflite_headers>)
    target_compile_options(tflite_headers INTERFACE -Wno-conversion
                                                    -Wno-sign-conversion
                                                    -Wno-unused-parameter
                                                    -Wno-unused-function)
endif()
target_link_libraries(${EXE_NAME}
                    PRIVATE
                    ${BELA_ROOT}/lib/libbelafull.so
                    ${COBALT_LIB}
                    ${NEON_LIB}
                    ${MATHNEON_LIB}
                    tflite_headers
                    flatbuffer_headers
                    ${TfLite_LIB}
                    ${Flatbuffers_LIB}
                    dl
                    prussdrv
                    asound
                    atomic
                    sndfile
                    pthread
)
add_custom_command(
    TARGET ${EXE_NAME}
    POST_BUILD
    COMMAND rsync --timeout=10 -P ${CMAKE_BINARY_DIR}/bin/${EXE_NAME} root@$ENV{BBB_HOSTNAME}:~
)