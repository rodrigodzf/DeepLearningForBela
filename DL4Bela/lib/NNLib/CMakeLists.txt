set(LIB_NAME NNLib)

if (ENABLE_TFLITE_FRONTEND)
    list(APPEND SRC_FILES TFLiteFrontend.cpp)
endif()

if (ENABLE_ARMNN_FRONTEND)
    list(APPEND SRC_FILES ArmNNFrontend.cpp)
endif()

add_library(${LIB_NAME} ${SRC_FILES})

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

if (ENABLE_ARMNN_FRONTEND)
    target_link_directories(${LIB_NAME} PUBLIC /opt/armnn-dist/armnn/lib)
    target_include_directories(${LIB_NAME} PUBLIC /opt/armnn-dist/armnn/include)
    target_include_directories(${LIB_NAME} PUBLIC /opt/armnn-dist/armnn/delegate/include)
endif()

target_include_directories(${LIB_NAME} PUBLIC include/)
# target_compile_definitions(${LIB_NAME} PRIVATE RTNEURAL_STATIC)

if (ENABLE_TFLITE_FRONTEND)
    target_link_libraries(${LIB_NAME}
                        PUBLIC
                        tflite_headers
                        flatbuffer_headers
                        ${TfLite_LIB}
                        ${Flatbuffers_LIB}
                        armnnDelegate
                        protobuf
                        dl
                        pthread)
endif()

if (ENABLE_ARMNN_FRONTEND)
    target_link_libraries(${LIB_NAME}
                        PUBLIC
                        armnn
                        armnnOnnxParser
                        armnnTfLiteParser
                        Utils
                        )
endif()

if (ENABLE_RTNEURAL_FRONTEND)
    target_link_libraries(${LIB_NAME}
                        PRIVATE
                        RTNeural)
endif()

target_link_libraries(${LIB_NAME}
                      PRIVATE
                      Utils)