set(EXE_NAME wavetable)

file(GLOB SRC_FILES *.cpp)

add_executable(${EXE_NAME} ${SRC_FILES})

target_include_directories(${EXE_NAME} PRIVATE ${BELA_ROOT} ${BELA_ROOT}/include)

target_link_libraries(${EXE_NAME}
                      PRIVATE
                      NNLib
                      ${BELA_ROOT}/lib/libbelafull.so
                      ${COBALT_LIB}
                      ${NEON_LIB}
                      ${MATHNEON_LIB}
                      prussdrv
                      asound
                      atomic
                      sndfile
                      pthread
                      Utils
                      )

add_custom_command(
    TARGET ${EXE_NAME}
    POST_BUILD
    COMMAND rsync --timeout=10 -P ${CMAKE_BINARY_DIR}/bin/${EXE_NAME} root@$ENV{BBB_HOSTNAME}:~
)