set(EXE_NAME benchmark)

add_executable(${EXE_NAME}
    main.cpp
)

target_link_libraries(${EXE_NAME}
                      PRIVATE
                      NNLib
                      Utils
                      )

add_custom_command(
    TARGET ${EXE_NAME}
    POST_BUILD
    COMMAND rsync --timeout=10 -P ${CMAKE_BINARY_DIR}/bin/${EXE_NAME} root@$ENV{BBB_HOSTNAME}:~
)