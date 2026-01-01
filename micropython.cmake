add_library(usermod_micropyML INTERFACE)

file(GLOB_RECURSE MICROPYML_SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.c)

target_sources(usermod_micropyML INTERFACE
        ${MICROPYML_SOURCES}
)

target_include_directories(usermod_micropyML INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(usermod INTERFACE usermod_micropyML)