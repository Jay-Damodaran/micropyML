add_library(usermod_ulabML INTERFACE)

file(GLOB_RECURSE ULABML_SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.c)

target_sources(usermod_ulabML INTERFACE
        ${ULABML_SOURCES}
)

target_include_directories(usermod_ulabML INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(usermod INTERFACE usermod_ulabML)