set(IQALAB_SOURCES
        iqalab.cpp
        blocking.cpp
        region_masks.cpp
        image_type.cpp
)

add_library(iqalab SHARED ${IQALAB_SOURCES})

target_include_directories(iqalab
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(iqalab
        PUBLIC
        ${OpenCV_LIBS}
)

set_target_properties(iqalab PROPERTIES
        OUTPUT_NAME "iqalab"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)

install(
        TARGETS iqalab
        EXPORT iqalabTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)
