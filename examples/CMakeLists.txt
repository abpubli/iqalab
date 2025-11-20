add_executable(jpeg_block_scan jpeg_block_scan.cpp)
target_link_libraries(jpeg_block_scan PRIVATE iqalab)

add_executable(regions_demo regions_demo.cpp)
target_link_libraries(regions_demo PRIVATE iqalab)
