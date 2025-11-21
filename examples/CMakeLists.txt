add_executable(jpeg_block_scan jpeg_block_scan.cpp)
target_link_libraries(jpeg_block_scan PRIVATE iqalab)

add_executable(regions_demo regions_demo.cpp)
target_link_libraries(regions_demo PRIVATE iqalab)

add_executable(lab_shift_demo lab_shift_demo.cpp)
target_link_libraries(lab_shift_demo PRIVATE iqalab)

add_executable(impulse_removal impulse_removal.cpp)
target_link_libraries(impulse_removal PRIVATE iqalab)

add_executable(dithering_removal dithering_removal.cpp)
target_link_libraries(dithering_removal PRIVATE iqalab)

add_executable(impulse_mask_tool impulse_mask_tool.cpp)
target_link_libraries(impulse_mask_tool PRIVATE iqalab)

add_executable(blocking_mask_tool blocking_mask_tool.cpp)
target_link_libraries(blocking_mask_tool PRIVATE iqalab)

add_executable(mse_info mse_info.cpp)
target_link_libraries(mse_info PRIVATE iqalab)
