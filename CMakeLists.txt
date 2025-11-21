cmake_minimum_required(VERSION 3.16)

project(iqalab
        VERSION 0.1.0
        LANGUAGES CXX
)

if(CMAKE_COMPILER_IS_GNUCXX)
    SET(GCC_COVERAGE_COMPILE_FLAGS "-Werror=return-type")
    add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV – lepiej w tej formie:
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Główna biblioteka
add_subdirectory(src)

# Przykłady
add_subdirectory(examples)

# Instalacja nagłówków:
install(
        DIRECTORY include/iqalab/
        DESTINATION include/iqalab
)

