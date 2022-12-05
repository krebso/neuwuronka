cmake_minimum_required(VERSION 3.13)  # CMake version check
project(neuwuronka)
set(CMAKE_CXX_STANDARD 17)            # Enable c++17 standard

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES neuwuronka/main.cpp)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -w")
set(CMAKE_CXX_FLAGS_DEBUG "-g")


# Add executable target with source files listed in SOURCE_FILES variable
add_executable(nn ${SOURCE_FILES})
