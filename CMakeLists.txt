cmake_minimum_required(VERSION 3.13)  # CMake version check
project(neuwuronka)
set(CMAKE_CXX_STANDARD 17)            # Enable c++17 standard

find_package(OpenMP)
if (OPENMP_FOUND)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES neuwuronka/main.cpp)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-w -Ofast -ffp-contract=fast -funsafe-math-optimizations -march=native -msse2 -mfpmath=sse -ftree-vectorize -std=c++17 -fopenmp -I/opt/homebrew/Cellar/libomp/15.0.6/include")
set(CMAKE_CXX_FLAGS_DEBUG "-g")



# Add executable target with source files listed in SOURCE_FILES variable
add_executable(nn ${SOURCE_FILES})
