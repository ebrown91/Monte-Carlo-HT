cmake_minimum_required(VERSION 3.15)
project(MonteCarloHeat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


cmake_minimum_required(VERSION 3.10)
project(MonteCarloHeat)

set(CMAKE_CXX_STANDARD 17)

# If vcpkg is installed, use its toolchain
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Build the executable
add_executable(MonteCarloHeat main.cpp)

# Link OpenCV libraries
target_link_libraries(MonteCarloHeat ${OpenCV_LIBS})
