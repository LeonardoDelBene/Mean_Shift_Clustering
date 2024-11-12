cmake_minimum_required(VERSION 3.10)

# Set the project name and the C++ standard
project(Mean_Shift_Clustering)

# Use C++17 or later
set(CMAKE_CXX_STANDARD 17)
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")
    # Get Homebrew home directory
    execute_process(
            COMMAND brew --prefix
            OUTPUT_VARIABLE HOMEBREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # The following 2 lines are needed on macOS.
    # Check the position of headers in libraries (e.g. Macports vs. Homebrew)
    include_directories(${HOMEBREW_PREFIX}/opt/libomp/include)
    link_directories(${HOMEBREW_PREFIX}/opt/libomp/lib)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Add the executable target
add_executable(Mean_Shift_Clustering main.cpp)





