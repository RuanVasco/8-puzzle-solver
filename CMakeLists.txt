cmake_minimum_required(VERSION 4.0)

project(8_puzzle_solver VERSION 1.0 LANGUAGES CXX)

set(CMAKE_XX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/models/PuzzleBoard.cpp
 "include/models/Solver.h" "src/models/Solver.cpp")