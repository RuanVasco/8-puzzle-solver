cmake_minimum_required(VERSION 4.0)

project(8_puzzle_solver VERSION 1.0 LANGUAGES CXX)

set(CMAKE_XX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/PuzzleBoard.cpp
)