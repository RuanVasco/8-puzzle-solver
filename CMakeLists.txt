cmake_minimum_required(VERSION 4.0)

project(8_puzzle_solver VERSION 1.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)
include(FetchContent)

FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG master
)
FetchContent_MakeAvailable(raylib)

add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/models/PuzzleBoard.cpp
    "include/models/Solver.h"
    "src/models/Solver.cpp"
    "include/ui/Button.h"
    "src/ui/Button.cpp"
    "include/ui/Application.h"
    "src/ui/Application.cpp"
)

target_link_libraries(${PROJECT_NAME} PRIVATE raylib)

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES 
        WIN32_EXECUTABLE $<$<CONFIG:Release>:TRUE>
    )
    
    target_link_options(${PROJECT_NAME} PRIVATE 
        $<$<CONFIG:Release>:/SUBSYSTEM:WINDOWS;/ENTRY:mainCRTStartup>
    )
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)