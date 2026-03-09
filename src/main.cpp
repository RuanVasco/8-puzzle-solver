#include <iostream>
#include <vector>
#include "models/PuzzleBoard.h"
#include <models/Solver.h>
#include "raylib.h"
#include <string>
#include <ui/Button.h>

int main() {
    const int screenWidth = 600;
    const int screenHeight = 700;
    const int cellSize = 200;
    const int boardOffset = 100;

    InitWindow(screenWidth, screenHeight, "8 Puzzle Solver");
    SetTargetFPS(60);

    std::vector<int> initialState = { 1, 2, 3, 4, 0, 5, 7, 8, 6 };
    PuzzleBoard puzzle(3, initialState);
    Solver solver = Solver();

    Button solveButton(200, 25, 200, 50, "Solve");

    std::vector<PuzzleBoard> solutionPath;
    bool isAnimating = false;
    int currentStep = 0;
    int framesCounter = 0;

    while (!WindowShouldClose()) {
        if (solveButton.is_clicked() && !isAnimating) {
            solveButton.set_disabled(true);

            solutionPath = solver.solve_bfs(puzzle);

            if (!solutionPath.empty()) {
                isAnimating = true;
                currentStep = 0;
                framesCounter = 0;
            }
            else {
                solveButton.set_disabled(false);
            }
        }

        if (isAnimating) {
            framesCounter++;
            if (framesCounter >= 30) {
                framesCounter = 0;
                if (currentStep < solutionPath.size() - 1) {
                    currentStep++;
                    puzzle = solutionPath[currentStep];
                }
                else {
                    isAnimating = false;
                    solveButton.set_disabled(false);
                }
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        solveButton.draw();

        const std::vector<int>& state = puzzle.get_board();

        for (int i = 0; i < 9; i++) {
            int row = i / 3;
            int col = i % 3;
            int value = state[i];

            int x = col * cellSize;
            int y = (row * cellSize) + boardOffset;

            if (value != 0) {
                DrawRectangle(x + 5, y + 5, cellSize - 10, cellSize - 10, DARKBLUE);

                std::string text = std::to_string(value);
                int valueTextWidth = MeasureText(text.c_str(), 60);
                DrawText(text.c_str(), x + (cellSize - valueTextWidth) / 2, y + (cellSize - 60) / 2, 60, WHITE);
            }
            else {
                DrawRectangle(x + 5, y + 5, cellSize - 10, cellSize - 10, LIGHTGRAY);
            }
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}