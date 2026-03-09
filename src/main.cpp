#include <iostream>
#include <vector>
#include "models/PuzzleBoard.h"
#include <models/Solver.h>

void print_board(const PuzzleBoard& board) {
    const std::vector<int>& state = board.get_board();
    for (int i = 0; i < 9; i++) {
        std::cout << state[i] << " ";
        if ((i + 1) % 3 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << "---" << std::endl;
}

int main() {
    std::vector<int> initial_state = { 1, 2, 3, 4, 0, 5, 7, 8, 6 };
    PuzzleBoard puzzle(3, initial_state);

    std::cout << "Starting BFS..." << std::endl;

    Solver solver;
    std::vector<PuzzleBoard> solution = solver.solve_bfs(puzzle);

    if (solution.empty()) {
        std::cout << "Unsolvable puzzle" << std::endl;
    }
    else {
        std::cout << "Solved in " << solution.size() - 1 << " moves!" << std::endl;

        for (const PuzzleBoard& step : solution) {
            print_board(step);
        }
    }

    return 0;
}