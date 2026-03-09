#pragma once
#include <vector>
#include "PuzzleBoard.h"

class Solver {
public:
    std::vector<PuzzleBoard> solve_bfs(PuzzleBoard initial_board);
};