#pragma once
#include <vector>
#include "PuzzleBoard.h"

class Solver {
private:
    int last_tested_states;
public:
    std::vector<PuzzleBoard> solve_bfs(PuzzleBoard initial_board, PuzzleBoard goal_board);
    int get_tested_states() const;
};