#pragma once
#include <vector>

class PuzzleBoard {
private:
    int n;
    std::vector<int> board;
    int zero_index;

public:
    PuzzleBoard(int size, const std::vector<int>& initial_state);

    int get_index(int row, int col) const {
        return row * n + col;
    }

    const std::vector<int>& get_board() const {
        return board;
    }
};