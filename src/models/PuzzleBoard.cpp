#include "models/PuzzleBoard.h"

PuzzleBoard::PuzzleBoard(int size, const std::vector<int>& initial_state) {
    this->n = size;
    this->board = initial_state;

    for (int i = 0; i < board.size(); i++) {
        if (board[i] == 0) {
            this->zero_index = i;
            break;
        }
    }
}


std::vector<PuzzleBoard> PuzzleBoard::get_neighbors() const {
    std::vector<PuzzleBoard> neighbors;
    int row = zero_index / n;
    int col = zero_index % n;

    std::vector<std::pair<int, int>> directions = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };

    for (const auto& dir : directions) {
        int new_row = row + dir.first;
        int new_col = col + dir.second;

        if (new_row >= 0 && new_row < n && new_col >= 0 && new_col < n) {
            int new_index = new_row * n + new_col;
            std::vector<int> new_state = board;
            std::swap(new_state[zero_index], new_state[new_index]);
            neighbors.push_back(PuzzleBoard(n, new_state));
        }
    }

    return neighbors;
}
