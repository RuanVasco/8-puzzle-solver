#include "PuzzleBoard.h"

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