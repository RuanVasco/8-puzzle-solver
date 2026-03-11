#include "models/PuzzleBoard.h"

PuzzleBoard::PuzzleBoard(int size, const std::vector<int>& initial_state) {
    this->n = size;
    this->board = initial_state;

    for (int i = 0; i < board.size(); i++) { // localiza a posição da peça vazia para facilitar as trocas futuras
        if (board[i] == 0) {
            this->zero_index = i;
            break;
        }
    }
}


std::vector<PuzzleBoard> PuzzleBoard::get_neighbors() const {
    std::vector<PuzzleBoard> neighbors; // converte o índice linear (1d) do zero para coordenadas de matriz (2d)
    int row = zero_index / n;
    int col = zero_index % n;

    std::vector<std::pair<int, int>> directions = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };

    for (const auto& dir : directions) {
        
        int new_row = row + dir.first;
        int new_col = col + dir.second;
// garante que o movimento não saia das bordas do tabuleiro
        if (new_row >= 0 && new_row < n && new_col >= 0 && new_col < n) {
            int new_index = new_row * n + new_col; // mapeia a nova coordenada 2d de volta para a posição linear do vetor
            std::vector<int> new_state = board;
            std::swap(new_state[zero_index], new_state[new_index]); // realiza o movimento trocando o zero com a peça adjacente
            neighbors.push_back(PuzzleBoard(n, new_state)); // cadastra o tabuleiro resultante como um estado vizinho válido
        }
    }

    return neighbors;
}
