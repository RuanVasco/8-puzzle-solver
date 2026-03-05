#include <vector>
#include <iostream>
#include "PuzzleBoard.h"

int main() {
    std::vector<int> state = {1, 2, 3, 4, 5, 6, 7, 8, 0};

    PuzzleBoard puzzle(3, state);

    std::cout << ">>> Puzzle criado com sucesso! <<<" << std::endl;

    int index = puzzle.get_index(2, 2);
    std::cout << "Se pedirmos a Linha 2 e Coluna 2, o indice 1D eh: " << index << std::endl;

    return 0;
}