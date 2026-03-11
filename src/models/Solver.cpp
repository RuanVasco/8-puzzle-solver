#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>
#include <models/PuzzleBoard.h>
#include <models/Solver.h>

int Solver::get_tested_states() const {
    return this->last_tested_states;
}

std::vector<PuzzleBoard> Solver::solve_bfs(PuzzleBoard initial_board, PuzzleBoard goal_board) {
    std::queue<PuzzleBoard> queue; // fila para explorar os estados nível por nível
    std::set<std::vector<int>> visited; // conjunto para registrar estados visitados e evitar loops infinitos
    std::map<std::vector<int>, PuzzleBoard> parent_map; // mapa de ancestralidade para reconstruir a sequência de movimentos no final

    queue.push(initial_board); // inicializa as estruturas com o estado de origem
    visited.insert(initial_board.get_board());

    bool found = false;

    while (!queue.empty()) {
        PuzzleBoard current = queue.front();
        queue.pop();

        if (current.get_board() == goal_board.get_board()) {
            found = true;
            break;
        }

        for (const PuzzleBoard& neighbor : current.get_neighbors()) { // processa o vizinho apenas se ele for um estado inédito
            if (visited.find(neighbor.get_board()) == visited.end()) {
                visited.insert(neighbor.get_board());
                parent_map.insert({ neighbor.get_board(), current }); // cadastra o estado atual como pai do novo vizinho descoberto
                queue.push(neighbor);
            }
        }
    }

    this->last_tested_states = visited.size(); // salva o total de nós analisados para exibição nas estatísticas

    if (!found) {
        return {};
    }

    std::vector<PuzzleBoard> path;
    PuzzleBoard current = goal_board;
    path.push_back(current);

    while (current.get_board() != initial_board.get_board()) { // faz o caminho reverso da meta até o início consultando os pais
        current = parent_map.at(current.get_board());
        path.push_back(current);
    }

    std::reverse(path.begin(), path.end()); // inverte a ordem para que o caminho fique do início para a meta
    return path;
}