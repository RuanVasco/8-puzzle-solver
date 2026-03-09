#include <queue>
#include <set>
#include <models/PuzzleBoard.h>
#include <models/Solver.h>

std::vector<PuzzleBoard> Solver::solve_bfs(PuzzleBoard initial_board) {
    std::queue<std::vector<PuzzleBoard>> queue;
    std::set<std::vector<int>> visited;

    queue.push({ initial_board });
    visited.insert(initial_board.get_board());

    while (!queue.empty()) {
        std::vector<PuzzleBoard> path = queue.front();
        queue.pop();

        PuzzleBoard current = path.back();

        if (current.is_goal()) {
            return path;
        }

        for (const PuzzleBoard& neighbor : current.get_neighbors()) {
            if (visited.find(neighbor.get_board()) == visited.end()) {
                visited.insert(neighbor.get_board());
                std::vector<PuzzleBoard> new_path = path;
                new_path.push_back(neighbor);
                queue.push(new_path);
            }
        }
    }

    return {};
}