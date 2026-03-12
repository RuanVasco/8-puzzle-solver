#include "ui/Application.h"
#include <algorithm>
#include <iostream>

const LangStrings Application::LANG_PT = {
    "Definir Estado Inicial", "Embaralhar", "Resolver", "Ver Arvore", "Voltar",
    "Estados Testados: ", "SEM SOLUCAO", "Passos: ", "Processando...",
    "Digite 9 digitos (0-8):", "ENTER confirma, ESC cancela",
    "Arvore da Solucao", "movimentos", "estados testados",
    "Estado Inicial", "Estado Final (Meta)", "Passo"
};

const LangStrings Application::LANG_EN = {
    "Set Initial State", "Shuffle", "Solve", "View Tree", "Back",
    "Tested States: ", "UNSOLVABLE", "Steps: ", "Processing...",
    "Enter 9 unique digits (0-8):", "ENTER to confirm, ESC to cancel",
    "Solution Tree", "moves", "states tested",
    "Initial State", "Final State (Goal)", "Step"
};

Application::Application()
    : screenWidth(850), screenHeight(600), sidebarWidth(250), cellSize(200),
    puzzle(3, { 1, 2, 3, 4, 0, 5, 7, 8, 6 }), solver(),
    btnInput(25, 50, 200, 50, "Set Initial State"),
    btnShuffle(25, 120, 200, 50, "Shuffle"),
    btnSolve(25, 190, 200, 50, "Solve", DARKGREEN, GREEN, GRAY),
    btnShowTree(25, 335, 200, 50, "View Tree", DARKBLUE, BLUE, GRAY),
    btnBack(screenWidth - 130, 10, 110, 40, "Back", DARKGRAY, GRAY, DARKGRAY),
    isAnimating(false), isProcessing(false), isSolved(false), isModalOpen(false),
    isTreeViewOpen(false), currentStep(0), framesCounter(0), testedStates(0),
    treeScrollOffset(0), currentLang(Language::PT), lang(LANG_PT) {

    InitWindow(screenWidth, screenHeight, "8 Puzzle Solver");
    SetTargetFPS(60);
    SetExitKey(KEY_NULL);
    setLanguage(Language::PT);
}

Application::~Application() {
    CloseWindow();
}

void Application::setLanguage(Language l) {
    currentLang = l;
    lang = (l == Language::PT) ? LANG_PT : LANG_EN;
    btnInput.set_text(lang.btnInput);
    btnShuffle.set_text(lang.btnShuffle);
    btnSolve.set_text(lang.btnSolve);
    btnShowTree.set_text(lang.btnShowTree);
    btnBack.set_text(lang.btnBack);
}

void Application::run() {
    while (!WindowShouldClose()) {
        handleEvents();
        updateLogic();
        render();
    }
}

void Application::handleEvents() {
    if (isTreeViewOpen) {
        btnBack.update();
        if (btnBack.get_is_hovered()) SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);
        else SetMouseCursor(MOUSE_CURSOR_DEFAULT);

        if (btnBack.is_clicked()) {
            isTreeViewOpen = false;
            treeScrollOffset = 0;
        }

        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            const int nodeSpacing = 179;  
            const int lastNodeH = 155;   
            const int contentTopPad = 10;
            const int bottomPad = 20;
            int n = (int)solutionPath.size();
            int totalHeight = contentTopPad + (n - 1) * nodeSpacing + lastNodeH + bottomPad;
            int viewHeight = screenHeight - 60;
            int maxScroll = std::max(0, totalHeight - viewHeight);
            treeScrollOffset -= (int)(wheel * 40);
            if (treeScrollOffset < 0) treeScrollOffset = 0;
            if (treeScrollOffset > maxScroll) treeScrollOffset = maxScroll;
        }
        return;
    }

    btnInput.update();
    btnShuffle.update();
    btnSolve.update();
    if (isSolved && !solutionPath.empty() && !isAnimating) btnShowTree.update();

    bool anyHovered = btnInput.get_is_hovered() || btnShuffle.get_is_hovered() ||
                      btnSolve.get_is_hovered() || btnShowTree.get_is_hovered();
    if (anyHovered && !isModalOpen) {
        SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);
    }
    else {
        SetMouseCursor(MOUSE_CURSOR_DEFAULT);
    }

    if (isModalOpen) {
        handleModalInput();
    }
    else {
        if (btnInput.is_clicked() && !isAnimating && !isProcessing) {
            isModalOpen = true;
            inputText = "";
        }

        if (btnShuffle.is_clicked() && !isAnimating && !isProcessing) {
            for (int i = 0; i < 100; i++) {
                std::vector<PuzzleBoard> neighbors = puzzle.get_neighbors();
                int randomIndex = GetRandomValue(0, neighbors.size() - 1);
                puzzle = neighbors[randomIndex];
            }
            isSolved = false;
        }

        if (btnSolve.is_clicked() && !isAnimating && !isProcessing) {
            btnInput.set_disabled(true);
            btnShuffle.set_disabled(true);
            btnSolve.set_disabled(true);
            isProcessing = true;
            isSolved = false;
            futureSolution = std::async(std::launch::async, &Solver::solve_bfs, &solver, puzzle, PuzzleBoard(3, { 1, 2, 3, 4, 5, 6, 7, 8, 0 }));
        }

        if (isSolved && !solutionPath.empty() && !isAnimating && btnShowTree.is_clicked()) {
            isTreeViewOpen = true;
            treeScrollOffset = 0;
        }
    }
}

void Application::pasteFromClipboard() {
    const char* clipboardText = GetClipboardText();
    if (clipboardText != nullptr) {
        std::string pasted(clipboardText);
        for (char c : pasted) {
            if (c >= '0' && c <= '8' && inputText.length() < 9) {
                if (inputText.find(c) == std::string::npos) {
                    inputText += c;
                }
            }
        }
    }
}

void Application::handleModalInput() {
    if ((IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL)) && IsKeyPressed(KEY_V)) {
        pasteFromClipboard();
    }

    int key = GetCharPressed();
    while (key > 0) {
        if ((key >= '0' && key <= '8') && inputText.length() < 9) {
            if (inputText.find((char)key) == std::string::npos) {
                inputText += (char)key;
            }
        }
        key = GetCharPressed();
    }

    if (IsKeyPressed(KEY_BACKSPACE) && inputText.length() > 0) {
        inputText.pop_back();
    }

    if ((IsKeyPressed(KEY_ENTER) || IsKeyPressed(KEY_KP_ENTER)) && inputText.length() == 9) {
        std::vector<int> newState;
        for (char c : inputText) {
            newState.push_back(c - '0');
        }
        puzzle = PuzzleBoard(3, newState);
        isModalOpen = false;
        isSolved = false;
    }

    if (IsKeyPressed(KEY_ESCAPE)) {
        isModalOpen = false;
    }
}

void Application::updateLogic() {
    if (isProcessing) {
        if (futureSolution.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            solutionPath = futureSolution.get();
            isProcessing = false;
            testedStates = solver.get_tested_states();
            isSolved = true;

            if (!solutionPath.empty()) {
                isAnimating = true;
                currentStep = 0;
                framesCounter = 0;
            }
            else {
                btnInput.set_disabled(false);
                btnShuffle.set_disabled(false);
                btnSolve.set_disabled(false);
            }
        }
    }

    if (isAnimating) {
        framesCounter++;
        if (framesCounter >= 30) {
            framesCounter = 0;
            if (currentStep < solutionPath.size() - 1) {
                currentStep++;
                puzzle = solutionPath[currentStep];
            }
            else {
                isAnimating = false;
                btnInput.set_disabled(false);
                btnShuffle.set_disabled(false);
                btnSolve.set_disabled(false);
            }
        }
    }
}

void Application::renderSidebar() {
    DrawRectangle(0, 0, sidebarWidth, screenHeight, LIGHTGRAY);

    BeginScissorMode(0, 0, sidebarWidth, screenHeight);

    btnInput.draw();
    btnShuffle.draw();
    btnSolve.draw();

    if (isSolved) {
        DrawText(lang.testedStates.c_str(), 25, 258, 17, DARKGRAY);
        DrawText(std::to_string(testedStates).c_str(), 25, 277, 20, DARKGRAY);

        if (solutionPath.empty()) {
            DrawText(lang.unsolvable.c_str(), 25, 305, 20, RED);
        }
        else if (!isAnimating) {
            std::string stepsText = lang.steps + std::to_string((int)solutionPath.size() - 1);
            DrawText(stepsText.c_str(), 25, 305, 20, DARKGRAY);
            btnShowTree.draw();
        }
    }

    if (isProcessing) {
        DrawText(lang.processing.c_str(), 25, 260, 20, DARKBLUE);
    }

    renderFlags();

    EndScissorMode();
}

void Application::renderBoard() {
    const std::vector<int>& state = puzzle.get_board();

    for (int i = 0; i < 9; i++) {
        int row = i / 3;
        int col = i % 3;
        int value = state[i];

        int x = sidebarWidth + (col * cellSize);
        int y = row * cellSize;

        if (value != 0) {
            DrawRectangle(x + 5, y + 5, cellSize - 10, cellSize - 10, DARKBLUE);
            std::string text = std::to_string(value);
            int valueTextWidth = MeasureText(text.c_str(), 60);
            DrawText(text.c_str(), x + (cellSize - valueTextWidth) / 2, y + (cellSize - 60) / 2, 60, WHITE);
        }
        else {
            DrawRectangle(x + 5, y + 5, cellSize - 10, cellSize - 10, RAYWHITE);
        }
    }
}

void Application::renderModal() {
    DrawRectangle(0, 0, screenWidth, screenHeight, Fade(BLACK, 0.8f));
    DrawRectangle(screenWidth / 2 - 200, screenHeight / 2 - 100, 400, 200, RAYWHITE);
    DrawText(lang.modalPrompt.c_str(), screenWidth / 2 - 180, screenHeight / 2 - 80, 20, DARKGRAY);
    DrawText(inputText.c_str(), screenWidth / 2 - 180, screenHeight / 2 - 20, 40, DARKBLUE);
    DrawText(lang.modalHint.c_str(), screenWidth / 2 - 180, screenHeight / 2 + 50, 15, GRAY);
}

void Application::renderFlags() {
    const int fw = 45, fh = 30;
    const int gap = 10;
    const int totalW = fw * 2 + gap;
    const int fx = (sidebarWidth - totalW) / 2;
    const int fy = screenHeight - fh - 15;

    Rectangle rectBR = { (float)fx, (float)fy, (float)fw, (float)fh };
    Rectangle rectUS = { (float)(fx + fw + gap), (float)fy, (float)fw, (float)fh };

    DrawRectangleRec(rectBR, { 0, 156, 59, 255 });
    Vector2 diamond[4] = {
        { rectBR.x + fw / 2.0f, rectBR.y + 2 },
        { rectBR.x + fw - 4,    rectBR.y + fh / 2.0f },
        { rectBR.x + fw / 2.0f, rectBR.y + fh - 2 },
        { rectBR.x + 4,         rectBR.y + fh / 2.0f }
    };
    DrawTriangle(diamond[0], diamond[3], diamond[1], { 255, 220, 0, 255 });
    DrawTriangle(diamond[3], diamond[2], diamond[1], { 255, 220, 0, 255 });
    DrawCircle((int)(rectBR.x + fw / 2), (int)(rectBR.y + fh / 2), 7, { 0, 56, 168, 255 });

    DrawRectangleRec(rectUS, { 178, 34, 52, 255 });
    for (int s = 0; s < 4; s++) {
        int sy = (int)(rectUS.y + s * (fh / 4.0f) + fh / 8.0f - 1);
        DrawRectangle((int)rectUS.x, sy, fw, 3, RAYWHITE);
    }
    DrawRectangle((int)rectUS.x, (int)rectUS.y, fw / 2, fh / 2, { 0, 40, 104, 255 });

    if (currentLang == Language::PT)
        DrawRectangleLinesEx(rectBR, 2, WHITE);
    else
        DrawRectangleLinesEx(rectUS, 2, WHITE);

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse = GetMousePosition();
        if (CheckCollisionPointRec(mouse, rectBR)) setLanguage(Language::PT);
        if (CheckCollisionPointRec(mouse, rectUS)) setLanguage(Language::EN);
    }
}

void Application::render() {
    BeginDrawing();

    if (isTreeViewOpen) {
        renderTreeView();
    }
    else {
        ClearBackground(RAYWHITE);
        renderSidebar();
        renderBoard();

        if (isModalOpen) {
            renderModal();
        }
    }

    EndDrawing();
}

void Application::renderTreeView() {
    ClearBackground(RAYWHITE);

    DrawRectangle(0, 0, screenWidth, 60, DARKBLUE);
    std::string title = lang.treeTitle + "  |  " +
                        std::to_string((int)solutionPath.size() - 1) + " " + lang.treeMoves + "  |  " +
                        std::to_string(testedStates) + " " + lang.treeStatesTested;
    DrawText(title.c_str(), 15, 20, 19, WHITE);
    btnBack.draw();

    const int miniCell = 40;
    const int boardPx = miniCell * 3;       
    const int borderPad = 4;                 
    const int labelH = 22;                   
    const int labelGap = 5;                 
    const int boardBlockH = boardPx + borderPad * 2; 
    const int arrowH = 24;                  
    const int nodeSpacing = labelH + labelGap + boardBlockH + arrowH;

    int boardX = (screenWidth - boardPx) / 2;
    int headerH = 60;
    int contentStartY = headerH + 10;
    int n = (int)solutionPath.size();

    BeginScissorMode(0, headerH, screenWidth, screenHeight - headerH);

    for (int i = 0; i < n; i++) {
        int baseY = contentStartY + i * nodeSpacing - treeScrollOffset;
        int boardBorderY = baseY + labelH + labelGap;
        int boardY = boardBorderY + borderPad;

        if (boardBorderY + boardBlockH < headerH || baseY > screenHeight) continue;

        std::string label;
        Color labelColor;
        if (i == 0) { label = lang.nodeInitial; labelColor = DARKBLUE; }
        else if (i == n - 1) { label = lang.nodeGoal; labelColor = DARKGREEN; }
        else { label = lang.nodeStep + " " + std::to_string(i); labelColor = DARKGRAY; }

        int labelW = MeasureText(label.c_str(), 18);
        DrawText(label.c_str(), (screenWidth - labelW) / 2, baseY, 18, labelColor);

        Color borderColor = (i == 0) ? DARKBLUE : (i == n - 1) ? DARKGREEN : DARKBLUE;
        DrawRectangle(boardX - borderPad, boardBorderY, boardPx + borderPad * 2, boardBlockH, borderColor);

        const auto& state = solutionPath[i].get_board();
        for (int j = 0; j < 9; j++) {
            int row = j / 3, col = j % 3;
            int val = state[j];
            int cx = boardX + col * miniCell;
            int cy = boardY + row * miniCell;

            if (val != 0) {
                Color tileColor = (i == n - 1) ? DARKGREEN : DARKBLUE;
                DrawRectangle(cx + 2, cy + 2, miniCell - 4, miniCell - 4, tileColor);
                std::string t = std::to_string(val);
                int tw = MeasureText(t.c_str(), 20);
                DrawText(t.c_str(), cx + (miniCell - tw) / 2, cy + (miniCell - 20) / 2, 20, WHITE);
            }
            else {
                DrawRectangle(cx + 2, cy + 2, miniCell - 4, miniCell - 4, RAYWHITE);
            }
        }

        if (i < n - 1) {
            int ax = screenWidth / 2;
            int ay1 = boardBorderY + boardBlockH + 2;
            int ay2 = baseY + nodeSpacing - 8;
            DrawLineEx({ (float)ax, (float)ay1 }, { (float)ax, (float)ay2 }, 2.0f, DARKGRAY);
            DrawTriangle(
                { (float)ax, (float)(ay2 + 8) },
                { (float)(ax - 7), (float)ay2 },
                { (float)(ax + 7), (float)ay2 },
                DARKGRAY
            );
        }
    }

    EndScissorMode();

    int contentTopPad = contentStartY - headerH;  
    int totalHeight = contentTopPad + (n - 1) * nodeSpacing + (labelH + labelGap + boardBlockH) + 20;
    int viewHeight = screenHeight - headerH;
    if (totalHeight > viewHeight) {
        float ratio = (float)viewHeight / totalHeight;
        int sbH = std::max(20, (int)(ratio * viewHeight));
        float scrollRatio = (float)treeScrollOffset / std::max(1, totalHeight - viewHeight);
        int sbY = headerH + (int)(scrollRatio * (viewHeight - sbH));
        DrawRectangle(screenWidth - 8, headerH, 8, viewHeight, LIGHTGRAY);
        DrawRectangle(screenWidth - 8, sbY, 8, sbH, DARKGRAY);
    }
}