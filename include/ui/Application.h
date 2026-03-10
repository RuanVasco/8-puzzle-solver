#pragma once
#include <models/PuzzleBoard.h>
#include <models/Solver.h>
#include "Button.h"
#include <future>
#include <string>

struct LangStrings {
	std::string btnInput, btnShuffle, btnSolve, btnShowTree, btnBack;
	std::string testedStates, unsolvable, steps, processing;
	std::string modalPrompt, modalHint;
	std::string treeTitle, treeMoves, treeStatesTested;
	std::string nodeInitial, nodeGoal, nodeStep;
};

class Application {
private:
	int screenWidth;
	int screenHeight;
	int sidebarWidth;
	int cellSize;

	PuzzleBoard puzzle;
	Solver solver;

	Button btnInput;
	Button btnShuffle;
	Button btnSolve;
	Button btnShowTree;
	Button btnBack;

	std::vector<PuzzleBoard> solutionPath;
	std::future<std::vector<PuzzleBoard>> futureSolution;

	bool isAnimating;
	bool isProcessing;
	bool isSolved;
	bool isModalOpen;
	bool isTreeViewOpen;

	int currentStep;
	int framesCounter;
	int testedStates;
	int treeScrollOffset;

	std::string inputText;

	enum class Language { PT, EN };
	Language currentLang;
	LangStrings lang;
	static const LangStrings LANG_PT;
	static const LangStrings LANG_EN;

	void setLanguage(Language l);
	void handleEvents();
	void handleModalInput();
	void pasteFromClipboard();
	void updateLogic();
	void render();
	void renderSidebar();
	void renderFlags();
	void renderBoard();
	void renderModal();
	void renderTreeView();

public:
	Application();
	~Application();
	void run();
};
