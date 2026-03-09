#pragma once
#include "raylib.h"
#include <string>

class Button {
private:
	Rectangle bounds;
	std::string text;
	Color normalColor;
	Color hoverColor;
	Color disabledColor;
	bool isHovered;
	bool isDisabled;
public:
	Button(float x, float y, float width, float height, const std::string& text);
	bool is_clicked();
	void draw() const;
	void set_disabled(bool disabled);
};