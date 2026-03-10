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
	Button(float x, float y, float width, float height, const std::string& text,
		Color normalColor = DARKPURPLE,
		Color hoverColor = PURPLE,
		Color disabledColor = GRAY);
	bool is_clicked();
	void draw() const;
	void set_disabled(bool disabled);
	void set_text(const std::string& text);
	bool get_is_hovered() const;
	void update();
};