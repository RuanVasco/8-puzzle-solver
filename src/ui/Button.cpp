#include <ui/Button.h>

Button::Button(float x, float y, float width, float height, const std::string& text) {
    this->bounds = { x, y, width, height };
    this->text = text;
    this->normalColor = DARKPURPLE;
    this->hoverColor = PURPLE;
    this->disabledColor = GRAY;
    this->isHovered = false;
    this->isDisabled = false;
}

bool Button::is_clicked() {
    if (isDisabled) return false;

    Vector2 mousePoint = GetMousePosition();
    this->isHovered = CheckCollisionPointRec(mousePoint, bounds);

    if (this->isHovered && IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        return true;
    }
    return false;
}

void Button::draw() const {
    Color currentColor;
    if (isDisabled) {
        currentColor = disabledColor;
    }
    else {
        currentColor = isHovered ? hoverColor : normalColor;
    }

    DrawRectangleRec(bounds, currentColor);

    int textWidth = MeasureText(text.c_str(), 30);
    Color textColor = isDisabled ? LIGHTGRAY : WHITE;
    DrawText(text.c_str(), bounds.x + (bounds.width - textWidth) / 2, bounds.y + 10, 30, textColor);
}

void Button::set_disabled(bool disabled) {
    this->isDisabled = disabled;
}