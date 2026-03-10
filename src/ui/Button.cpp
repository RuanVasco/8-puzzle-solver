#include <ui/Button.h>

Button::Button(float x, float y, float width, float height, const std::string& text,
    Color normalColor, Color hoverColor, Color disabledColor) {
    this->bounds = { x, y, width, height };
    this->text = text;
    this->normalColor = normalColor;
    this->hoverColor = hoverColor;
    this->disabledColor = disabledColor;
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
    Color currentColor = isDisabled ? disabledColor : (isHovered ? hoverColor : normalColor);
    DrawRectangleRec(bounds, currentColor);

    float maxWidth = bounds.width - 20;
    std::string displayText = text;
    int fontSize = 20;

    if (MeasureText(displayText.c_str(), fontSize) > maxWidth) {
        while (!displayText.empty() && (MeasureText((displayText + "...").c_str(), fontSize) > maxWidth)) {
            displayText.pop_back(); 
        }
        displayText += "...";
    }

    int textWidth = MeasureText(displayText.c_str(), fontSize);
    Color textColor = isDisabled ? LIGHTGRAY : WHITE;

    DrawText(displayText.c_str(),
        bounds.x + (bounds.width - textWidth) / 2,
        bounds.y + (bounds.height - fontSize) / 2,
        fontSize,
        textColor);
}

void Button::set_disabled(bool disabled) {
    this->isDisabled = disabled;
}

void Button::set_text(const std::string& t) {
    this->text = t;
}

bool Button::get_is_hovered() const {
    return this->isHovered;
}

void Button::update() {
    if (isDisabled) {
        this->isHovered = false;
        return;
    }
    Vector2 mousePoint = GetMousePosition();
    this->isHovered = CheckCollisionPointRec(mousePoint, bounds);
}