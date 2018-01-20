import cv2

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    # cv2.rectangle(img, topLeftPoint, bottomRightPoint, rgbColor, lineWidth)
    cv2.rectangle(img, (x, y), (x + w, y + h), (139,64,39), 2)

def draw_text(img, text, x, y):
    # cv2.putText(img, text, startPoint, font, fontSize, rgbColor, lineWidth)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (139,64,39), 2)
