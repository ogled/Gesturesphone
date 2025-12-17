import cv2
import mediapipe as mp

# Входной и выходной файлы
INPUT = "input.jpg"
OUTPUT = "output.jpg"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Красный цвет
RED = (0, 0, 255)
GREEN = (0, 102, 51)
line_drawing = mp_draw.DrawingSpec(color=GREEN, thickness=5)
point_drawing = mp_draw.DrawingSpec(color=RED, thickness=15)

# Загружаем изображение
image = cv2.imread(INPUT)

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

    # Обработка
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Рисуем найденные landmark’ы
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                image,
                hand,
                mp_hands.HAND_CONNECTIONS,
                point_drawing,
                line_drawing
            )

# Сохраняем результат
cv2.imwrite(OUTPUT, image)
print("Готово! Сохранено в", OUTPUT)
