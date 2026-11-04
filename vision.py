import cv2
import time
import json
import os
from ultralytics import YOLO

print("Загружаем модель YOLO...")
model = YOLO("yolo26n.pt")
print("Модель загружена!")

folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder, "names_ru.json")

with open(file_path, "r", encoding="utf-8") as file:
    names_ru = json.load(file)

names_ru = {int(key): value for key, value in names_ru.items()}

print("Названий загружено:", len(names_ru))

model_names = names_ru

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 30)

if not camera.isOpened():
    raise RuntimeError("Камера не открылась!")

print("Камера запущена! Нажми Q для выхода.")

fps_values = []

while True:
    success, frame = camera.read()

    if not success:
        print("Ошибка получения изображения")
        break

    start = time.time()

    results = model.predict(
        frame,
        imgsz=960,
        conf=0.4,
        verbose=False,
        device="cpu"
    )

    results[0].names = names_ru


    image = results[0].plot().copy()

    elapsed = time.time() - start
    fps = 1 / elapsed if elapsed > 0 else 0


    fps_values.append(fps)

    if len(fps_values) > 20:
        fps_values.pop(0)

    average_fps = sum(fps_values) / len(fps_values)

    cv2.putText(
        image,
        "FPS: " + str(round(average_fps, 1)),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO - Распознавание объектов", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
