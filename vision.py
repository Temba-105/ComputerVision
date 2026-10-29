import cv2
import time
from ultralytics import YOLO

print("Загружаем модель YOLO...")
model = YOLO("yolo26n.pt")
print("Модель загружена!")


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)


camera.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 312)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("Ошибка: не удалось открыть камеру")
    raise SystemExit

print("Камера запущена!")
print("Для выхода нажми Q")

while True:
    success, frame = camera.read()

    if not success:
        print("Не удалось получить кадр")
        break


    frame = cv2.resize(frame, (416, 312))

    start = time.time()


    results = model.predict(
        frame,
        imgsz=256,
        conf=0.35,
        verbose=False,
        device="cpu"
    )


    image = results[0].plot()


    elapsed = time.time() - start
    fps = 1 / elapsed if elapsed > 0 else 0

    cv2.putText(
        image,
        f"Detection: {fps:.1f} FPS",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    cv2.imshow("YOLO - Object Detection", image)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Программа завершена")