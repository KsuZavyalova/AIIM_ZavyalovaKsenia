import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from google.colab import files
from oauth2client.client import GoogleCredentials

# Авторизация Google Drive
drive.mount('/content/drive')
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# диапазон цвета объекта (в формате HSV)
lower_color = np.array([60, 100, 30])  # нижний предел цвета
upper_color = np.array([80, 255, 100])  # верхний предел цвета

# Определим диапазон цвета вывески (помехи) (в формате HSV)
color_sign = np.uint8([[[71, 167, 68]]])  # BGR цвет #44A747
hsv_sign = cv2.cvtColor(color_sign, cv2.COLOR_BGR2HSV)
lower_sign = np.array([hsv_sign[0][0][0] - 10, hsv_sign[0][0][1] - 30, hsv_sign[0][0][2] - 30])
upper_sign = np.array([hsv_sign[0][0][0] + 10, hsv_sign[0][0][1] + 30, hsv_sign[0][0][2] + 30])

points = []

# Инициализация объекта класса VideoWriter, используемого для записи кадров в файл
output_file = 'result.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
cap = cv2.VideoCapture("/content/drive/My Drive/VID_20241001_204736.mp4") 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 25
writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем кадр в пространство цветов HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Применяем цветовую фильтрацию для объекта
    mask_object = cv2.inRange(hsv, lower_color, upper_color)

    # Применяем цветовую фильтрацию для вывески
    mask_sign = cv2.inRange(hsv, lower_sign, upper_sign)

    # Исключаем вывеску из маски объекта
    mask = cv2.bitwise_and(mask_object, cv2.bitwise_not(mask_sign))

    # Находим контуры в маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Находим наибольший контур
        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            points.append((cx, cy))

    for point in points:
        cv2.drawMarker(frame, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    # Записываем кадр в видеофайл
    writer.write(frame)

    # Отображаем кадр с нарисованным путем (если нужно)
    # cv2_imshow(frame)

# Освобождаем ресурсы
cap.release()
writer.release()
cv2.destroyAllWindows()

file_drive = drive.CreateFile({'title': output_file})
file_drive.SetContentFile(output_file)
file_drive.Upload()

print(f'Файл {output_file} загружен на Google Drive.')
