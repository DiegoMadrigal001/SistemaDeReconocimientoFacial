import cv2
import csv
import os
import time
import face_recognition

# Inicializa la cámara
cap = cv2.VideoCapture(0)

# Inicializa el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Número de muestras a tomar
num_samples = 10
sample_count = 0

# Directorio para almacenar las imágenes faciales
save_directory = 'data/faces/'

# Verificar y crear el directorio si no existe
os.makedirs(save_directory, exist_ok=True)

# Archivo CSV para almacenar la información de las imágenes faciales
csv_file = 'data/facial_info.csv'

# Lista para almacenar la información de las imágenes faciales
facial_info = []

while True:
    # Captura un frame de la cámara
    ret, frame = cap.read()

    # Convierte a escala de grises para el detector de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Dibuja un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Calcula la codificación facial
        face_encoding = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]

        # Almacena la información de la imagen facial en la lista
        nombre_persona = "Diego Madrigal"
        facial_info.append({'nombre_persona': nombre_persona, 'face_encoding': face_encoding.tolist()})

        # Incrementa el contador de muestras
        sample_count += 1

        # Espera 2 segundos antes de capturar la siguiente imagen
        time.sleep(0.5)

    # Si se han tomado suficientes muestras, sale del bucle
    if sample_count == num_samples:
        break

    # Muestra el frame con el rectángulo del rostro
    cv2.imshow('Capturando Rostros', frame)

    # Espera la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guarda la información de las codificaciones faciales en el archivo CSV
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = ['nombre_persona', 'face_encoding']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for info in facial_info:
        writer.writerow(info)

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()












































"""



import cv2
import csv
import os
import time
import face_recognition

# Inicializa la cámara
cap = cv2.VideoCapture(0)

# Inicializa el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Número de muestras a tomar
num_samples = 10
sample_count = 0

# Directorio para almacenar las imágenes faciales
save_directory = 'data/faces/'

# Verificar y crear el directorio si no existe
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Archivo CSV para almacenar la información de las imágenes faciales
csv_file = 'data/facial_info.csv'

# Lista para almacenar la información de las imágenes faciales
facial_info = []

while True:
    # Captura un frame de la cámara
    ret, frame = cap.read()

    # Convierte a escala de grises para el detector de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Dibuja un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Guarda la imagen facial en el directorio
        try:
            face_roi = gray[y:y+h, x:x+w]
            file_path = save_directory + f"face_{sample_count}.png"
            cv2.imwrite(file_path, face_roi)

            # Solicita al usuario ingresar el nombre de la persona
            nombre_persona = "Diego Madrigal"

            # Almacena la información de la imagen facial en la lista
            facial_info.append({'file_path': file_path, 'nombre_persona': nombre_persona, 'top': y, 'right': x+w, 'bottom': y+h, 'left': x})

            # Incrementa el contador de muestras
            sample_count += 1

            # Espera 2 segundos antes de capturar la siguiente imagen
            time.sleep(0.5)

        except Exception as e:
            print(f"Error al guardar la imagen facial: {e}")
            continue

    # Si se han tomado suficientes muestras, sale del bucle
    if sample_count == num_samples:
        break

    # Muestra el frame con el rectángulo del rostro
    cv2.imshow('Capturando Rostros', frame)

    # Espera la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guarda la información de las imágenes faciales en el archivo CSV
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = ['file_path', 'nombre_persona', 'top', 'right', 'bottom', 'left', 'face_encoding']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for info in facial_info:
        try:
            # Cargar la imagen de referencia para obtener la codificación facial
            ref_image = face_recognition.load_image_file(info['file_path'])
            ref_encoding = face_recognition.face_encodings(ref_image)

            if ref_encoding:
                info['face_encoding'] = ref_encoding[0].tolist()
                writer.writerow(info)
            else:
                print(f"Error: No se pudo encontrar un rostro en la región de la imagen {info['file_path']}.")

        except Exception as e:
            print(f"Error al cargar la imagen de referencia: {e}")

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()
"""