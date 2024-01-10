import cv2
import face_recognition
import csv
import ast
import numpy as np
from numba import jit, cuda
from timeit import default_timer as timer

# Archivo CSV con la información de las imágenes faciales
csv_file = 'data/facial_info.csv'

# Cargar la información de las imágenes faciales desde el archivo CSV
try:
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        facial_info = [row for row in reader]
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    exit()

# Convertir las cadenas de codificación facial a listas de floats
for info in facial_info:
    try:
        info['face_encoding'] = ast.literal_eval(info['face_encoding'])
    except Exception as e:
        print(f"Error al convertir la codificación facial: {e}")
        exit()

# Función para comparar características faciales utilizando CUDA
@jit
def compare_faces_cuda(ref_encoding, face_encoding):
    match = face_recognition.compare_faces([ref_encoding], face_encoding)[0]
    return match

@jit
def create_location(frame):
    return face_recognition.face_locations(frame)

@jit
def create_encoder(frame, face_locations):
    return face_recognition.face_encodings(frame, [face_locations[0]])[0]

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Captura un frame de la cámara
    re, frame = cap.read()

    # Detectar rostros en la imagen
    face_locations = create_location(frame)

    if face_locations:
        # Obtener las características faciales de la región facial
        face_encoding = create_encoder(frame, face_locations)

        # Comparar las características faciales con cada imagen almacenada utilizando CUDA
        for info in facial_info:
            try:
                ref_encoding = np.array(info['face_encoding'], dtype=np.float64)

                # Comparar las características faciales utilizando CUDA
                match = compare_faces_cuda(ref_encoding, face_encoding)

                # Imprimir el resultado
                if match:
                    # Dibujar un rectángulo alrededor del rostro detectado
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Mostrar el nombre de la persona encima del rectángulo
                    nombre_persona = info['nombre_persona']
                    cv2.putText(frame, f"Persona: {nombre_persona}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    break

            except Exception as e:
                print(f"Error al comparar características faciales: {e}")

    # Mostrar la imagen con los rostros detectados
    cv2.imshow('Comparación Facial en Tiempo Real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()















"""
import cv2
import face_recognition
import csv
import ast
import numpy as np
#import dlib
#dlib.DLIB_USE_CUDA = True


# Archivo CSV con la información de las imágenes faciales
csv_file = 'data/facial_info.csv'

# Cargar la información de las imágenes faciales desde el archivo CSV
try:
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        facial_info = [row for row in reader]
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    exit()

# Convertir las cadenas de codificación facial a listas de floats
for info in facial_info:
    try:
        info['face_encoding'] = ast.literal_eval(info['face_encoding'])
    except Exception as e:
        print(f"Error al convertir la codificación facial: {e}")
        exit()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Captura un frame de la cámara
    re, frame = cap.read()


    # Detectar rostros en la imagen
    face_locations = face_recognition.face_locations(frame)
    
    if face_locations:
        # Obtener las características faciales de la región facial
        face_encoding = face_recognition.face_encodings(frame, [face_locations[0]])[0]

        # Comparar las características faciales con cada imagen almacenada
        for info in facial_info:
            try:
                ref_encoding = np.array(info['face_encoding'], dtype=np.float64)

                # Comparar las características faciales
                match = face_recognition.compare_faces([ref_encoding], face_encoding)[0]

                

                # Imprimir el resultado
                if match:
                    # Dibujar un rectángulo alrededor del rostro detectado
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Mostrar el nombre de la persona encima del rectángulo
                    nombre_persona = info['nombre_persona']
                    cv2.putText(frame, f"Persona: {nombre_persona}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    break

            except Exception as e:
                print(f"Error al comparar características faciales: {e}")

    # Mostrar la imagen con los rostros detectados
    cv2.imshow('Comparación Facial en Tiempo Real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
"""