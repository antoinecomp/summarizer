import sys
import cv2
import numpy as np
import cmath
from matplotlib import pyplot as plt

import pytesseract

from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

import os

##################################################################
# 1. Lo que ya tenías
imagen = cv2.imread("test.png")
alto, ancho, canales = imagen.shape
contador = 0

# Establece el espacio (desde el el borde izquierdo de la hoja) para considerar como margen 
margen = 40

# Evita que identifique bordes de tablas o figuras, como si fueran "marcas"
alto_max = 50

# Puesto que la imagen esta y podria estar inclinada, los extremos de la linea (vertical) podrian estar distantes
ancho_max = 10

# Transforma a escala de grises y luego encuentra los bordes
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

lineas = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=15, maxLineGap=10)

#########################################################################
# 1b. El filtrado de líneas válidas
def linea_valida(linea, margen, ancho_max, alto_max):
    x1, y1, x2, y2 = linea
    return x1 < margen and abs(x2 - x1) < ancho_max and abs(y2 - y1) < alto_max

validas = [linea[0] for linea in lineas if linea_valida(linea[0], margen, ancho_max, alto_max)]


##########################################################################
# 2. Detección del ángulo girado
def detecta_giro(lineas):
  angulos = []
  for linea in lineas:
    x1, y1, x2, y2 = linea
    vector = complex(x1, y1) - complex(x2, y2)
    angulo = cmath.phase(vector)
    if angulo<0: 
      angulos.append(angulo)
  return np.mean(angulos)

angulo = detecta_giro(validas)

# Convertir a grados
angulo = np.rad2deg(angulo + np.pi/2)

##########################################################################
# 3. Girar la imagen para enderezarla
(h, w) = imagen.shape[:2]
centro = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(centro, angulo, 1.0)

girada = cv2.warpAffine(imagen, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

##########################################################################
# 4. Repetir la transformada de Hough, ahora sobre la imagen rectificada
gray = cv2.cvtColor(girada, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
lineas = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=15, maxLineGap=10)

# Debido al giro hay menos margen
margen = 35
validas = [linea[0] for linea in lineas if linea_valida(linea[0], margen, ancho_max, alto_max)]

##########################################################################
# 5. Extraer los trozos en la girada
for x1, y1, x2, y2 in validas:
    contador += 1
    cv2.line(girada, (x1,y1), (x2,y2), (0, 0, 255), 1, cv2.LINE_AA)
    # Para ver los recuadros en donde encontro marcas
    cv2.rectangle(girada, (0, y1), (ancho, y2), (255, 0, 0), 1) 
    recorte = girada[y1:y2, 1:ancho-1]
    if len(recorte) > 0:
    	#on doit extraire le texte de recorte
        #cv2.imshow("recorte", recorte)
        cv2.imwrite("recorte_"+str(contador)+".png", recorte)
        text = pytesseract.image_to_string(Image.open("recorte_"+str(contador)+".png"))
        os.remove("recorte_"+str(contador)+".png")
        #cv2.destroyWindow("recorte")
        with open('resume.txt', 'a+') as f:
            print('***:', text, file=f) 

cv2.imwrite("final.png", girada)
plt.imshow(cv2.cvtColor(girada, cv2.COLOR_BGR2RGB))
plt.gcf().set_size_inches((6, 10))
plt.axis('off')