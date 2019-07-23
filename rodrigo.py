"""
Es una version simplificada de lo que creo que necesitas. Espero que puedas sacarle algun provecho.

Falto:
1.- Enderezar la imagen en caso de estar inclinada.
2.- Agregar margenes superior e inferior para que el encuadre del recorte no sea tan justo.
"""
import sys

sys.path.append('C:/Python36/Lib/site-packages')

import cv2
import numpy as np

imagen = cv2.imread("test.png")
alto, ancho, canales = imagen.shape
contador = 0

# Establece el espacio (desde el el borde izquierdo de la hoja) para considerar como margen 
margen_del_texto = 40

# Evita que identifique bordes de tablas o figuras, como si fueran "marcas"
altura_maxima_de_la_marca = 50

# Puesto que la imagen esta y podria estar inclinada, los extremos de la linea (vertical) podrian estar distantes
anchura_maxima_de_la_marca = 10

# Transforma a escala de grises y luego encuentra los bordes
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

lineas = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=15, maxLineGap=10)

for linea in lineas:
    x1, y1, x2, y2 = linea[0]
    # Controla que se busquen las lineas solo dentro del margen, y las medidas especificadas
    if x1 < margen_del_texto and (abs(x2 - x1) < anchura_maxima_de_la_marca) and abs(y2 - y1) < altura_maxima_de_la_marca:
        contador += 1
        cv2.line(imagen, (x1,y1), (x2,y2), (0, 0, 255), 1, cv2.LINE_AA)
        # Para ver los recuadros en donde encontro marcas
        cv2.rectangle(imagen, (0, y1), (ancho, y2), (255, 0, 0), 1) 
        recorte = imagen[y1:y2, 1:ancho-1]
        if len(recorte) > 0:
            cv2.imshow("recorte", recorte)
            cv2.imwrite("recorte_"+str(contador)+".png", recorte)
            cv2.destroyWindow("recorte")


cv2.imshow('Marcas detectadas', imagen)



cv2.waitKey(0)