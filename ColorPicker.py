#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:39:28 2019

@author: CT
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:56:32 2019

@author: CT
"""

import csv
import os
import sys
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from random import shuffle
from tqdm import tqdm
import requests


# =============================================================================
#
# Observaciones: si la imagen tiene una sombra que su superficie es mayor del 40% (approx)
# de la superficie de la piedra el algoritmo la identificara como una piedra negra
#
# Es imperativo eliminar la sombra para generar resultados mas certeros
# =============================================================================

# =============================================================================
# variable colors declarations
# COMVERTIRLO EN DICCIONARIO
# =============================================================================

blck = [5, 5, 5]  # 2 negro            #T4
aqua = [0, 204, 204]  # 1 aqua
prpl = [60, 40, 80]  # 0 morado    147,112,219
ylw = [235, 235, 100]  # 3 amarillo  255,191,0
grn = [34, 139, 34]  # 4 verde
red = [148, 28, 21]  # 5 rojo     #t4
gray = [0, 0, 0]  # 6 gris [220,220,220]
blue = [40, 95, 155]  # 7 azul        error t5 error t14
brwn = [139, 69, 19]  # 8 cafe      #t12  #t3  39,69,19  error t4
whte = [254, 254, 254]  # 9 blanco

colors = []
colors.append(blck)
colors.append(aqua)
colors.append(prpl)
colors.append(ylw)
colors.append(grn)
colors.append(red)
colors.append(gray)
colors.append(blue)
colors.append(brwn)
colors.append(whte)

colorlist = ['black', 'aqua', 'purple', 'yellow', 'green', 'red', 'gray', 'blue', 'brown', 'white']
titles = ['BGR', 'RGB', 'Grayscale', 'Gaussian Blur', 'Otsu Thresholding', 'Filled', 'Imagen Final + Sombra',
          'last color blur']

# =============================================================================
# tiempo de ejecucion, conclusion al final
# =============================================================================

start = time.time()
# Read image

# =============================================================================
#  Entrada: ubicacion de la foto y directorio
# =============================================================================

dirpath = os.getcwd()
foldername = os.path.basename(dirpath)
# print(dirpath)
# print(foldername)
# for i in range(17):
#    pic = str(i+1)
#
# img_string = "t7.jpg"

img_string = sys.argv[1]
imgname = sys.argv[2]
im_bgr = cv2.imread(img_string)  # BGR
#    plt.imshow(im_col)
#    plt.title("BGR")
#    plt.show()
b, g, r = cv2.split(im_bgr)
im_col = cv2.merge([r, g, b])  # RGB

#    plt.imshow(im_col)
#    plt.title("RGB")
#    plt.show()
#

im_in = cv2.imread(img_string, cv2.IMREAD_GRAYSCALE);  # black&white

#    plt.imshow(im_in)
#    plt.title("Grayscale")
#    plt.show()
#
blur = cv2.medianBlur(im_in, 7)
#    plt.imshow(blur)
#    plt.title("Med Blur")
#    plt.show()

blurg = cv2.GaussianBlur(im_in, (15, 15), 0)
#    plt.imshow(blurg)
#    plt.title("Gaussian Blur")
#    plt.show()
#
#


ret3, im_th = cv2.threshold(blurg, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # t4 la detecta negra t5 detecta como azul t6 como morada t2 como negra

# =============================================================================
# th, im_th = cv2.threshold(im_in, 150, 255, cv2.THRESH_BINARY_INV); #150/255 no detecta t13 (citrina brillante)
#
#
# usando valores estaticos no detectaba piedras    t13                                                                #240/255 si detecta a t13
#
# plt.imshow(th3)
# plt.show()
#
#
# =============================================================================

#    plt.imshow(im_th)
#    plt.title("Threshold OTSU")
#    plt.show()

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

#    plt.imshow(im_out)
#    plt.title("Filled")
#    plt.show()
#
im_end = cv2.cvtColor(~im_out, cv2.COLOR_GRAY2RGB) | im_col
n_white_pix = np.sum(im_end == 255)

#    plt.imshow(im_end)
#    plt.title("Imagen Final + Sombra")
#    plt.show()

Lblur = cv2.medianBlur(im_end, 55)  # anotar

#    plt.imshow(Lblur)
#    plt.title("LAST Blur")
#    plt.show()


images = [im_bgr, im_col, im_in, blurg, im_th, im_out, im_end, Lblur]
# for i in range(8):
#     plt.subplot(4, 2, i + 1), plt.imshow(images[i])
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

# =============================================================================
#  Intento de remover sombra
# =============================================================================

# ret,th1 = cv2.threshold(cv2.cvtColor(im_end, cv2.COLOR_RGB2GRAY),127,255,cv2.THRESH_BINARY_INV)
# th2 = cv2.adaptiveThreshold(cv2.cvtColor(im_end, cv2.COLOR_RGB2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY_INV,3,2)
# th3 = cv2.adaptiveThreshold(cv2.cvtColor(im_end, cv2.COLOR_RGB2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv2.THRESH_BINARY_INV,3,2)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [im_end, th1, th2, th3]
# for i in range(4):
#    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
# plt.show()
#
#
# im_floodfill = th3.copy()
#
#
#
## Mask used to flood filling.
## Notice the size needs to be 2 pixels than the image.
# h, w = im_end.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)
#
## Floodfill from point (0, 0)
# cv2.floodFill(im_floodfill, mask, (int(h/2),int(w/2)), 255, flags=4);
#
## Invert floodfilled image
# im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#
#
#
## Combine the two images to get the foreground.
# im_out = th3 | im_floodfill_inv
#
# plt.imshow(im_out)
# plt.title("Filled")
# plt.show()
#
# im_SS = cv2.cvtColor(~im_out,cv2.COLOR_GRAY2RGB) | im_col
#
# plt.imshow(im_SS)
# plt.title("Imagen Final + Sin Sombra")
# plt.show()
#

# =============================================================================
# #mask for  the color
# mask = np.zeros(im_end.shape[:2], np.uint8)
# mask[200:300, 350:450] = 255 #primeros valores son X, segundos son Y
# masked_img = cv2.bitwise_and(im_end,im_end,mask = mask)
#
# # Calculate histogram with mask and without mask
# # Check third argument for mask
# hist_full = cv2.calcHist([im_end],[0],None,[255],[1,255])   #se tienen que poner 255 en vez de 256
# hist_mask = cv2.calcHist([im_end],[0],mask,[255],[1,255])   # para eliminar el 255,255,255
#                                                             # que es el blanco que se relleno
# plt.subplot(221), plt.imshow(im_end, 'gray')
# plt.subplot(222), plt.imshow(mask,'gray')
# plt.subplot(223), plt.imshow(masked_img, 'gray')
# plt.subplot(224), plt.plot(hist_mask)
# plt.xlim([0,256])
#
# plt.show()
# =============================================================================


# print(type(colorcount))

# =============================================================================
# HACEMOS 3 HISTOGRAMAS (BGR) DE UN CUADRANTE ESPECIFICO Y LOS GUARDAMOS EN UNA LISTA
# =============================================================================

colorcount = []  # es una lista

# plt.subplot(121),plt.imshow(im_end),plt.title("End Result")
# plt.subplot(122),plt.imshow(Lblur),plt.title("Blurred image")
# plt.show()
#

color = ('r', 'g', 'b')
for i, col in enumerate(color):
    histr = cv2.calcHist([Lblur], [i], None, [255], [0, 255])  # Cambiar mask a None y None a Mask
    colorcount.insert(i, histr)  # se agrega a cada elemento de la lista (0,1,2) un array del histograma
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.title("Histogram")
# plt.show()

# print("colorcount")
# print(type(colorcount))
# print(len(colorcount))
# print(colorcount[0])
# print(colorcount[1])
# print(colorcount[2])


# =============================================================================
# BUSCA EL VALOR MAS GRANDE DE RGB
# =============================================================================

highred = max(colorcount[0])  # Red
highgreen = max(colorcount[1])  # Green
highblue = max(colorcount[2])  # Blue
# print("hred") #es un array de 2 valores, primero el color segundo un 0 NO SE PORQUE
# print(highred[0])
# print(type(highred))

reds = colorcount[0].flatten()
greens = colorcount[1].flatten()
blues = colorcount[2].flatten()

highred = max(colorcount[0])  # Red
reds.sort()
secondred = reds[-2]
highgreen = max(colorcount[1])  # Green
greens.sort()
secondgreen = greens[-2]
highblue = max(colorcount[2])  # Blue
blues.sort()
secondblue = blues[-2]

# =============================================================================
# IDENTIFICA CUAL EL INDEX (0-255) CON EL VALOR MAS ALTO Y EL SEGUNDO MAS ALTO
# =============================================================================


indexred = np.where(colorcount[0] == highred[0])
indexgreen = np.where(colorcount[1] == highgreen[0])
indexblue = np.where(colorcount[2] == highblue[0])
secondindexred = np.where(colorcount[0] == secondred)
secondindexgreen = np.where(colorcount[1] == secondgreen)
secondindexblue = np.where(colorcount[2] == secondblue)
# print("indexred")
# print(indexred)
# print(type(indexred))
# print(type(indexred[0][0]))

# =============================================================================
# GUARDAMOS L0S 2 RGB EN UNA LISTA
# =============================================================================

RGB = []
RGB.append(indexred[0][0])
RGB.append(indexgreen[0][0])
RGB.append(indexblue[0][0])
# print("Main:", end='')
# print(RGB)

RGB2 = []
RGB2.append(secondindexred[0][0])
RGB2.append(secondindexgreen[0][0])
RGB2.append(secondindexblue[0][0])
# print("Second:", end='')
# print(RGB2)

# print(type(RGB))


# =============================================================================
# BUSCAMOS CUAL ES EL COLOR QUE MAS SE ACERCA CON LA MENOR DISTANCIA EUCLIDIANA USANDO MATH
#
# Alternativa para el futuro: usar cv2.inrange()

# =============================================================================
#

distance = 999999999999  # punto de referencia para encontrar mas chicos
index = 0

for i in colors:
    distanceTest = math.sqrt((RGB[0] - i[0]) ** 2 + (RGB[1] - i[1]) ** 2 + (RGB[2] - i[2]) ** 2)
    if distanceTest < distance:
        distance = distanceTest
        index = colors.index(i)

# print("distance:",distance)
# print("index:",index)

# =============================================================================
#     DELIMITAR SI EL COLOR ES GRIS
# =============================================================================
#
#   ni idea como
#
#
#
#
# =============================================================================
# ANALIZAR SI EL SEGUNDO PICO DE COLOR ES COLOR O NEGRO
# =============================================================================

if (index == 0):

    distance = 999999999999  # punto de referencia para encontrar mas chicos
    index2 = 0

    for i in colors:
        distanceTest = math.sqrt((RGB2[0] - i[0]) ** 2 + (RGB2[1] - i[1]) ** 2 + (RGB2[2] - i[2]) ** 2)
        if distanceTest < distance:
            distance = distanceTest
            index2 = colors.index(i)

    #    print("distance:",distance)
    #    print("index:",index2)

    if (index == index2):
        #        print("distance:",distance)
        #        print("index:",colorlist[index])
        color = colorlist[index]
    else:
        #        print("distance:",distance)
        #        print("index:",colorlist[index2])
        color = colorlist[index2]

else:
    #    print("distance:",distance)
    #    print("index:",colorlist[index])
    color = colorlist[index]

# =============================================================================
#   Rugosidad de piedra
# =============================================================================

im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imwrite('bw_image.png', im_bw)
bg_white_pix = np.sum(im_end[:, :, 2] == 255)
total_white_pix = np.sum(im_bw == 255)
true_white_pix = total_white_pix - bg_white_pix
n_black_pix = np.sum(im_bw == 0)

rugosidad = true_white_pix / n_black_pix

# plt.imshow(im_bw)
# plt.show()

# =============================================================================
# Guardar nueva imagen Crear CSV con datos
# =============================================================================

cv2.imwrite(imgname, cv2.cvtColor(im_end, cv2.COLOR_RGB2BGR))

csvData = [["Color", "Rugosidad"], [color, rugosidad]]
with open('ML.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()

# Display images.
# cv2.imshow("Threshold ", im_th)
# cv2.imshow("Floodfill", im_floodfill)
# cv2.imshow("Inv Floodfille", im_floodfill_inv)
# cv2.imshow("Foreground", im_out)
###cv2.imshow("end product", im_end)



# =============================================================================
# SEND IMG & TAG TO MODEL
# =============================================================================

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/model_merged:predict', data=im_end, headers=headers)
predictions = json.loads(json_response.text)['predictions']

show(0, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
  class_names[np.argmax(predictions[0])], test_labels[0], class_names[np.argmax(predictions[0])], test_labels[0]))


end = time.time()
# print("time:",end - start)
###cv2.waitKey(0)
# =============================================================================
# OUTPUT CSV TAGS TXT
# =============================================================================
