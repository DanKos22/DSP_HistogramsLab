# -*- coding: utf-8 -*-
"""
@author: Dan Koskiranta
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read original image, returns a numpy.ndarray (h, w, channels)
image = cv.imread('flamingo-3309628_1920.jpg', cv.IMREAD_COLOR)

# Get dimensions of the image
dim_orig = image.shape
height, width, channels = image.shape

# Scale, preserving aspect ratio
def scale(image, scale_by):
    w = int(image.shape[1] * scale_by)
    h = int(image.shape[0] * scale_by)
    dim = (w, h)
    image_scaled = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    #cv.imshow('', image_scaled)
    #cv.waitKey(0)
    return image_scaled

scale_by = .4
image = scale(image, scale_by)
cv.imshow('Scaled Image', image)
cv.waitKey(0)

# RGB Histograms using opencv
fig1 = plt.figure(figsize = (14, 10))
ax = fig1.add_subplot(111)
color = ('b','g','r')
for i, c in enumerate(color):
    hist = cv.calcHist([image],[i],None,[256],[0,256])
    ax.plot(hist, color=c)
    ax.set_xlim([0, 256])
ax.set_title("RGB Histogram, using OpenCV")



# Histograms using matplotlib
fig2 = plt.figure(figsize = (18, 8))
ax = fig2.add_subplot(121)
#ax.set_ylim(0, 10000)
#ax2.tick_params(axis='both', which='both', labelsize=16, length=5, pad=10)
ax.hist(image.ravel(), 256, [0,256]); 
ax.set_title('Original image histogram')#, fontsize=22, color='k',pad=10)
# Grayscale the input image before equalisation
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_eq = cv.equalizeHist(image_gray)
stack = np.hstack((image_gray, image_eq))
ax  = fig2.add_subplot(122)
#ax.set_ylim(0, 0000)
#ax.tick_params(axis='both', which='both', labelsize=16, length=5, pad=10)
# Add code to repeat histogram on equalised image (copy & adapt call above to ax.hist)
ax.hist(image_eq.ravel(), 256, [0,256]);  
ax.set_title('Equalised image histogram')#, fontsize=22, color='k',pad=10)
cv.imshow('Original greyscale image and equalised image', stack)
cv.waitKey(0)

'''
# Now try adaptive equalisationL CLAHE Contrast Limited Adaptive Histogram Equalisation
# Add clode to create CLAHE, use cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# Add code to apply CLAHE, use clahe.apply on grayscale image
stack = np.hstack((image_gray, image_eq_clahe)) # Stacking images side-by-side
cv.imshow('Original greyscale image and CLAHE adaptive equalisation image', stack)
cv.waitKey(0)

fig3 = plt.figure(figsize = (20, 10))
ax = fig3.add_subplot(121)
#ax.set_ylim(0, 30000)
#ax.tick_params(axis='both', which='both', labelsize=16, length=5, pad=10)
ax.hist(image.ravel(),256,[0,256]); 
ax.set_title('Original image histogram')#, fontsize=22, color='k',pad=10)
ax = fig3.add_subplot(122)
#ax.tick_params(axis='both', which='both', labelsize=16, length=5, pad=10)
ax.hist(image_eq_clahe.ravel(),256,[0,256]);
#ax.set_ylim(0, 30000)
ax.set_title('CLAHE equalised image histogram')#, fontsize=22, color='k', pad=10)
plt.subplots_adjust(hspace = 0.5)
cv.waitKey(0)
'''
cv.destroyAllWindows()
