#%%

# Some imports
import sys,os

sys.path.append('../../project_files/scripts/')
import segment_aerial_images as imanip

import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
# Loaded a set of images
root_dir = "../../project_files/data/training/"

image_dir = root_dir + "images/"
grt_dir = root_dir + "groundtruth/"
files = os.listdir(image_dir)

file = "satImage_056.png"#files[0]

print(file)
img = cv2.imread(image_dir + file)
grt = cv2.imread(grt_dir + file)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# plt.imshow(img)
# plt.show()
# plt.imshow(grt)
# plt.show()

#%%
# Binarize 

def binarize(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    lowerb = np.array([50, 50, 50])
    upperb = np.array([130, 130, 130])
    binary = cv2.inRange(blurred, lowerb, upperb)
    return binary

binary = binarize(img)
plt.imshow(binarize(img))

#%%
# 

def auto_canny(image, sigma=0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

	# compute the median of the single channel pixel intensities
    v = np.median(blurred)
 
	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
 
    # return the edged image
    return edged

canny = auto_canny(img)
plt.imshow(canny)

#%%
#

def dilate(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilated = cv2.morphologyEx(image,cv2.MORPH_DILATE,kernel)
    return dilated

plt.imshow(dilate(canny))
#%%

def subtract_invert(image,subtracted_inv):
    one_idx = subtracted_inv > 0
    zero_idx = subtracted_inv == 0
    subtracted_inv[zero_idx] = 1
    subtracted_inv[one_idx] = 0
    diff = cv2.subtract(image,subtracted_inv)
    return diff

plt.imshow(subtract_invert(binary,canny))
#%%

