
# local imports
from patch import img_patch
from img_load import load_training
from img_manipulation import convert_to_one_hot

# librairies imports
import numpy as np
import matplotlib.pyplot as plt

PATCH_SIZE = 80

nTrain = 90
im_width = 400
im_height = 400

root_dir = "../../project_files/data/"
training_dir = root_dir + "training/"

imgs, gt_imgs = load_training(training_dir, nTrain)


def patchesAndOverlap(imgs,patch_size, is_gt = False):
    """ Returns all patches concatenated in one big array and all overlap_images"""
    imgs_patches = []
    overlap_images = []
    for i in range(0,nTrain):
        img_patches,overlap_image = img_patch(imgs[i],patch_size)
        if(is_gt):
            img_patches = np.asarray(img_patches).reshape((img_patches.shape[0],patch_size,patch_size))
        # print(img_patches.shape)
        imgs_patches.append(img_patches)
        overlap_images.append(overlap_image)

    imgs_patches = np.asarray(imgs_patches)
    tmpShape = imgs_patches.shape

    if(not is_gt):
        imgs_patches = imgs_patches.reshape(tmpShape[0]*tmpShape[1],tmpShape[2],tmpShape[3],tmpShape[4])
    if(is_gt):
        imgs_patches = imgs_patches.reshape(tmpShape[0]*tmpShape[1],tmpShape[2],tmpShape[3])

    return imgs_patches,np.asarray(overlap_images)


imgs_patches,overlap_imgs = patchesAndOverlap(imgs,PATCH_SIZE)

gt_imgs = gt_imgs.reshape((nTrain,im_width,im_height,1))
gts_patches,overlap_gts = patchesAndOverlap(gt_imgs,PATCH_SIZE,is_gt = True)

plt.imshow(gts_patches[1200])

plt.imshow(imgs_patches[1200])
