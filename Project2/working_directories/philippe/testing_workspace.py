#%%
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys

from PIL import Image

sys.path.append('../../project_files/scripts/')
import segment_aerial_images as sai
import cv2

#%%
# Loaded a set of images
root_dir = "../../project_files/data/training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = min(98, len(files)) # Load maximum 20 images
print("Loading " + str(n) + " images")
imgs = [cv2.imread(image_dir + files[i]) for i in range(n)]
print(files[0])

#%%
######## TESTS ########
import image_processing as ip 

bin_imgs = imgs.copy()

for i,_ in enumerate(imgs):
    binary = ip.binarize(imgs[i])
    canny = ip.auto_canny(imgs[i])
    bin_imgs[i] = ip.subtract_invert(binary,canny)
    bin_imgs[i] =  (bin_imgs[i]-np.mean(bin_imgs[i])) / np.std(bin_imgs[i])

plt.imshow(bin_imgs[0])
plt.show()

#####################
#%%
gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [cv2.imread(gt_dir + files[i]) for i in range(n)]
print(files[0])

n = 98 # Only use 50 images for training

#%%
print('Image size = ' + str(imgs[0].shape[0]) + ',' + str(imgs[0].shape[1]))

# Show first image and its groundtruth image
plt.imshow(imgs[0])
plt.show()
plt.imshow(gt_imgs[0])
plt.show()

#%%
# Extract patches from input images
patch_size = 16 # each patch is 16*16 pixels

bin_img_patches = [sai.img_crop(bin_imgs[i], patch_size, patch_size) for i in range(n)]
img_patches = [sai.img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
gt_patches = [sai.img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

# Linearize list of patches
bin_img_patches = np.asarray([bin_img_patches[i][j] for i in range(len(bin_img_patches)) for j in range(len(bin_img_patches[i]))])
img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    
#%%
# Compute features for each image patch

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch


def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def extract_features_with_bin(image,binary):
    feat_m = np.mean(image)
    feat_v = np.var(image)
    feat_m_bin = np.mean(binary)
    feat_v_bin = np.var(binary)
    feat = np.append(feat_m, feat_v)
    feat = np.append(feat,feat_m_bin)
    feat = np.append(feat,feat_v_bin)
    return feat

X = np.asarray([ extract_features_with_bin(img_patches[i],bin_img_patches[i]) for i in range(len(img_patches))])
Y = np.asarray([ value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])


print(X)
# Print feature statistics

print('Computed ' + str(X.shape[0]) + ' features')
print('Feature dimension = ' + str(X.shape[1]))
print('Number of classes = ' + str(np.max(Y)))  #TODO: fix, length(unique(Y)) 

Y0 = [i for i, j in enumerate(Y) if j == 0]
Y1 = [i for i, j in enumerate(Y) if j == 1]
print('Class 0: ' + str(len(Y0)) + ' samples')
print('Class 1: ' + str(len(Y1)) + ' samples')


#%%
# Display a patch that belongs to the foreground class
plt.imshow(gt_patches[Y1[3]], cmap='Greys_r')


#%%
# Plot 2d features using groundtruth to color the datapoints
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)


#%%
# train a logistic regression classifier

from sklearn import linear_model


# we create an instance of the classifier and fit the data
logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
logreg.fit(X, Y)


#%%
# Predict on the training set
Z = logreg.predict(X)

# Get non-zeros in prediction and grountruth arrays
Zn = np.nonzero(Z)[0]
Yn = np.nonzero(Y)[0]

TPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
print('True positive rate = ' + str(TPR))


#%%
# Plot features using predictions to color datapoints
plt.scatter(X[:, 0], X[:, 1], c=Z, edgecolors='k', cmap=plt.cm.Paired)


#%%
# Run prediction on the img_idx-th image
img_idx = 40


# Extract features for a given image
def extract_img_features(filename):
    img = cv2.imread(filename)
    bin_img = ip.binarize(img)
    bin_img =  (bin_img-np.mean(bin_img)) / np.std(bin_img)
    bin_img_patches = sai.img_crop(bin_img, patch_size, patch_size)
    img_patches = sai.img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features_with_bin(img_patches[i],bin_img_patches[i]) for i in range(len(img_patches))])
    return X

Xi = extract_img_features(image_dir + files[img_idx])
Zi = logreg.predict(Xi)
plt.scatter(Xi[:, 0], Xi[:, 1], c=Zi, edgecolors='k', cmap=plt.cm.Paired)


#%%
# Display prediction as an image
ref_img = cv2.imread(image_dir + files[img_idx])
gt_ref_img = cv2.imread(gt_dir + files[img_idx])
w = ref_img.shape[0]
h = ref_img.shape[1]
predicted_im = sai.label_to_img(w, h, patch_size, patch_size, Zi)
plt.imshow(predicted_im)
plt.show()
plt.imshow(ref_img)
plt.show()
plt.imshow(gt_ref_img)
plt.show()

#%%