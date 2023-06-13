### Author: Gary Chen
### Date: June 13, 2023
### Description: Radon Transform (img to sinogram)
###     There's still some bug; 
###     The sinograms look like a sinograms but the second col arent the same.
try:
    import numpy as np
except Exception as e:
    print("numpy Importing exception")
try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib Importing exception")
try:
    import cv2
except Exception as e:
    print("opencv Importing exception")

img_id = 1 # adjust this id from 0 to 50 for any images in the dataset
# Load the .npy file
imamges = np.load('foam_training.npy')
print("images shape", imamges.shape)
image1 = imamges[img_id,:,:]
print("image shape", image1.shape)
#plt.imshow(image1)#, cmap='gray')  # remove cmap='gray' if your image is colored
#plt.show()

sinograms = np.load('x_train_sinograms.npy')
print("sinograms shape", sinograms.shape)
sinogram1 = sinograms[img_id,:,:]
print("sinogram shape", sinogram1.shape)
#plt.imshow(sinogram1)#, cmap='gray')  # remove cmap='gray' if your image is colored
#plt.show()

#hyperparameters
t_num = 184 # pixel number on the discretized detector
theta_num = 180 # number of projections
size_x = image1.shape[0]
size_y = image1.shape[1]

#initialize sinogram as a black img
mysinogram = np.zeros((180, 184))

# initialize the collection of rotated iamges as 180 by 128 by 128 empty arrays
rotated_imgs = np.zeros((theta_num,128, 128))
# assign the unrotated img as the first img in the rotated collection
rotated_imgs[0,:,:] = image1
# image0 = image1.copy()
for i in range(1,theta_num):
    (h, w) = image1.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, i, 1.0)  # 45 degree rotation around the center
    rotated = cv2.warpAffine(image1, M, (w, h))
    rotated_imgs[i,:,:] = rotated

# initialize the summing vector
sum_vec = np.ones((size_x, 1))

for i in range(theta_num):
    # ith_p is the i-th row of the sinogram corresponding to each angle
    ith_p = rotated_imgs[i,:,:]@sum_vec/180 # 128 by 128 @ 128 by 1 means sum across the all rows of the img
    #print("p function shape is", ith_p.shape) 
    # Calculate start index for ith row in the sinogram 
    start = (mysinogram.shape[0] - ith_p.shape[0]) // 2
    # Assign p values to the ith row of the sinogram, centered
    mysinogram[i,start:start+ith_p.shape[0]] = ith_p.flatten()

# Display images for radon transform

# create figure for intermediate visualization
rotate_fig, ax = plt.subplots(1, 3, figsize=(10, 6))
ax[0].imshow(rotated_imgs[0,:,:], cmap='gray')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title(f'img {img_id} rotated 0')
ax[1].imshow(rotated_imgs[90,:,:], cmap='gray')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title(f'img {img_id} rotated 90')
ax[2].imshow(rotated_imgs[179,:,:], cmap='gray')
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title(f'img {img_id} rotated 179')

# Create a figure and a 1x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(6, 6))

axs[0,0].imshow(image1, cmap='gray')  # remove cmap='gray' if your image is colored
axs[0,0].set_title(f'TF Image {img_id}')
axs[0,1].imshow(sinogram1, cmap='gray')  # remove cmap='gray' if your image is colored
axs[0,1].set_title(f'TF sinogram {img_id}')
axs[1,0].imshow(image1, cmap='gray')  # remove cmap='gray' if your image is colored
axs[1,1].imshow(mysinogram, cmap='gray')  # remove cmap='gray' if your image is colored
axs[1,0].set_title(f'Our Image {img_id}')
axs[1,1].set_title(f'Our sinogram {img_id}')

# Remove the x and y ticks
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])

# Display the figure
fig.savefig("radon comp")
rotate_fig.savefig("rotate images")

plt.show()



