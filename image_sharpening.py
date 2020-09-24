import numpy as np
import cv2
import os

# Note that sets of 3 values are BGR values - blue, green and red

########################################################################################################################
#                     Image Enhancement via Adaptive Unsharp Masking
########################################################################################################################
def calc_z_x(n, m, img_in):
    if m == 0:
        z_x = 2 * img_in[n][m] - img_in[n][m + 1]
    elif m == img_in.shape[1] - 1:
        z_x = 2 * img_in[n][m] - img_in[n][m - 1]
    else:
        z_x = 2 * img_in[n][m] - img_in[n][m - 1] - img_in[n][m + 1]

    return z_x

def calc_z_y(n, m, img_in):
    if n == 0:
        z_y = 2 * img_in[n][m] - img_in[n - 1][m] - img_in[n + 1][m]
    elif n == img_in.shape[0] - 1:
        z_y = 2 * img_in[n][m] - img_in[n - 1][m]
    else:
        z_y = 2 * img_in[n][m] - img_in[n - 1][m] - img_in[n + 1][m]

    return z_y

path = os.getcwd()+'\\motion-blur.jpg'
img_in = cv2.imread(path, 0)
img_out = np.zeros(shape = img_in.shape, dtype = img_in.dtype)
for n in range(img_in.shape[0]):
    for m in range(img_in.shape[1]):
        z_x = calc_z_x(n, m, img_in)
        z_y = calc_z_y(n, m, img_in)

        #img_out[n][m] = img_in[n][m]+lmbda_x[n][m]*z_x+lmbda_y[n][m]*z_y
        img_out[n][m] = img_in[n][m] + 2 * z_x + 2 * z_y

cv2.imshow('image', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()








