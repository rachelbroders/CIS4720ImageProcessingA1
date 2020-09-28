import cv2
import image_sharpening as sharp
import numpy as np
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

path = os.getcwd()+'\\motion-blur.jpg'
img_in = cv2.imread(path, 0)
filter = sharp.adaptiveUnsharpMasking(img_in = img_in,
                             tau_1 = 60, tau_2 = 200,
                             alpha_dh_1 = 4, alpha_dh_2 = 3,
                             mu = 0.1, beta = 0.5)
blockPrint()
img_out = filter.main()


cv2.imshow('image', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

alpha = np.zeros(shape = img_in.shape, dtype = img_in.dtype)
g = np.zeros(shape = img_in.shape, dtype = img_in.dtype)
g_z_x = np.zeros(shape = img_in.shape, dtype = img_in.dtype)
g_z_y = np.zeros(shape = img_in.shape, dtype = img_in.dtype)
g_d = np.zeros(shape = img_in.shape, dtype = img_in.dtype)
g_y = np.zeros(shape = img_in.shape, dtype = img_in.dtype)

g_y = filter.apply_g_y(479, 639)
for i in range(img_in.shape[0]):
    for j in range(img_in.shape[1]):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ", i, ", ", j, " ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #alpha[i, j] = filter.calc_alpha(i, j)
        #g[i, j] = filter.apply_g(i, j, filter.direct)
        #g_z_x[i, j] = filter.apply_g(i, j, filter.calc_z_x)
        #g_z_y[i, j] = filter.apply_g(i, j, filter.calc_z_y)
        #g_d[i, j] = filter.apply_g_d(i, j)
        blockPrint()
        g_y[i, j] = filter.apply_g_y(i, j)
        enablePrint()
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ", g_y[i, j], " ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")






