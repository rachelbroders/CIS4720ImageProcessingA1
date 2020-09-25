import cv2
import os
import image_sharpening as sharp

path = os.getcwd()+'\\motion-blur.jpg'
img_in = cv2.imread(path, 0)
filter = sharp.adaptiveUnsharpMasking(img_in = img_in,
                             tau_1 = 60, tau_2 = 200,
                             alpha_dh_1 = 4, alpha_dh_2 = 3,
                             mu = 0.1, beta = 0.5)
img_out = filter.main()


cv2.imshow('image', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()





