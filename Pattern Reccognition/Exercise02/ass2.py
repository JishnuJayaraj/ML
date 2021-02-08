import matplotlib.pyplot as plt
import numpy as np


face = scipy.misc.face(gray = True)


face_gauss = scipy.ndimage.gaussian_filter(face,3)


plt.imshow(face_gauss, cmap = 'gray')