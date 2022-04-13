import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

img =cv.imread('./fft.jpg',0)#直接用cv库读取灰度图像
img2 =plt.imread('./fft.jpg')
plt.subplot(231), plt.imshow(img2), plt.title('原图')
# 进行傅立叶变换，并显示结果
plt.subplot(232), plt.imshow(img,cmap='gray'), plt.title('灰度图')
fft2 = np.fft.fft2(img)#fft2 就是fft在2—dimension的范围
plt.subplot(233), plt.imshow(np.abs(fft2), 'gray'), plt.title('fft')

# 将图像变换的原点移动到频域矩形的中心，并显示效果
shift2center = np.fft.fftshift(fft2)
plt.subplot(234), plt.imshow(np.abs(shift2center), 'gray'), plt.title('shift2center')

# 对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(235), plt.imshow(log_fft2, 'gray'), plt.title('log_fft')

# 对中心化后的结果进行对数变换，并显示结果
log_shift2center = np.log(1 + np.abs(shift2center))
plt.subplot(236), plt.imshow(log_shift2center, 'gray'), plt.title('log_shift2center')
plt.show()