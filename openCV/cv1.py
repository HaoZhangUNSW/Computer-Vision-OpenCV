from scipy import *
from pylab import *
import numpy as np
import cv2 as cv2
# image = imread("img/me1.jpg")[:, :, 0]
image = cv2.imread("img//test.jpg")
print(f'width: {image.shape[1]} pixels')

print(f'height: {image.shape[0]} pixels')

print(f'channels: {image.shape[2]}')

cv2.imshow('image', image)
cv2.waitKey(0)
# cv2.imwrite('new_image.png', image)
# patch1 = image[:100, :100]
# cv2.imshow("patch1", patch1)
# cv2.waitKey(0)
canvas = np.zeros((300,300,3), dtype="uint8")
# print(canvas)
for _ in range(0, 25):
    r = np.random.randint(5, 200)
    color = np.random.randint(0, 256, size = (3, )).tolist()
    pt = np.random.randint(0, 200, size = (2, ))

    cv2.circle(canvas, tuple(pt), r, color, -1)

cv2.imshow('canvas', canvas)
cv2.waitKey(0)
#shift position to right down
# M = np.float32([[1, 0, 25], [0, 1, 50]])
# shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
# cv2.imshow('shifted_image', shifted_image)
# cv2.waitKey(0)

#rotate
# (h, w) = image.shape[:2]
# center = (w // 2, h // 2)
#
# M = cv2.getRotationMatrix2D(center, 135, 1.0)
# Rotated_iamge = cv2.warpAffine(image, M, (w, h))
# cv2.imshow('Rotated_iamge', Rotated_iamge)
# cv2.waitKey(0)
# new_w, new_h = 100, 200
# resize = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
# cv2.imshow('resize', resize)
# cv2.waitKey(0)

# flip
# flipped_image = cv2.flip(image, -1)
# cv2.imshow('flipped_image', flipped_image)
# cv2.waitKey(0)

# 位运算
# rectangle = np.zeros((100, 100), dtype='uint8')
# cv2.rectangle(rectangle, (30, 30), (70, 70), 255, -1)
# cv2.imshow('rectangle', rectangle)
# cv2.waitKey(0)
#
# circle = np.zeros((100, 100), dtype='uint8')
# cv2.circle(circle, (50, 50), 25, 255, -1)
# cv2.imshow('circle', circle)
# cv2.waitKey(0)

# bitwiseAnd = cv2.bitwise_and(rectangle, circle)
# cv2.imshow('And', bitwiseAnd)
# cv2.waitKey(0)
#
# bitwiseOr = cv2.bitwise_or(rectangle, circle)
# cv2.imshow('And', bitwiseOr)
# cv2.waitKey(0)
#
# bitwiseXOR = cv2.bitwise_xor(rectangle, circle)
# cv2.imshow('And', bitwiseXOR)
# cv2.waitKey(0)
#
# bitwiseNot = cv2.bitwise_not(circle)
# cv2.imshow('And', bitwiseNot)
# cv2.waitKey(0)

#masking
# mask = np.zeros(image.shape[:2], dtype='uint8')
# (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
# cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75, cY + 75), 255, -1)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
#
# masked = cv2.bitwise_and(image, image, mask = mask)
# cv2.imshow('masked image', masked)
# cv2.waitKey(0)

# split RGB channels and merge
# (B, G, R) = cv2.split(image)
# merged = cv2.merge([B, G, R])
# cv2.imshow('RED', R)
# cv2.imshow('Blue', B)
# cv2.imshow('Green', G)
# cv2.imshow('Merged', merged)
# cv2.waitKey(0)

# change color to gray/HSV/LAB
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# cv2.imshow('gray', gray)
# cv2.imshow('HSV', HSV)
# cv2.imshow('LAB', LAB)
# cv2.waitKey(0)

# colorful plt hist
# from matplotlib import pyplot as plt
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.figure()
#
# p1 = plt.subplot(121)
# p2 = plt.subplot(122)
#
# p1.plot(hist)
# chans = cv2.split(image)
# colors = ('b', 'g', 'r')
# for (chan, color) in zip(chans, colors):
#     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#     p2.plot(hist, color=color)
# plt.show()

# blur 平滑 模糊
# blurred = np.hstack([cv2.blur(image, (3, 3)), cv2.blur(image, (5, 5)), cv2.blur(image, (7, 7))])
# cv2.imshow('Averaged', blurred)
#
# blurred = np.hstack([cv2.GaussianBlur(image, (3, 3), 0), cv2.GaussianBlur(image, (5, 5), 0), cv2.GaussianBlur(image, (7, 7), 0)])
# cv2.imshow('GaussianBlur', blurred)
#
# blurred = np.hstack([cv2.medianBlur(image, 3), cv2.medianBlur(image, 5), cv2.medianBlur(image, 7)])
# cv2.imshow('Median', blurred)
#
# blurred = np.hstack([cv2.bilateralFilter(image, 5, 21, 21), cv2.bilateralFilter(image, 7, 31, 31), cv2.bilateralFilter(image, 9, 41, 41)])
# cv2.imshow('Bilateral', blurred)
# cv2.waitKey(0)

# 边缘检测
# blured = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blured = cv2.GaussianBlur(blured, (5, 5), 0)
# cv2.imshow('blurred', blured)
#
# canny = cv2.Canny(blured, 30, 150)
# cv2.imshow('canny', canny)
# cv2.waitKey(0)

# openCV for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
faces_rects = faceCascade.detectMultiScale(image, scaleFactor=1.02, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
for (x, y, w, h) in faces_rects:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eyes_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
cv2.imshow('iamge', image)
cv2.imwrite('new_test.jpg', image)
cv2.waitKey(0)
