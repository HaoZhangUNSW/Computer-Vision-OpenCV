from scipy import *
from pylab import *

img = imread("img/me1.jpg")[:, :, 0]

gray()
figure(1)
imshow(img)
print("original size:" + str(img.shape[0] * img.shape[1]))
m, n = img.shape
U, S, Vt = svd(img)
S = resize(S, [m, 1])*eye(m,n)

k = 10
figure(2)
imshow(dot(U[:,1:k], dot(S[1:k, 1:k], Vt[1:k, :])))
show()
size = m * k + k + k * n
print("compress size:" + str(size))
