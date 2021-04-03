# 1.Write a computer program that performs the following:
#### (a) Take a grayscale image and form a matrix A. Each entry in A is a pixel in the image, and the size of matrix A is equal to the resolution of the image. If needed, you might want to chop off the image and make A an n × n matrix.
#### (b) For given n = 2^t, t = 0, 1, ..., construct an n-point Haar matrix H in the lecture. Every column in Hn should be normalized, so that H^(−1) = H^(T).
#### (c) Perform 2-D Discrete Haar Wavelet Transform (DHWT) B = H^(T)AH.
#### (d) For given k = 2^s, s = 0, 1, ..., construct an n × n matrix Bˆ such that {its k × k upper left corner is equal to the k × k upper left corner of B / all the other entries are zero.} So if k = n, Bˆ = B.
#### (e) Perform the inverse 2-D Discrete Haar Wavelet Transform (IDHWT) Aˆ = HBHˆ H^(T).
#### (f) Show the reconstructed image Aˆ on the screen.

### • CODE(1)
```
import cv2
import numpy as np
import math
# ImageFile load
imageFile = '../image/taran_grayscale.jpg'
imgMat = cv2.imread(imageFile, 0) / 255
n, n = imgMat.shape
# Denormalized_HaarMatrix return function
def dHaarMatrix(n):
 if n == 1:
 return np.array([1.0])
 else:
 hm = np.kron(dHaarMatrix(int(n / 2)), ([1], [1]))
 hi = np.kron(np.identity(int(n / 2), dtype=int), ([1], [-1]))
 h = np.hstack([hm, hi])
 h = np.array(h, dtype=float)
 return h
# Normalize HaarMatrix function
def normalize(h, n):
 for i in range(0, n):
 temp = 0
 j = 0
 for j in range(0, n):
 if h[j][i] != 0:
 temp += 1
 if temp != 0:
 h[:, i] = 1 / math.sqrt(temp) * h[:, i]
 return h
# Discrete Haar Wavelet Transform return function
def getB(imgMat, HaarMat):
 return np.dot(np.dot(HaarMat.T, imgMat), HaarMat)
# B hat return function
def getBHat(B, s, n):
 k = 2**s
 for i in range(k, n):
 for j in range(0, k):
 B[i, j] = 0
 for i in range(0, n):
 for j in range(k, n):
 B[i, j] = 0
 return B
# A Hat return function
def getAHat(BHat, HaarMat):
 return np.dot(np.dot(HaarMat, BHat), HaarMat.T)
s = int(input("s:"))
HaarMat = normalize(dHaarMatrix(n), n)
B = getB(imgMat, HaarMat)
BHat = getBHat(B, s, n)
AHat = getAHat(BHat, HaarMat)
cv2.imshow('test', AHat)
cv2.waitKey(0)
```
### • How to Work(1)
: 사용자에게 s 값을 입력 받아 B hat 을 생성할 때 사용할 k 를 지정해주었고, 이에 따라 이미지를 윈도우에 보여주도록 함.
