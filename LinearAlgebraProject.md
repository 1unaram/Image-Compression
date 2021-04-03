# 1. Write a computer program that performs the following:
#### (a) Take a grayscale image and form a matrix A. Each entry in A is a pixel in the image, and the size of matrix A is equal to the resolution of the image. If needed, you might want to chop off the image and make A an n × n matrix.
#### (b) For given n = 2^t, t = 0, 1, ..., construct an n-point Haar matrix H in the lecture. Every column in Hn should be normalized, so that H^(−1) = H^(T).
#### (c) Perform 2-D Discrete Haar Wavelet Transform (DHWT) B = H^(T)AH.
#### (d) For given k = 2^s, s = 0, 1, ..., construct an n × n matrix Bˆ such that {its k × k upper left corner is equal to the k × k upper left corner of B / all the other entries are zero.} So if k = n, Bˆ = B.
#### (e) Perform the inverse 2-D Discrete Haar Wavelet Transform (IDHWT) Aˆ = HBHˆ H^(T).
#### (f) Show the reconstructed image Aˆ on the screen.

### • CODE(1)
```python
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

* * *

# 2. Get any 2 grayscale images of any format from anywhere (Internet, your personal photos, etc.). It is recommended that you get one image with low frequency components, and one image filled with high frequency components. Use your computer program to do the following:
#### (a) As k increases, observe the quality of reconstructed image.
#### (b) Describe any difference between low-freq. image and high-freq. image.
#### (c) Discuss any findings or thoughts.

### • CODE(2)
```python
import cv2
import numpy as np
import math


# ImageFile load
lowFreqImageFile = '../image/sky_grayscale.jpg'
highFreqImageFile = '../image/taran_grayscale.jpg'
lowFreqImgMat = cv2.imread(lowFreqImageFile, 0) / 255
highFreqImgMat = cv2.imread(highFreqImageFile, 0) / 255
n, n = lowFreqImgMat.shape


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


# HaarMatrix normalize function
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


HaarMat = normalize(dHaarMatrix(n), n)
# Observe the quality of reconstructed image (low-freq components)
for s in range(int(math.log2(n)/math.log2(2)) + 1):
    B = getB(lowFreqImgMat, HaarMat)
    BHat = getBHat(B, s, n)
    AHat = getAHat(BHat, HaarMat)

    originalImg = AHat * 255
    cv2.imwrite('../image/lowResult/Reconstructed-' + str(2**s) + 'by' + str(2**s) + '.jpg', originalImg)


# Observe the quality of reconstructed image (high-freq components)
for s in range(int(math.log2(n)/math.log2(2)) + 1):
    B = getB(highFreqImgMat, HaarMat)
    BHat = getBHat(B, s, n)
    AHat = getAHat(BHat, HaarMat)

    originalImg = AHat * 255
    cv2.imwrite('../image/highResult/Reconstructed-' + str(2**s) + 'by' + str(2**s) + '.jpg', originalImg)
```

### • Main Code Description
|Line|Description|
|:-----:|:----:|
|9-10|cv2.imread로 불러온 이미지 파일 행렬 각 요소를 255로 나눠줌으로 정규화|
|14-24|denormalized Haar Matrix를 반환하는 함수로써, 크로네커 곱을 수행하는 np.kron() 연산과 단위행렬을 만드는 np.identity(), 두 행렬을 연결하는 np.hstack(), 행렬 요소를 실수형으로 반환하는 dtype=float을 사용하여 구성되었으며 재귀 함수로 구성|
|27-40|Denormalized Haar Matrix의 각 column의 길이가 1이 되도록 정규화|
|43-45|DHWT을 수행하여 B를 반환하는 함수로, np.dot() 연산으로 행렬 곱 수행|
|48-60|B의 kxk upper left corner를 제외한 나머지 요소를 0으로 바꾸어 B hat을 반환하는 함수|
|69-86|반복문 for로 2^s가 n이 될 때까지 s값을 0부터 1씩 증가됨에 따라 k값이 증가하고 이에 재구성된 이미지를 cv2.imwrite()로 저장하는 구문. Line9, 10에서 이미지를 255로 나누어 정규화 해주었으므로 여기서 255를 다시 스칼라 곱을 해주어 원본 이미지 행렬을 만들도록 함|

### • (a) As k increases, observe the quality of reconstructed image.
: high-freq. image 의 재구성된 이미지를 ./LAproject/image/highResult 에, lowfreq. image 의 재구성된 이미지를 ./LAproject/image/lowResult 에 저장하도록 하였음.

#### <Image filled with high frequency components>
![image](https://user-images.githubusercontent.com/37824335/113482454-cf336c80-94d9-11eb-94dd-1f294de926e6.png)

