# 1.Write a computer program that performs the following:
#### (a) Take a grayscale image and form a matrix A. Each entry in A is a pixel in the image, and the size of matrix A is equal to the resolution of the image. If needed, you might want to chop off the image and make A an n × n matrix.
#### (b) For given n = 2^t, t = 0, 1, ..., construct an n-point Haar matrix H in the lecture. Every column in Hn should be normalized, so that H^(−1) = H^(T).
#### (c) Perform 2-D Discrete Haar Wavelet Transform (DHWT) B = H^(T)AH.
#### (d) For given k = 2^s, s = 0, 1, ..., construct an n × n matrix Bˆ such that {its k × k upper left corner is equal to the k × k upper left corner of B / all the other entries are zero.} So if k = n, Bˆ = B.
#### (e) Perform the inverse 2-D Discrete Haar Wavelet Transform (IDHWT) Aˆ = HBHˆ H^(T).
#### (f) Show the reconstructed image Aˆ on the screen.
