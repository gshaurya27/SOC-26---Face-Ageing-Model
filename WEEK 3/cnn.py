import numpy as np
import matplotlib.pyplot as plt

# ====================== Common Kernels ======================
kernels = {
    "Identity": np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]),
    
    "Edge_Laplacian": np.array([[ 0, -1,  0],
                                [-1,  4, -1],
                                [ 0, -1,  0]]),
    
    "Sobel_Horizontal": np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]),
    
    "Sobel_Vertical": np.array([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]]),
    
    "Sharpen": np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]]),
    
    "Gaussian_Blur": np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]]) / 16.0,
    
    "Emboss": np.array([[-2, -1, 0],
                        [-1,  1, 1],
                        [ 0,  1, 2]])
}

# ====================== YOUR IMPLEMENTATIONS ======================

def convolve2d(image, kernel, stride=1, padding=0):
    
    s=stride
    p=padding
    if p > 0:
        image = np.pad(image, pad_width=p, mode='constant', constant_values=0)  

    H,W = image.shape
    KH, KW = kernel.shape
    OH = (H - KH ) // s + 1
    OW = (W - KW ) // s + 1
    output = np.zeros((OH, OW))

    for i in range(OH):
        for j in range(OW):
            vert_start = i * s
            vert_end = vert_start + KH
            horiz_start = j * s
            horiz_end = horiz_start + KW

            mat = image[vert_start:vert_end, horiz_start:horiz_end]
            output[i, j] = np.sum(mat * kernel)
    return output


def max_pool2d(image, pool_size=2, stride=2):

    s = stride
    p = pool_size
    H,W = image.shape
    PH, PW = p, p
    OH = ((H - PH) // s) + 1
    OW = ((W - PW) // s) + 1
    output = np.zeros((OH, OW))
    
    for i in range(OH):
        for j in range(OW):
            vert_start = i * s
            vert_end = vert_start + PH
            horiz_start = j * s
            horiz_end = horiz_start + PW

            mat = image[vert_start:vert_end, horiz_start:horiz_end]
            output[i, j] = np.max(mat)
    return output

def avg_pool2d(image, pool_size=2, stride=2):
    
    s = stride
    p = pool_size
    H,W = image.shape
    PH, PW = p, p
    OH = ((H - PH) // s) + 1
    OW = ((W - PW) // s) + 1
    output = np.zeros((OH, OW))

    for i in range(OH):
        for j in range(OW):
            vert_start = i * s
            vert_end = vert_start + PH
            horiz_start = j * s
            horiz_end = horiz_start + PW

            mat = image[vert_start:vert_end, horiz_start:horiz_end]
            output[i, j] = np.mean(mat)
    return output
