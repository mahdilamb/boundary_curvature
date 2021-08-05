import numpy as np
from matplotlib import pyplot as plt

from boundary_curvature.curvature import curvature

if __name__ == "__main__":
    from skimage.io import imread

    img: np.ndarray = imread("blob.png")[..., 0] > 0
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="Greys_r")
    for shape_XY, shape_curvature in curvature(img):
        plt.scatter(shape_XY[:, 0], shape_XY[:, 1], s=2,
                    c=shape_curvature, cmap="Spectral_r")
        plt.show()
