import math
from enum import auto, Enum, unique
from typing import Tuple, List, Type, Optional, TypeVar, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from skimage import measure
from skimage.measure._regionprops_utils import perimeter


def _snakeinterp1(x: np.ndarray, y: np.ndarray, res: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    % SNAKEINTERP1  Interpolate the snake to have equal distance RES
    %     [xi,yi] = snakeinterp(x,y,RES)
    %
    %     RES: resolution desired

    %     update on snakeinterp after finding a bug

    %      Chenyang Xu and Jerry L. Prince, 3/8/96, 6/17/97
    %      Copyright (c) 1996-97 by Chenyang Xu and Jerry L. Prince
    %      image Analysis and Communications Lab, Johns Hopkins University

    :param x:
    :param y:
    :param res:
    :return:
    """

    def uppertri(M: int, N: int, dtype: Optional[Type] = bool) -> np.ndarray:
        j, i = np.meshgrid(np.arange(M), np.arange(N))
        return (j >= i).astype(dtype)

    x: np.ndarray = np.append(x, x[0])
    y: np.ndarray = np.append(y, y[0])

    dx: np.ndarray = np.diff(x)
    dy: np.ndarray = np.diff(y)
    d: np.ndarray = np.append([0], np.sqrt(dx * dx + dy * dy))
    M: int = len(d)
    d: np.ndarray = d.dot(uppertri(M, M, float))
    maxd: float = d[-1]
    if maxd / res < 3:
        raise ArithmeticError('argument \'res\' is too big compare to the length of original curve')
    di: np.ndarray = np.arange(0, maxd, res)
    xi: np.ndarray = interpolate.interp1d(d, x)(di)
    yi: np.ndarray = interpolate.interp1d(d, y)(di)
    if maxd - di[-1] < res / 2:
        return xi[:-1], yi[:-1]
    return xi, yi


_T = TypeVar("_T")


def _filled(size: Tuple[int, ...], value=_T) -> np.ndarray:
    out = np.empty(size)
    out[:] = value
    return out


@unique
class _WindingRule(Enum):
    def __new__(cls, keycode: int, is_in: Callable[[int], bool]) -> object:
        obj = object.__new__(cls)
        obj._value_ = keycode
        obj.wn_is_in = is_in
        return obj

    EVEN_ODD = auto(), lambda x: (x % 2) == 1
    NON_ZERO = auto(), lambda x: x == 0


@unique
class _PointPolygonPosition(Enum):
    IN = auto()
    ON = auto()
    OUT = auto()


def _in_polygon(point: Tuple[float, float], xv: np.ndarray, yv: np.ndarray,
                rule: _WindingRule = _WindingRule.EVEN_ODD) -> _PointPolygonPosition:
    """
    Adapted from https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
    :param point:
    :param xv:
    :param yv:
    :return:
    """

    def is_left(P0: Tuple[float, float], P1: Tuple[float, float], P2: Tuple[float, float]) -> float:
        return ((P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1]))

    n: int = len(xv) - 1
    wn: int = 0
    for i in range(n):
        _wn: float = is_left((xv[i], yv[i]), (xv[i + 1], yv[i + 1]), point)
        if _wn == 0:
            return _PointPolygonPosition.ON
        if yv[i] <= point[1]:  # start y <= P.y
            if yv[i + 1] > point[1]:  # an upward crossing
                if _wn > 0:  # P left of  edge
                    wn += 1  # have  a valid up intersect
        else:  # start y > P.y (no test needed)
            if yv[i + 1] <= point[1]:  # a downward crossing
                if _wn < 0:  # P right of  edge
                    wn -= 1  # have  a valid down intersect

    return _PointPolygonPosition.IN if rule.wn_is_in(wn) else _PointPolygonPosition.OUT


def mean_neg_curvature(shape_XY: np.ndarray, shape_curvature: np.ndarray) -> float:
    M: int = shape_XY.shape[0]
    list_curve = shape_curvature[:M - 1]
    list_neg_curve = abs(list_curve[list_curve < 0])
    if len(list_neg_curve):
        return sum(list_neg_curve) / (M - 1)
    else:
        return 0


def num_indents(shape_XY: np.ndarray, shape_curvature: np.ndarray) -> int:
    M: int = shape_XY.shape[0]
    list_curve = shape_curvature[:M - 1]
    curve_mask = list_curve < 0
    curve_mask_labeled = measure.label(curve_mask)
    num_indents: int = np.max(curve_mask_labeled)
    if curve_mask[0] and curve_mask[-1]:
        num_indents -= 1
    return num_indents


def tortuosity(image: np.ndarray, shape_XY: np.ndarray, shape_curvature: np.ndarray):
    _perimeter: float = perimeter(image)
    M: int = shape_XY.shape[0]
    return sum(np.gradient(shape_curvature[:M - 1]) ** 2) / _perimeter


def shape_tangent_angles(shape_XY: np.ndarray, bp_tangent: int = 10) -> np.ndarray:
    M: int = shape_XY.shape[0]

    shape_tangent_angle: np.ndarray = _filled((M,), float("nan"))

    bp_positions_tangent = np.concatenate(
        (shape_XY[M - 1 - bp_tangent:M - 1, :], shape_XY[:M - 1, :], shape_XY[:bp_tangent + 1, :]))

    for j in range(M):
        point1 = bp_positions_tangent[j, :]
        point2 = bp_positions_tangent[j + 2 * bp_tangent, :]
        shape_tangent_angle[j] = np.mod(math.atan2(point1[1] - point2[1], point1[0] - point2[0]), np.pi)
    return shape_tangent_angle


def curvature(image: np.ndarray, boundary_point: int = 10, interp_resolution: float = .3,
              loop_close: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    output: List[Tuple[np.ndarray, np.ndarray]] = []
    boundaries: List[np.ndarray] = measure.find_contours(image)
    for contour in boundaries:
        x, y = contour[:, 1], contour[:, 0]
        xn: np.ndarray = x + np.random.rand(len(x)) * 1e-8
        yn: np.ndarray = y + np.random.rand(len(x)) * 1e-8

        xi, yi = _snakeinterp1(xn, yn, interp_resolution)
        xn = xi[::boundary_point]
        yn = yi[::boundary_point]
        if loop_close:
            xn = np.append(xn, xi[0])
            yn = np.append(yn, yi[0])
        shape_XY = np.asarray((xn, yn)).transpose()
        M: int = shape_XY.shape[0]
        shape_curvature: np.ndarray = _filled((M,), float("nan"))

        bp_positions = np.concatenate(
            (shape_XY[M - 1 - boundary_point:M - 1, :], shape_XY[:M - 1, :], shape_XY[:boundary_point + 1, :]))
        for j in range(M):
            point1 = bp_positions[j, :]
            point2 = bp_positions[j + boundary_point, :]
            point3 = bp_positions[j + 2 * boundary_point, :]
            slope12 = (point1[1] - point2[1]) / (point1[0] - point2[0])
            slope23 = (point2[1] - point3[1]) / (point2[0] - point3[0])
            if np.isinf(slope23) or slope12 == 0:
                point0 = point2
                point2 = point3
                point3 = point0
                slope12 = (point1[1] - point2[1]) / (point1[0] - point2[0])
                slope23 = (point2[1] - point3[1]) / (point2[0] - point3[0])

            if np.isinf(slope23):
                point0 = point1
                point1 = point2
                point2 = point0
                slope12 = (point1[1] - point2[1]) / (point1[0] - point2[0])
                slope23 = (point2[1] - point3[1]) / (point2[0] - point3[0])

            if slope12 == slope23:
                # boundary is flat
                shape_curvature[j] = 0
            else:
                # boundary is curved
                x_center = (slope12 * slope23 * (point1[1] - point3[1]) + slope23 * (
                        point1[0] + point2[0]) - slope12 * (point2[0] + point3[0])) / (2 * (slope23 - slope12))
                midpoint12 = (point1 + point2) / 2
                midpoint13 = (point1 + point3) / 2
                y_center = (-1 / slope12) * (x_center - midpoint12[0]) + midpoint12[1]
                shape_curvature[j] = 1 / np.sqrt(
                    (point1[0] - x_center) ** 2 + (point1[1] - y_center) ** 2)
                in_poly = _in_polygon((midpoint13[0], midpoint13[1],), shape_XY[:, 0], shape_XY[:, 1])
                if in_poly is _PointPolygonPosition.OUT:
                    shape_curvature[j] = - shape_curvature[j]

                if in_poly is _PointPolygonPosition.ON or np.isinf(shape_curvature[j]):
                    shape_curvature[j] = 0
        output.append((shape_XY, shape_curvature))
    return output


if __name__ == "__main__":
    from skimage.io import imread

    img: np.ndarray = imread("blob.png")[..., 0] > 0
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="Greys_r")
    for shape_XY, shape_curvature in curvature(img):
        plt.scatter(shape_XY[:, 0], shape_XY[:, 1], s=2,
                    c=shape_curvature, cmap="Spectral_r")
        plt.show()
